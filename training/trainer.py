import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchmetrics.classification import Precision, Recall
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib.pyplot as plt

from training.losses import smooth_labels, r1_regularization
from training.metrics import ms_ssim_per_class
from training.schedulers import unfreeze_vit_last_blocks

# ---------------------------------------------------------------------
# Device & SDR backends
# ---------------------------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# Optional: tweak scaled-dot product attention backend for stability
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

# ---------------------------------------------------------------------
# ImageNet normalization helpers
# ---------------------------------------------------------------------
imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)


def normalize_imagenet(x01: torch.Tensor) -> torch.Tensor:
    """Inputs in [0,1] → ImageNet normalized."""
    return (x01 - imagenet_mean) / imagenet_std


def to_imagenet_norm_from_tanh(x_minus1_1: torch.Tensor) -> torch.Tensor:
    """
    Generator outputs in [-1,1] → [0,1] → ImageNet normalized.
    """
    x01 = (x_minus1_1 + 1.0) / 2.0
    return normalize_imagenet(x01)


def denormalize_imagenet(x_norm: torch.Tensor) -> torch.Tensor:
    """Inverse of ImageNet norm → clamp to [0,1]."""
    return (x_norm * imagenet_std + imagenet_mean).clamp(0.0, 1.0)


# ---------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------
def show_generated_samples(generator, device, epoch=None, save_dir="results/generated_samples"):
    """
    Shows and saves a 2x2 grid: one generated image for each of 4 classes.
    Assumes classes: [glioma, meningioma, notumor, pituitary].
    """
    class_names = ["glioma", "meningioma", "notumor", "pituitary"]
    num_classes = len(class_names)

    generator.eval()
    with torch.no_grad():
        labels = torch.arange(num_classes, device=device)
        z = torch.randn(num_classes, generator.z_dim, device=device)
        fake_imgs = generator(z, labels)
        fake_imgs = (fake_imgs + 1) / 2.0  # [-1,1] → [0,1]

    os.makedirs(save_dir, exist_ok=True)
    if epoch is not None:
        save_image(fake_imgs, f"{save_dir}/epoch_{epoch:03d}.png", nrow=2)

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    axes = axes.flatten()
    for i, (img, label) in enumerate(zip(fake_imgs, labels)):
        ax = axes[i]
        ax.imshow(img.cpu().permute(1, 2, 0))
        ax.set_title(class_names[label.item()], fontsize=12)
        ax.axis("off")

    plt.suptitle(
        f"Generated Samples at Epoch {epoch}"
        if epoch is not None else "Generated Samples",
        fontsize=14
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    generator.train()


# ---------------------------------------------------------------------
# Discriminator optimizer builder
# ---------------------------------------------------------------------
def build_discriminator_optimizer(
    discriminator,
    base_lr=1e-4,
    vit_lr=2e-5,
    betas=(0.5, 0.999),
    use_adamw=False
):
    """
    Builds an optimizer with separate LR groups:
      - one for CNN / other params
      - one for ViT params
    """
    Optim = optim.AdamW if use_adamw else optim.Adam

    vit_params, other_params = [], []
    for n, p in discriminator.named_parameters():
        if not p.requires_grad:
            continue
        if n.startswith("vit."):
            vit_params.append(p)
        else:
            other_params.append(p)

    param_groups = []
    if other_params:
        param_groups.append({"params": other_params, "lr": base_lr, "betas": betas})
    if vit_params:
        param_groups.append({"params": vit_params, "lr": vit_lr, "betas": betas})

    return Optim(param_groups)


# ---------------------------------------------------------------------
# Full training loop
# ---------------------------------------------------------------------
def full_train_val(
    generator,
    discriminator,
    train_loader,
    val_loader,
    epochs=100,
    save_path=None,
    base_glr=1e-4,
    base_dlr=4e-4,
    vit_lr=2e-5,
    betas=(0.5, 0.999),
    gamma_r1=5.0,
    label_smooth=0.05,
    start_partial_epoch=4,
    partial_blocks=2,
    start_full_epoch=10,
    full_blocks=4,
    use_metric_trigger=True,
    plateau_tol=0.01,
    max_val_batches=None
):
    """
    Full training loop with:
      - BCE-with-logits adversarial objective
      - AC-GAN classification head
      - R1 regularization
      - Label smoothing
      - Progressive ViT unfreezing
      - FID/KID/MS-SSIM, precision/recall monitoring
    """

    generator.to(device)
    discriminator.to(device)

    adversarial_loss = nn.BCEWithLogitsLoss()
    classification_loss = nn.CrossEntropyLoss()

    optimizer_D = build_discriminator_optimizer(
        discriminator,
        base_lr=base_dlr,
        vit_lr=vit_lr,
        betas=betas
    )
    optimizer_G = optim.Adam(generator.parameters(), lr=base_glr, betas=betas)

    history = {
        "train_loss": [], "val_loss": [],
        "train_prec": [], "val_prec": [],
        "train_rec": [], "val_rec": [],
        "fid": [], "kid": [], "ms_ssim_mean": [],
        "train_total": [], "val_total": [],
        "train_d_real": [], "train_d_fake": [],
        "train_d_c_real": [], "train_d_c_fake": [],
        "train_r1": [],
        "train_g_adv": [], "train_g_cls": [],
        "val_d_real": [], "val_d_fake": [],
        "val_d_c_real": [], "val_d_c_fake": [],
        "val_g_adv": [], "val_g_cls": []
    }

    fid_metric = FrechetInceptionDistance(feature=2048).to(device)
    kid_metric = KernelInceptionDistance(subset_size=500).to(device)
    ms_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    best_val_fid = float("inf")
    vit_stage = "frozen"  # "frozen" → "partial" → "full"

    def metrics_plateaued():
        if len(history["val_prec"]) < 2 or len(history["val_rec"]) < 2:
            return False
        dp = abs(history["val_prec"][-1] - history["val_prec"][-2])
        dr = abs(history["val_rec"][-1] - history["val_rec"][-2])
        return (dp < plateau_tol) and (dr < plateau_tol)

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        generator.train()
        discriminator.train()

        # --- Train epoch accumulators ---
        d_real_sum = d_fake_sum = 0.0
        d_c_real_sum = d_c_fake_sum = 0.0
        r1_sum = g_adv_sum = g_cls_sum = 0.0
        steps = 0

        train_losses = []
        train_total_losses = []

        precision = Precision(num_classes=4, task="multiclass", average="macro").to(device)
        recall = Recall(num_classes=4, task="multiclass", average="macro").to(device)

        loop = tqdm(train_loader, desc="Training", leave=False)
        for imgs, labels in loop:
            imgs = imgs.to(device)
            labels = labels.to(device).long()
            b = imgs.size(0)
            steps += 1

            # ----------------------
            # Discriminator step
            # ----------------------
            optimizer_D.zero_grad()
            imgs.requires_grad_(True)  # for R1

            real_out, class_out_real = discriminator(imgs, labels)
            real_targets = smooth_labels(torch.ones(b, 1, device=device), label_smooth)
            fake_targets = smooth_labels(torch.zeros(b, 1, device=device), label_smooth)

            d_loss_real = adversarial_loss(real_out, real_targets)
            c_loss_real = classification_loss(class_out_real, labels)

            # Sample noise and generate fake images
            z = torch.randn(b, generator.z_dim, device=device)
            fake_imgs = generator(z, labels)  # [-1,1]
            fake_for_disc = to_imagenet_norm_from_tanh(fake_imgs)

            fake_out, class_out_fake = discriminator(fake_for_disc.detach(), labels)
            d_loss_fake = adversarial_loss(fake_out, fake_targets)
            c_loss_fake = classification_loss(class_out_fake, labels)

            r1 = r1_regularization(real_out, imgs)
            d_loss = d_loss_real + d_loss_fake + c_loss_real + c_loss_fake + 0.5 * gamma_r1 * r1
            d_loss.backward()
            optimizer_D.step()
            imgs.requires_grad_(False)

            # ----------------------
            # Generator step
            # ----------------------
            optimizer_G.zero_grad()
            g_out, g_class = discriminator(fake_for_disc, labels)
            g_loss_adv = adversarial_loss(g_out, real_targets)
            g_loss_cls = classification_loss(g_class, labels)
            g_loss = g_loss_adv + g_loss_cls
            g_loss.backward()
            optimizer_G.step()

            # Accumulate stats
            d_real_sum += d_loss_real.item()
            d_fake_sum += d_loss_fake.item()
            d_c_real_sum += c_loss_real.item()
            d_c_fake_sum += c_loss_fake.item()
            r1_sum += (0.5 * gamma_r1 * r1).item()
            g_adv_sum += g_loss_adv.item()
            g_cls_sum += g_loss_cls.item()

            train_losses.append(g_loss.item())  # legacy G-only
            train_total_losses.append(d_loss.item() + g_loss.item())

            preds = torch.argmax(class_out_real, dim=1)
            precision.update(preds, labels)
            recall.update(preds, labels)

            loop.set_postfix({"G_loss": g_loss.item(), "D_loss": d_loss.item()})

        train_loss = sum(train_losses) / max(1, len(train_losses))
        train_total = sum(train_total_losses) / max(1, len(train_total_losses))
        train_precision = precision.compute().item()
        train_recall = recall.compute().item()

        history["train_d_real"].append(d_real_sum / steps)
        history["train_d_fake"].append(d_fake_sum / steps)
        history["train_d_c_real"].append(d_c_real_sum / steps)
        history["train_d_c_fake"].append(d_c_fake_sum / steps)
        history["train_r1"].append(r1_sum / steps)
        history["train_g_adv"].append(g_adv_sum / steps)
        history["train_g_cls"].append(g_cls_sum / steps)
        history["train_total"].append(train_total)

        # ----------------------
        # Validation loop
        # ----------------------
        generator.eval()
        discriminator.eval()

        val_losses = []
        val_total_losses = []
        v_d_real_sum = v_d_fake_sum = 0.0
        v_d_c_real_sum = v_d_c_fake_sum = 0.0
        v_g_adv_sum = v_g_cls_sum = 0.0
        v_steps = 0

        val_precision = Precision(num_classes=4, task="multiclass", average="macro").to(device)
        val_recall = Recall(num_classes=4, task="multiclass", average="macro").to(device)

        with torch.no_grad():
            loop_val = tqdm(val_loader, desc="Validation", leave=False)
            for bi, (imgs, labels) in enumerate(loop_val):
                if (max_val_batches is not None) and (bi >= max_val_batches):
                    break

                imgs = imgs.to(device)
                labels = labels.to(device).long()
                b = imgs.size(0)
                v_steps += 1

                real_targets_val = smooth_labels(torch.ones(b, 1, device=device), label_smooth)
                fake_targets_val = smooth_labels(torch.zeros(b, 1, device=device), label_smooth)

                real_out, class_out_real = discriminator(imgs, labels)
                d_loss_real = adversarial_loss(real_out, real_targets_val)
                c_loss_real = classification_loss(class_out_real, labels)

                z = torch.randn(b, generator.z_dim, device=device)
                fake_imgs = generator(z, labels)
                fake_for_disc = to_imagenet_norm_from_tanh(fake_imgs)

                fake_out, class_out_fake = discriminator(fake_for_disc, labels)
                d_loss_fake = adversarial_loss(fake_out, fake_targets_val)
                c_loss_fake = classification_loss(class_out_fake, labels)

                d_loss_val = d_loss_real + d_loss_fake + c_loss_real + c_loss_fake
                g_loss_val = adversarial_loss(fake_out, real_targets_val)
                c_loss_gen_val = classification_loss(class_out_fake, labels)
                total_g_loss_val = g_loss_val + c_loss_gen_val

                batch_val_total = d_loss_val.item() + total_g_loss_val.item()
                val_total_losses.append(batch_val_total)

                batch_val_loss = (d_loss_val.item() + total_g_loss_val.item()) / 2.0
                val_losses.append(batch_val_loss)

                preds = torch.argmax(class_out_real, dim=1)
                val_precision.update(preds, labels)
                val_recall.update(preds, labels)

                loop_val.set_postfix({"Val_loss": batch_val_loss})

                v_d_real_sum += d_loss_real.item()
                v_d_fake_sum += d_loss_fake.item()
                v_d_c_real_sum += c_loss_real.item()
                v_d_c_fake_sum += c_loss_fake.item()
                v_g_adv_sum += g_loss_val.item()
                v_g_cls_sum += c_loss_gen_val.item()

        val_loss = sum(val_losses) / max(1, len(val_losses))
        val_total = sum(val_total_losses) / max(1, len(val_total_losses))
        val_precision_score = val_precision.compute().item()
        val_recall_score = val_recall.compute().item()

        history["val_d_real"].append(v_d_real_sum / max(1, v_steps))
        history["val_d_fake"].append(v_d_fake_sum / max(1, v_steps))
        history["val_d_c_real"].append(v_d_c_real_sum / max(1, v_steps))
        history["val_d_c_fake"].append(v_d_c_fake_sum / max(1, v_steps))
        history["val_g_adv"].append(v_g_adv_sum / max(1, v_steps))
        history["val_g_cls"].append(v_g_cls_sum / max(1, v_steps))
        history["val_total"].append(val_total)

        print(
            f"Epoch {epoch}: "
            f"Train Loss(G only)={train_loss:.4f}, Val Loss(avg)={val_loss:.4f}, "
            f"Train Total(D+G)={train_total:.4f}, Val Total(D+G)={val_total:.4f}, "
            f"Train Prec={train_precision:.4f}, Val Prec={val_precision_score:.4f}, "
            f"Train Rec={train_recall:.4f}, Val Rec={val_recall_score:.4f}\n"
            f"  Train D(real)={history['train_d_real'][-1]:.3f}, "
            f"D(fake)={history['train_d_fake'][-1]:.3f}, "
            f"Dc(real)={history['train_d_c_real'][-1]:.3f}, "
            f"Dc(fake)={history['train_d_c_fake'][-1]:.3f}, "
            f"R1={history['train_r1'][-1]:.3f} | "
            f"Gadv={history['train_g_adv'][-1]:.3f}, Gcls={history['train_g_cls'][-1]:.3f}\n"
            f"  Val   D(real)={history['val_d_real'][-1]:.3f}, "
            f"D(fake)={history['val_d_fake'][-1]:.3f}, "
            f"Dc(real)={history['val_d_c_real'][-1]:.3f}, "
            f"Dc(fake)={history['val_d_c_fake'][-1]:.3f} | "
            f"Gadv={history['val_g_adv'][-1]:.3f}, Gcls={history['val_g_cls'][-1]:.3f}"
        )

        if epoch % 5 == 0:
            show_generated_samples(generator, device, epoch=epoch)

        # ----------------------
        # FID / KID on validation
        # ----------------------
        fid_metric.reset()
        kid_metric.reset()
        with torch.no_grad():
            for bi, (imgs, labels) in enumerate(val_loader):
                if (max_val_batches is not None) and (bi >= max_val_batches):
                    break
                imgs = imgs.to(device)
                labels = labels.to(device).long()
                b = imgs.size(0)

                real01 = denormalize_imagenet(imgs)
                z = torch.randn(b, generator.z_dim, device=device)
                fake_imgs = generator(z, labels)
                fake01 = (fake_imgs + 1.0) / 2.0

                fid_metric.update((real01 * 255).to(torch.uint8), real=True)
                fid_metric.update((fake01 * 255).to(torch.uint8), real=False)
                kid_metric.update((real01 * 255).to(torch.uint8), real=True)
                kid_metric.update((fake01 * 255).to(torch.uint8), real=False)

        fid_val = fid_metric.compute().item()
        kid_mean, kid_std = kid_metric.compute()
        kid_val = kid_mean.item()

        ms_val = ms_ssim_per_class(
            generator,
            device=device,
            num_classes=4,
            samples_per_class=32
        )

        print(f"Epoch {epoch}: FID={fid_val:.2f} | KID={kid_val:.4f} | MS-SSIM(gen, per-class)={ms_val:.4f}")

        # Save best-by-FID checkpoint
        if save_path and fid_val < best_val_fid:
            best_val_fid = fid_val
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(
                {"G": generator.state_dict(),
                 "D": discriminator.state_dict(),
                 "epoch": epoch,
                 "val_fid": fid_val},
                save_path
            )
            print(f"Saved best checkpoint (G+D+epoch) by FID to {save_path}")

        # Rolling checkpoints (optional)
        if epoch > 65:
            os.makedirs("ckpts", exist_ok=True)
            torch.save(
                {
                    "G": generator.state_dict(),
                    "D": discriminator.state_dict(),
                    "optG": optimizer_G.state_dict(),
                    "optD": optimizer_D.state_dict(),
                    "epoch": epoch,
                    "val_fid": fid_val,
                },
                f"ckpts/ckpt_{epoch:03}.pt"
            )

        # Store scalar history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_prec"].append(train_precision)
        history["val_prec"].append(val_precision_score)
        history["train_rec"].append(train_recall)
        history["val_rec"].append(val_recall_score)
        history["fid"].append(fid_val)
        history["kid"].append(kid_val)
        history["ms_ssim_mean"].append(ms_val)

        # ----------------------
        # Staged ViT unfreezing
        # ----------------------
        if use_metric_trigger:
            if vit_stage == "frozen" and epoch >= start_partial_epoch and metrics_plateaued():
                print("Metrics plateaued; unfreezing last ViT block(s).")
                unfreeze_vit_last_blocks(discriminator, num_blocks=partial_blocks)
                optimizer_D = build_discriminator_optimizer(
                    discriminator, base_lr=base_dlr, vit_lr=vit_lr, betas=betas
                )
                vit_stage = "partial"
            elif vit_stage == "partial" and epoch >= start_full_epoch and metrics_plateaued():
                print("Further plateau; unfreezing more ViT block(s).")
                unfreeze_vit_last_blocks(discriminator, num_blocks=full_blocks)
                optimizer_D = build_discriminator_optimizer(
                    discriminator, base_lr=base_dlr, vit_lr=vit_lr, betas=betas
                )
                vit_stage = "full"

    return history


# ---------------------------------------------------------------------
# One-shot test evaluation
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate_on_test(
    generator,
    test_loader,
    num_classes=4,
    samples_per_class_for_ms=32,
    max_test_batches=None
):
    """
    Computes final FID, KID on held-out test set, and MS-SSIM (generated-only) per class.
    Call this once using the best validation-FID checkpoint.
    """
    generator.to(device).eval()

    fid = FrechetInceptionDistance(feature=2048).to(device)
    kid = KernelInceptionDistance(subset_size=500).to(device)
    ms_metric = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    for bi, (imgs, labels) in enumerate(test_loader):
        if (max_test_batches is not None) and (bi >= max_test_batches):
            break
        imgs = imgs.to(device)
        labels = labels.to(device).long()
        b = imgs.size(0)

        real01 = denormalize_imagenet(imgs)

        z = torch.randn(b, generator.z_dim, device=device)
        fake_imgs = generator(z, labels)
        fake01 = (fake_imgs + 1.0) / 2.0

        fid.update((real01 * 255).to(torch.uint8), real=True)
        fid.update((fake01 * 255).to(torch.uint8), real=False)
        kid.update((real01 * 255).to(torch.uint8), real=True)
        kid.update((fake01 * 255).to(torch.uint8), real=False)

    fid_test = fid.compute().item()
    kid_mean, kid_std = kid.compute()
    kid_test = kid_mean.item()

    ms_test = ms_ssim_per_class(
        generator,
        device=device,
        num_classes=num_classes,
        samples_per_class=samples_per_class_for_ms
    )

    print(f"[TEST] FID={fid_test:.2f} | KID={kid_test:.4f} | MS-SSIM(gen, per-class)={ms_test:.4f}")
    return {"fid": fid_test, "kid": kid_test, "ms_ssim_mean": ms_test}
