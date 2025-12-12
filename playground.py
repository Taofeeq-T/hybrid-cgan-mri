"""
playground.py
-------------
Main script to:
- Load dataset
- Initialize Generator + Hybrid Discriminator
- Train (with progressive ViT unfreezing)
- Save best checkpoint
- Evaluate test set
- Generate sample images
"""

import torch
from models.generator import ConvGenerator
from models.discriminator import HybridDiscriminator
from utils.dataset import get_dataloaders
from training.trainer import full_train_val, evaluate_on_test
from torchvision.utils import save_image


# -----------------------------------------
# Hyperparameters
# -----------------------------------------
z_dim = 100
num_classes = 4
epochs = 150
batch_size = 32
save_path = "ckpts/best_model.pt"

# -----------------------------------------
# Load dataset
# -----------------------------------------
train_loader, val_loader, test_loader, class_names = get_dataloaders(
    data_dir="data/brain_mri",
    batch_size=batch_size,
)

print("\nClasses discovered:", class_names)

# -----------------------------------------
# Initialize models
# -----------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

G = ConvGenerator(z_dim=z_dim, num_classes=num_classes).to(device)
D = HybridDiscriminator(num_classes=num_classes, freeze_vit=True).to(device)

print("Using device:", device)

# -----------------------------------------
# Train
# -----------------------------------------
print("\nStarting GAN training...\n")

history = full_train_val(
    generator=G,
    discriminator=D,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=epochs,
    save_path=save_path,

    base_glr=1e-4,
    base_dlr=4e-4,
    vit_lr=2e-5,
    gamma_r1=5.0,
    label_smooth=0.05,

    start_partial_epoch=4,
    partial_blocks=2,
    start_full_epoch=10,
    full_blocks=4,

    use_metric_trigger=True,
)

print("\nTraining completed.\n")

# -----------------------------------------
# Evaluate best model
# -----------------------------------------
best_ckpt = torch.load(save_path, map_location=device)
G.load_state_dict(best_ckpt["G"])

print("Evaluating best checkpoint...\n")
results = evaluate_on_test(
    generator=G,
    test_loader=test_loader,
    num_classes=num_classes,
)

print("\nFinal Test Results:", results)

# -----------------------------------------
# Generate sample images
# -----------------------------------------
print("\nGenerating sample images...")

labels = torch.arange(num_classes).to(device)
z = torch.randn(num_classes, z_dim).to(device)
samples = (G(z, labels) + 1) / 2.0  # [-1,1] → [0,1]

save_image(samples, "results/sample_grid.png", nrow=2)
print("Saved sample grid at results/sample_grid.png\n")
