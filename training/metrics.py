import torch
from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

def ms_ssim_per_class(generator, device, num_classes=4, samples_per_class=32):
    ms = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    generator.eval()
    values = []

    for c in range(num_classes):
        labels = torch.full((samples_per_class,), c, device=device)
        z1 = torch.randn(samples_per_class, generator.z_dim, device=device)
        z2 = torch.randn(samples_per_class, generator.z_dim, device=device)

        f1 = (generator(z1, labels) + 1) / 2
        f2 = (generator(z2, labels) + 1) / 2
        values.append(ms(f1, f2).item())

    generator.train()
    return sum(values) / len(values)
