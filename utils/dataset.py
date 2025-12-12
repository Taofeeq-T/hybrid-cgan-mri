import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

def get_dataloaders(data_dir="data/brain_mri", batch_size=32):
    """
    Loads brain MRI dataset using ImageFolder.
    Applies ImageNet normalization for ViT compatibility.
    """

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size]
    )

    num_workers = min(4, os.cpu_count() or 0)

    train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False,
                             num_workers=num_workers)

    return train_loader, val_loader, test_loader, dataset.classes
