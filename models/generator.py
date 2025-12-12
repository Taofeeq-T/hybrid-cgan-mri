import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvGenerator(nn.Module):
    def __init__(self, z_dim=128, num_classes=4, img_channels=3):
        super().__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes

        self.fc = nn.Linear(z_dim + num_classes, 512 * 4 * 4)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.BatchNorm2d(16), nn.ReLU(True),
            nn.Conv2d(16, img_channels, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, z, labels):
        labels_1h = torch.nn.functional.one_hot(labels, num_classes=self.num_classes).float()
        x = torch.cat((z, labels_1h), dim=1)
        x = self.fc(x).view(-1, 512, 4, 4)
        img = self.net(x)
        return torch.nn.functional.interpolate(img, (224, 224),
                                               mode="bilinear",
                                               align_corners=False)
