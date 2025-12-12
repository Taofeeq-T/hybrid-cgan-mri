import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm import create_model
import torch.nn.utils as utils

class HybridDiscriminator(nn.Module):
    def __init__(self, num_classes=4, freeze_vit=True):
        super().__init__()

        self.cnn = nn.Sequential(
            utils.spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)), nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)), nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)), nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Conv2d(256, 512, 4, 2, 1)), nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Conv2d(512, 512, 3, 1, 1)), nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Conv2d(512, 512, 3, 1, 1)), nn.LeakyReLU(0.2),
        )

        self.vit = create_model("vit_base_patch16_224", pretrained=True)
        self.vit.head = nn.Identity()

        if freeze_vit:
            for p in self.vit.parameters():
                p.requires_grad = False

        self.fdim = 512 + 768

        self.fusion_mlp = nn.Sequential(
            utils.spectral_norm(nn.Linear(self.fdim, 1024)),
            nn.LeakyReLU(0.2),
            utils.spectral_norm(nn.Linear(1024, self.fdim)),
            nn.LeakyReLU(0.2),
        )

        self.rf_linear = utils.spectral_norm(nn.Linear(self.fdim, 1))
        self.embed = nn.Embedding(num_classes, self.fdim)
        nn.init.normal_(self.embed.weight, 0, 0.1)
        self.class_head = utils.spectral_norm(nn.Linear(self.fdim, num_classes))

    def forward(self, x, y):
        cnn_feats = torch.mean(self.cnn(x), dim=[2, 3])
        vit_feats = self.vit(F.interpolate(x, (224,224)))

        fused = self.fusion_mlp(torch.cat([cnn_feats, vit_feats], dim=1))

        rf = self.rf_linear(fused)
        proj = (fused * self.embed(y)).sum(dim=1, keepdim=True) / math.sqrt(self.fdim)

        class_out = self.class_head(fused)
        return rf + proj, class_out
