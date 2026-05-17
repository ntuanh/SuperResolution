import torch
import torch.nn as nn
import torchvision.models as models


class JEPAEncoder(nn.Module):
    def __init__(self, latent_dim=256):
        super().__init__()
        self.backbone = models.mobilenet_v3_small(weights=None)

        orig_conv = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels=2, out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size, stride=orig_conv.stride,
            padding=orig_conv.padding, bias=False
        )

        in_feat = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_feat, latent_dim)

    def forward(self, x):
        return self.backbone(x)


class JEPAPredictor(nn.Module):
    def __init__(self, latent_dim=256, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, s_x):
        return self.net(s_x)


class JEPADecoder(nn.Module):
    def __init__(self, latent_dim=256, target_hw=(256, 256)):
        super().__init__()

        # Combine Context and Predictor vectors (256 + 256 = 512)
        combined_dim = latent_dim * 2

        # Base spatial grid (Shrink image by 16x)
        self.init_h = target_hw[0] // 16
        self.init_w = target_hw[1] // 16

        # UPGRADE: Project into 256 channels instead of 128
        self.fc = nn.Linear(combined_dim, 256 * self.init_h * self.init_w)

        # UPGRADE: Thicker layers for better image generation
        self.upsample = nn.Sequential(
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            # 128x128 -> 256x256 (Final 2 Channels: Real/Imaginary)
            nn.ConvTranspose2d(32, 2, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, s_x, s_y_pred):
        z = torch.cat([s_x, s_y_pred], dim=1)
        z = self.fc(z)
        z = z.view(-1, 256, self.init_h, self.init_w)  # Match the 256 channels here!
        return self.upsample(z)