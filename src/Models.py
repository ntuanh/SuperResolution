from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import models as tv_models

import config

# Configuration mapping
VIDEO_PATH = config.VIDEO_PATH
IMAGE_PATH = config.IMAGE_PATH
CHECKPOINT_PATH = config.CHECKPOINT_PATH
PHASE1_CHECKPOINT_PATH = config.PHASE1_CHECKPOINT_PATH
PHASE1_READY_FOR_PHASE2_PATH = config.PHASE1_READY_FOR_PHASE2_PATH
INFERENCE_OUTPUT_PATH = config.INFERENCE_OUTPUT_PATH

PHASE1_CHECKPOINT_CANDIDATES = config.PHASE1_CHECKPOINT_CANDIDATES
PHASE1_ADDITIONAL_EPOCHS = config.PHASE1_ADDITIONAL_EPOCHS
PHASE1_LEARNING_RATE = config.PHASE1_LEARNING_RATE
PHASE1_WEIGHT_DECAY = config.PHASE1_WEIGHT_DECAY
PHASE1_EMA_DECAY = config.PHASE1_EMA_DECAY
PHASE1_MAX_STEPS_PER_EPOCH = config.PHASE1_MAX_STEPS_PER_EPOCH

ADDITIONAL_EPOCHS = config.ADDITIONAL_EPOCHS
BATCH_SIZE = config.BATCH_SIZE
MAX_STEPS_PER_EPOCH = config.MAX_STEPS_PER_EPOCH
SAVE_EVERY_STEPS = config.SAVE_EVERY_STEPS
LEARNING_RATE = config.LEARNING_RATE
TARGET_HW = config.TARGET_HW
RADIUS_RATIO = config.RADIUS_RATIO
LATENT_DIM = config.LATENT_DIM
PREDICTOR_HIDDEN_DIM = config.PREDICTOR_HIDDEN_DIM
USE_PRETRAINED_BACKBONE = config.USE_PRETRAINED_BACKBONE
ARCHITECTURE_VERSION = config.ARCHITECTURE_VERSION





class JEPAEncoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, pretrained=False):
        super().__init__()

        weights = None

        if pretrained:
            try:
                weights = tv_models.MobileNet_V3_Small_Weights.DEFAULT
            except:
                weights = None

        backbone = tv_models.mobilenet_v3_small(weights=weights)

        # ==========================================
        # CHANGE INPUT CHANNELS: 3 -> 2
        # ==========================================
        orig_conv = backbone.features[0][0]

        backbone.features[0][0] = nn.Conv2d(
            in_channels=2,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )

        # ==========================================
        # REMOVE CLASSIFIER HEAD
        # ==========================================
        self.features = backbone.features

        # ==========================================
        # PROJECT FEATURES
        # ==========================================
        self.proj = nn.Sequential(
            nn.Conv2d(576, latent_dim, kernel_size=1),
            nn.BatchNorm2d(latent_dim),
            nn.GELU(),
        )

    def forward(self, x):

        # MobileNet feature map
        x = self.features(x)

        # Shape:
        # [B, 576, 8, 8] تقريباً

        x = self.proj(x)

        # Shape:
        # [B, latent_dim, 8, 8]

        return x


# =========================================================
# JEPA PREDICTOR
# =========================================================

class JEPAPredictor(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, hidden_dim=PREDICTOR_HIDDEN_DIM):
        super().__init__()

        self.net = nn.Sequential(

            nn.Conv2d(latent_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),

            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.GELU(),

            nn.Conv2d(hidden_dim, latent_dim, 1),
        )

    def forward(self, s_x):

        # Input:
        # [B, latent_dim, H, W]

        return self.net(s_x)


# =========================================================
# JEPA DECODER
# =========================================================

class JEPADecoder(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM, target_hw=(256, 256)):
        super().__init__()

        self.decoder = nn.Sequential(

            # ======================================
            # INPUT:
            # [B, latent_dim*2, 8, 8]
            # ======================================

            nn.Conv2d(latent_dim * 2, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),

            # 8 -> 16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # 16 -> 32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # 32 -> 64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            # 64 -> 128
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),

            # 128 -> 256
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            # OUTPUT FFT:
            # 2 channels = real + imag
            nn.Conv2d(16, 2, 3, padding=1),
        )

    def forward(self, s_x, s_y_pred):

        # ======================================
        # CONCAT FEATURE MAPS
        # ======================================

        z = torch.cat([s_x, s_y_pred], dim=1)

        # Shape:
        # [B, latent_dim*2, H, W]

        out = self.decoder(z)

        # Output:
        # [B, 2, 256, 256]

        return out