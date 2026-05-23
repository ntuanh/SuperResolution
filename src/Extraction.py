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

class Extraction:

    def __init__(
        self,
        target_hw=(256, 256),
        radius_ratio=0.25,
        cutoff_frequency=25,
        device="cpu"
    ):

        self.target_hw = target_hw

        self.radius_ratio = radius_ratio

        self.cutoff_frequency = cutoff_frequency

        self.device = torch.device(device)

        self._mask_cache = {}

    # =====================================================
    # GAUSSIAN LOW PASS MASK
    # =====================================================

    def _create_circular_mask(self, height, width):

        cache_key = (
            height,
            width,
            self.cutoff_frequency
        )

        if cache_key in self._mask_cache:
            return self._mask_cache[cache_key]

        # ==============================================
        # CREATE GRID
        # ==============================================

        y = torch.arange(
            height,
            device=self.device
        ).float()

        x = torch.arange(
            width,
            device=self.device
        ).float()

        yy, xx = torch.meshgrid(
            y,
            x,
            indexing='ij'
        )

        # ==============================================
        # FFT CENTER
        # ==============================================

        center_y = height / 2
        center_x = width / 2

        # ==============================================
        # DISTANCE FROM CENTER
        # ==============================================

        dist = torch.sqrt(
            (yy - center_y) ** 2 +
            (xx - center_x) ** 2
        )

        # ==============================================
        # GAUSSIAN LOW PASS
        # ==============================================

        sigma = self.cutoff_frequency

        mask = torch.exp(
            -(dist ** 2) / (2 * sigma ** 2)
        )

        # ==============================================
        # SHAPE:
        # [1, 1, H, W]
        # ==============================================

        mask = mask.unsqueeze(0).unsqueeze(0)

        # ==============================================
        # CACHE
        # ==============================================

        self._mask_cache[cache_key] = mask

        return mask

    # =====================================================
    # VIDEO STREAM
    # =====================================================

    def stream_video_batches(
        self,
        video_path,
        batch_size=32
    ):

        cap = cv2.VideoCapture(str(video_path))

        height, width = self.target_hw

        frame_buffer = np.empty(
            (batch_size, height, width),
            dtype=np.uint8
        )

        try:

            while True:

                filled = 0

                # ======================================
                # LOAD BATCH
                # ======================================

                while filled < batch_size:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    gray_frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2GRAY
                    )

                    resized = cv2.resize(
                        gray_frame,
                        (width, height),
                        interpolation=cv2.INTER_AREA
                    )

                    frame_buffer[filled] = resized

                    filled += 1

                if filled == 0:
                    break

                # ======================================
                # CPU -> TORCH
                # ======================================

                cpu_tensor = torch.from_numpy(
                    frame_buffer[:filled]
                )

                if self.device.type == "cuda":
                    cpu_tensor = cpu_tensor.pin_memory()

                with torch.no_grad():

                    # ==================================
                    # NORMALIZE
                    # ==================================

                    spatial_tensor = cpu_tensor.to(
                        device=self.device,
                        dtype=torch.float32,
                        non_blocking=self.device.type == "cuda",
                    )

                    spatial_tensor = spatial_tensor.unsqueeze(1)

                    spatial_tensor = spatial_tensor / 255.0

                    # ==================================
                    # FFT
                    # ==================================

                    fft_complex = torch.fft.fft2(
                        spatial_tensor
                    )

                    fft_complex = torch.fft.fftshift(
                        fft_complex
                    )

                    # ==================================
                    # LOW PASS MASK
                    # ==================================

                    low_pass_mask = self._create_circular_mask(
                        height,
                        width
                    )

                    # ==================================
                    # LOW / HIGH SPLIT
                    # ==================================

                    low_freq_complex = (
                        fft_complex * low_pass_mask
                    )

                    high_freq_complex = (
                        fft_complex - low_freq_complex
                    )

                    # ==================================
                    # REAL + IMAG
                    # ==================================

                    low_frequency = torch.cat([
                        low_freq_complex.real,
                        low_freq_complex.imag
                    ], dim=1)

                    high_frequency = torch.cat([
                        high_freq_complex.real,
                        high_freq_complex.imag
                    ], dim=1)

                    # ==================================
                    # SAFE NORMALIZATION
                    # ==================================

                    low_frequency = (
                        low_frequency /
                        (
                            low_frequency.abs().mean(
                                dim=(1,2,3),
                                keepdim=True
                            ) + 1e-6
                        )
                    )

                    high_frequency = (
                        high_frequency /
                        (
                            high_frequency.abs().mean(
                                dim=(1,2,3),
                                keepdim=True
                            ) + 1e-6
                        )
                    )

                yield low_frequency, high_frequency

        finally:

            cap.release()