import cv2
import torch
import numpy as np


class Extraction:
    def __init__(self, target_hw=(256, 256), radius_ratio=0.25, device='cpu'):
        self.target_hw = target_hw
        self.radius_ratio = radius_ratio
        self.device = device

    def _create_circular_mask(self, H, W):
        max_radius = min(H, W) / 2.0
        radius = max_radius * self.radius_ratio

        Y, X = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing='ij'
        )

        center_Y, center_X = H // 2, W // 2
        dist_from_center = torch.sqrt((Y - center_Y) ** 2 + (X - center_X) ** 2)
        mask = (dist_from_center <= radius).float()
        return mask.view(1, 1, H, W)

    def stream_video_batches(self, video_path, batch_size=32):
        cap = cv2.VideoCapture(video_path)

        while True:
            frames = []
            while len(frames) < batch_size:
                ret, frame = cap.read()
                if not ret: break

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                H, W = self.target_hw
                resized_frame = cv2.resize(gray_frame, (W, H))
                frames.append(resized_frame)

            if len(frames) == 0:
                break

            spatial_tensor = torch.tensor(
                np.array(frames), dtype=torch.float32, device=self.device
            ).unsqueeze(1) / 255.0

            B, C, H, W = spatial_tensor.shape

            fft_complex = torch.fft.fftshift(torch.fft.fft2(spatial_tensor))
            low_pass_mask = self._create_circular_mask(H, W)
            high_pass_mask = 1.0 - low_pass_mask

            low_freq_complex = fft_complex * low_pass_mask
            high_freq_complex = fft_complex * high_pass_mask

            low_frequency = torch.cat([low_freq_complex.real, low_freq_complex.imag], dim=1)
            high_frequency = torch.cat([high_freq_complex.real, high_freq_complex.imag], dim=1)

            yield low_frequency, high_frequency

        cap.release()