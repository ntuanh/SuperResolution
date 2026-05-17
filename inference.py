import torch
import cv2
import numpy as np
from models import JEPAEncoder, JEPAPredictor, JEPADecoder
from extract.extraction import Extraction

data_path = "data/QAvideo2.mp4"


def simulate_real_world_deployment(image_path="test_frame.jpg"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---------------------------------------------------------
    # SENDER SIDE (e.g., Jetson Orin Nano)
    # ---------------------------------------------------------
    print("--- SENDER DEVICE ---")
    sender_encoder = JEPAEncoder(latent_dim=256).to(device)
    sender_encoder.load_state_dict(torch.load("sender_edge.pth", map_location=device, weights_only=True))
    sender_encoder.eval()

    # Extract Low Frequency from raw image (Simulating your camera)
    extractor = Extraction(target_hw=(256, 256), radius_ratio=0.25, device=device)

    # Just grabbing one frame for the test
    low_freq_2d, _ = next(extractor.stream_video_batches(data_path, batch_size=1))

    # Compress 2D spectrum into 1D Vector
    with torch.no_grad():
        transmitted_vector = sender_encoder(low_freq_2d)

    print(f"Vector ready for transmission. Shape: {transmitted_vector.shape}")

    # --- NETWORK TRANSMISSION HAPPENS HERE ---

    # ---------------------------------------------------------
    # RECEIVER SIDE (e.g., Cloud Server)
    # ---------------------------------------------------------
    print("\n--- RECEIVER SERVER ---")
    cloud_weights = torch.load("receiver_cloud.pth", map_location=device, weights_only=True)

    predictor = JEPAPredictor(latent_dim=256).to(device)
    decoder = JEPADecoder(latent_dim=256).to(device)

    predictor.load_state_dict(cloud_weights['predictor'])
    decoder.load_state_dict(cloud_weights['decoder'])
    predictor.eval()
    decoder.eval()

    with torch.no_grad():
        # 1. Predict High Frequencies
        predicted_high_vector = predictor(transmitted_vector)

        # 2. Decode into 2D High Frequency Spectrum
        reconstructed_high_freq = decoder(transmitted_vector, predicted_high_vector)

        # 3. Combine Low and High frequencies
        # low_freq_2d was theoretically kept on the server (if transmitting residual)
        # or we just reconstruct the high-detail parts. Let's combine them mathematically:
        final_spectrum = low_freq_2d + reconstructed_high_freq

        # 4. Turn Real/Imaginary channels back into PyTorch Complex numbers
        complex_spectrum = torch.complex(final_spectrum[:, 0, :, :], final_spectrum[:, 1, :, :])

        # 5. The Magic Math: 2D-IFFT (Inverse FFT)
        # We unshift the center and apply the inverse transform to get spatial pixels
        spatial_image = torch.fft.ifft2(torch.fft.ifftshift(complex_spectrum)).real

        # 6. Format for OpenCV to save/show
        output_image = spatial_image[0].cpu().numpy()
        output_image = np.clip(output_image * 255.0, 0, 255).astype(np.uint8)

        cv2.imwrite("final_reconstruction.jpg", output_image)
        print("Success! Final image saved as 'final_reconstruction.jpg'")


if __name__ == "__main__":
    simulate_real_world_deployment()