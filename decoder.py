import torch
import torch.nn as nn
import torch.optim as optim
from models import JEPAEncoder, JEPAPredictor, JEPADecoder
from extract.extraction import Extraction

video_path = "data/QAvideo2.mp4"

def run_phase2_training():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting Upgraded Phase 2 on {device}...")

    # 1. Initialize models
    context_encoder = JEPAEncoder(latent_dim=256).to(device)
    predictor = JEPAPredictor(latent_dim=256).to(device)
    decoder = JEPADecoder(latent_dim=256, target_hw=(256, 256)).to(device)

    # 2. Load Phase 1 Weights
    phase1_weights = torch.load("jepa_ready_for_phase2.pth", map_location=device, weights_only=True)
    context_encoder.load_state_dict(phase1_weights['context_encoder'])
    predictor.load_state_dict(phase1_weights['predictor'])

    # 3. Freeze Phase 1 models (Crucial!)
    for param in context_encoder.parameters(): param.requires_grad = False
    for param in predictor.parameters(): param.requires_grad = False

    context_encoder.eval()
    predictor.eval()
    decoder.train()

    # 4. UPGRADE: Faster Learning Rate + Cosine Scheduler
    epochs = 50  # You might need a few more epochs now
    optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.L1Loss()

    extractor = Extraction(target_hw=(256, 256), radius_ratio=0.25, device=device)

    # 5. Training Loop
    for epoch in range(epochs):
        step = 0
        epoch_loss = 0.0

        for low_freq, high_freq_target in extractor.stream_video_batches(video_path,  batch_size=16):
            optimizer.zero_grad()

            # Extract Vectors
            with torch.no_grad():
                s_x = context_encoder(low_freq)
                s_y_pred = predictor(s_x)

            # Generate Image Frequencies
            reconstructed_high_freq = decoder(s_x, s_y_pred)

            # --- UPGRADE: DATA STANDARDIZATION ---
            # Lock the wild FFT numbers into a safe [-1, 1] range for the neural network
            target_mean = high_freq_target.mean()
            target_std = high_freq_target.std() + 1e-8

            normalized_target = (high_freq_target - target_mean) / target_std
            normalized_prediction = (reconstructed_high_freq - target_mean) / target_std

            # Calculate stable L1 Loss
            loss = criterion(normalized_prediction, normalized_target)

            # Backpropagate
            loss.backward()
            optimizer.step()

            step += 1
            epoch_loss += loss.item()

            # Print every 5 steps to keep the terminal clean
            # if step % 5 == 0:
            #     print(f"Epoch {epoch + 1} | Step {step} | Decoder L1 Loss: {loss.item():.4f}")

        # UPGRADE: Step the learning rate scheduler at the end of the epoch
        scheduler.step()

        # Print average epoch loss
        avg_loss = epoch_loss / max(1, step)
        print(
            f"--> End of Epoch {epoch + 1} | Average Loss: {avg_loss:.4f} | Current LR: {scheduler.get_last_lr()[0]:.6f}\n")

    # 6. Export Deployment Files
    print("Training Complete! Exporting Split-Inference Files...")
    torch.save(context_encoder.state_dict(), "sender_edge.pth")
    torch.save({'predictor': predictor.state_dict(), 'decoder': decoder.state_dict()}, "receiver_cloud.pth")
    print("Success: 'sender_edge.pth' and 'receiver_cloud.pth' are ready for deployment!")


if __name__ == "__main__":
    run_phase2_training()