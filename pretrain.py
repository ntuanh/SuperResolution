import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from models import JEPAEncoder, JEPAPredictor
from extract.extraction import Extraction

data_path = "data/QAvideo2.mp4"

class JEPATrainer:
    def __init__(self, latent_dim=256, lr=1e-4, tau=0.996, device='cpu'):
        self.device = device
        self.tau = tau

        self.context_encoder = JEPAEncoder(latent_dim=latent_dim).to(self.device)
        self.target_encoder = JEPAEncoder(latent_dim=latent_dim).to(self.device)
        self.predictor = JEPAPredictor(latent_dim=latent_dim).to(self.device)

        self.target_encoder.load_state_dict(self.context_encoder.state_dict())
        for param in self.target_encoder.parameters():
            param.requires_grad = False

        self.optimizer = optim.Adam(
            list(self.context_encoder.parameters()) + list(self.predictor.parameters()), lr=lr
        )
        self.criterion = nn.MSELoss()

    def train_step(self, low_freq, high_freq):
        self.context_encoder.train()
        self.predictor.train()
        self.target_encoder.eval()

        self.optimizer.zero_grad()

        s_x = self.context_encoder(low_freq)
        s_y_pred = self.predictor(s_x)

        with torch.no_grad():
            s_y_true = self.target_encoder(high_freq)

        # L2 Normalization to prevent Representation Collapse
        s_y_pred = F.normalize(s_y_pred, dim=-1)
        s_y_true = F.normalize(s_y_true, dim=-1)

        loss = self.criterion(s_y_pred, s_y_true)
        loss.backward()
        self.optimizer.step()

        # EMA Update
        with torch.no_grad():
            for p_target, p_context in zip(self.target_encoder.parameters(), self.context_encoder.parameters()):
                p_target.data.mul_(self.tau).add_(p_context.data, alpha=1.0 - self.tau)

        return loss.item()


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Starting Phase 1 on {device}...")

    extractor = Extraction(target_hw=(256, 256), radius_ratio=0.25, device=device)
    trainer = JEPATrainer(device=device)

    for epoch in range(5):
        print(f"\n--- Epoch {epoch + 1}/10 ---")
        step = 0
        for low_freq, high_freq in extractor.stream_video_batches(data_path, batch_size=16):
            loss = trainer.train_step(low_freq, high_freq)
            step += 1
            if step % 10 == 0:
                print(f"Step {step} | Latent Loss: {loss:.4f}")

    torch.save({
        'context_encoder': trainer.context_encoder.state_dict(),
        'predictor': trainer.predictor.state_dict(),
    }, "jepa_ready_for_phase2.pth")
    print("Phase 1 Complete! Saved jepa_ready_for_phase2.pth")