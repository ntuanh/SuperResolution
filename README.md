
---

# Super-Resolution with JEPA 🔍

A self-supervised learning framework for Image Super-Resolution leveraging the **Joint-Embedding Predictive Architecture (JEPA)**. This project separates feature representation learning from the decoding process to reconstruct high-resolution images effectively from low-frequency domains.

## 🧠 Architecture Overview

The pipeline operates in the frequency domain using Fast Fourier Transform (FFT) to separate low and high-frequency image components. The training is divided into two distinct sequential phases:

* **Phase 1 (Representation Learning):** Uses a `JEPAEncoder` and `JEPAPredictor` to learn meaningful mappings from low-frequency contexts to high-frequency targets without decoding back to pixel space.
* **Phase 2 (Super-Resolution Decoding):** A `JEPADecoder` takes the frozen latent representations learned in Phase 1 and reconstructs the final high-resolution image.

## 📊 Hardware & Performance Metrics

This model is optimized for efficient training and real-time inference, utilizing real-world urban and traffic data.

**Training Setup:**

* **Hardware:** NVIDIA GeForce RTX 3050 (8GB VRAM)
* **Training Time:** ~1 hour per 5 epochs
* **Dataset Size:** ~61,000 training frames
* **Data Sources:** * Traffic camera footage provided by the Split Inference Team (SI Team).
* Urban Tracker dataset.
* Supplementary high-resolution images sourced from the internet.



**⚡ Inference Speeds (RTX 3050):**

* **Model Load Time:** ~781.8 ms (*781,820,800 ns*)
* **Inference Time (per image):** ~15.8 ms (*15,879,400 ns*)
* **Effective Framerate:** ~63 FPS

## 📈 Model Evaluation & Metrics

The training pipeline evaluates performance across both phases using distinct metric strategies:

* **Phase 1 Loss (Representation):** The encoder and predictor are optimized using **Mean Squared Error (MSE Loss / L2 Loss)**. This measures the distance between the predicted latent representation and the actual high-frequency target representation.
* **Phase 2 Loss (Reconstruction):** The decoder is trained using **Mean Absolute Error (L1 Loss)**. L1 Loss is utilized here to encourage sharper high-resolution image reconstructions and penalize pixel-level deviations.
* **Accuracy & Inference:** Accuracy is visually quantified during inference by calculating the **Absolute Error** map between the original image and the reconstructed image (`np.abs(original_image - reconstructed_image)`), highlighting areas where the model struggles with high-frequency detail.

## ⚖️ Strengths & Weaknesses

### Strong Points

* **Real-Time Capability:** With an inference time of ~15.8 ms per image (63 FPS), the model is highly suitable for real-time edge deployment on traffic cameras.
* **Representation Efficiency:** By utilizing the JEPA framework, the model avoids pixel-level collapse during Phase 1, focusing entirely on learning underlying spatial structures and semantics.
* **Frequency Domain Isolation:** Using FFT to isolate frequencies allows the network to specifically target the difficult task of predicting missing high-frequency details rather than reconstructing the entire image from scratch.
* **Resource Optimization:** The architecture is designed to train efficiently on mid-range hardware by freezing Phase 1 weights during Phase 2, reducing the overall computational footprint.

### Weaknesses (Areas for Improvement)

* **Lack of Perceptual Loss:** Relying purely on L1 and MSE loss functions can sometimes result in overly smooth or blurry textures. Adding a perceptual loss (e.g., VGG loss) or an adversarial discriminator (GAN) could improve human-perceived sharpness.
* **Evaluation Metrics:** The current pipeline relies heavily on loss curves and Absolute Error maps. Implementing standard super-resolution metrics like **PSNR (Peak Signal-to-Noise Ratio)** and **SSIM (Structural Similarity Index)** is necessary for standardized benchmarking.

## 📁 Repository Structure

* **`Extraction.py`**: Handles data preprocessing. Extracts frames from video/image sources and applies FFT to split inputs into low and high-frequency tensors.
* **`Models.py`**: Contains the PyTorch neural network architectures (`JEPAEncoder`, `JEPAPredictor`, and `JEPADecoder`) utilizing upsampling and GeLU activation layers.
* **`JEPA.ipynb`**: The main execution notebook. Orchestrates the two-phase training loop, manages model checkpoints, and handles the inference visualizations.
* **`config.py`** *(Required)*: Configuration file for hyperparameters, paths, and training variables.

## 🚀 Getting Started

### Prerequisites

Ensure you have the following dependencies installed in your Python environment:

```bash
pip install torch torchvision opencv-python numpy matplotlib

```

### Running the Project

1. Clone the repository and ensure your dataset is placed in the directories specified by your `config.py`.
2. Open `JEPA.ipynb` using Jupyter Notebook or VS Code.
3. Select **"Run All"**.
> **Note:** The notebook handles the sequential training of Phase 1 followed by Phase 2. Do not attempt to run it as a standard Python script.



---
<!-- 
**Author:** Nguyen Tuan Anh | Hanoi University of Science and Technology (HUST) -->