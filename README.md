# MRI Brain Tumor Segmentation using UNet with KAN Networks

This project implements a **UNet model** for MRI brain tumor segmentation using **KAN (Kernel Activation Networks)**. The goal is to accurately segment tumor regions in MRI scans to assist in medical diagnosis and research.

---

## Features
- **UNet Architecture**: A popular encoder-decoder architecture for image segmentation.
- **KAN Layers**: Replacing standard convolutions with Kernel Activation Networks for improved expressiveness.
- **Binary Segmentation**: Segmenting tumor vs. non-tumor regions.
- **Visualization Tools**: Output segmentation masks and overlays for intuitive analysis.

---

## Dataset
This model is trained and evaluated on an MRI brain tumor dataset. Each MRI scan is of size `3x256x256` (RGB format).

### Preprocessing
- Images are resized to `256x256` to match the input size of the model.
- Normalization is applied to bring pixel values to the range `[0, 1]`.

---

## Installation
Follow these steps to set up the project:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mohamed-12-4/unetkan
   ```

2. install packages.
```bash
pip install -r requirements.txt
```
3. clone KAN repo This project uses the torch-conv-kan repo for Conv-Kan layers
```bash
git clone https://github.com/IvanDrokin/torch-conv-kan.git
```

