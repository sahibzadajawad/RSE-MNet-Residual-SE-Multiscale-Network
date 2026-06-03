# A Next-Generation Multi-Scale CNN with Residual Gradient Stability and Adaptive Channel Recalibration Using RSE-MNet

Official PyTorch implementation of **A Next-Generation Multi-Scale CNN with Residual Gradient Stability and Adaptive Channel Recalibration Using RSE-MNet)**, a novel deep convolutional architecture that integrates **Residual Learning**, **Squeeze-and-Excitation (SE) Channel Attention**, and **Multi-Scale Convolutional Feature Extraction** into a unified framework for fine-grained visual recognition.

RSE-MNet is designed to address the limitations of conventional convolutional neural networks by simultaneously improving gradient propagation, channel-wise feature discrimination, and multi-scale spatial representation learning. The architecture combines residual identity mappings, adaptive channel recalibration, and heterogeneous convolutional branches (3×3, 5×5, and 7×7) to capture both local and global contextual information.

The proposed framework was extensively evaluated on the CIFAR-100 benchmark dataset using advanced optimization and regularization strategies, including MixUp augmentation, Label Smoothing, Exponential Moving Average (EMA), and Cosine Annealing Warm Restarts. Experimental results demonstrate that RSE-MNet achieves superior classification performance while maintaining a practical accuracy–efficiency trade-off.

---

# Project Overview

Fine-grained image classification remains a challenging task due to high inter-class similarity and significant intra-class variation. Traditional CNN architectures often struggle to simultaneously capture discriminative local features and broader contextual information.

RSE-MNet addresses this challenge through the joint integration of:

* Residual Learning for stable optimization and efficient gradient flow.
* Squeeze-and-Excitation Attention for adaptive channel recalibration.
* Multi-Scale Convolutional Branches for capturing diverse spatial patterns.
* Feature Fusion Mechanisms for improved representation learning.

The resulting architecture provides a powerful and scalable framework for fine-grained visual recognition tasks.

---

# Core Components

## 1. Residual Learning

Residual identity shortcuts enable efficient information propagation across layers, mitigate vanishing gradient problems, and facilitate deeper network optimization.

## 2. Squeeze-and-Excitation Attention

SE blocks dynamically recalibrate channel responses by emphasizing informative feature maps and suppressing redundant activations.

## 3. Multi-Scale Feature Extraction

Parallel convolutional branches with kernel sizes:

* 3 × 3
* 5 × 5
* 7 × 7

allow the network to simultaneously capture local details and broader contextual structures.

## 4. Feature Fusion

Features extracted from different receptive fields are integrated through residual fusion mechanisms to produce richer and more discriminative representations.

---

# Dataset

Experiments were conducted on the CIFAR-100 benchmark dataset.

Dataset Characteristics:

| Property        | Value   |
| --------------- | ------- |
| Classes         | 100     |
| Training Images | 50,000  |
| Test Images     | 10,000  |
| Resolution      | 32 × 32 |
| Image Type      | RGB     |

The dataset is automatically downloaded using torchvision.

---

# Training Strategy

The training pipeline incorporates several modern deep learning techniques:

### MixUp Augmentation

Improves generalization by generating interpolated training samples.

### Label Smoothing

Reduces model overconfidence and improves calibration.

### Exponential Moving Average (EMA)

Maintains a smoothed version of model weights for more stable evaluation.

### Cosine Annealing Warm Restarts

Provides adaptive learning-rate scheduling and improves convergence behavior.

### Fine-Tuning Phase

A dedicated fine-tuning stage further refines the learned feature representations and improves final classification accuracy.

---

# Experimental Performance

## CIFAR-100 Top-1 Accuracy

| Model                   | Top-1 Accuracy (%) |
| ----------------------- | ------------------ |
| MobileNetV2             | 73.52              |
| ShuffleNetV2            | 74.81              |
| ResNet-50               | 75.42              |
| EfficientNet-B0         | 77.68              |
| DenseNet-121            | 77.94              |
| SENet-50                | 77.86              |
| CBAM-ResNet             | 78.65              |
| Multihead-Res-SE        | 79.14              |
| **RSE-MNet (Proposed)** | **80.78**          |

---

# Repository Structure

```text
RSE-MNet-CIFAR100/
│
├── train_rse_mnet.py
├── requirements.txt
├── README.md
├── LICENSE
├── CITATION.cff
├── .gitignore
│
├── checkpoints/
│   └── best_ema_model.pth
│
├── results/
│   ├── confusion_matrix.png
│   ├── classification_report.csv
│   ├── ema_accuracy_curve.png
│   └── sample_predictions.png
│
├── figures/
│   ├── architecture.png
│   ├── training_curves.png
│   └── gradcam_visualizations.png
│
└── docs/
    └── paper.pdf
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/RSE-MNet-CIFAR100.git

cd RSE-MNet-CIFAR100
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Training

Run model training:

```bash
python train_rse_mnet.py
```

---

# Evaluation

The repository supports:

* Top-1 Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix
* Classification Report
* Sample Prediction Visualization

---

# Research Contribution

RSE-MNet demonstrates that residual learning, squeeze-and-excitation attention, and multi-scale feature extraction can be effectively unified within a single architecture to improve fine-grained image classification performance.

The framework achieves a strong balance between representational power, optimization stability, and computational efficiency, making it suitable for a wide range of computer vision applications.

---

# Citation

If you use this repository in your research, please cite the associated publication.

```bibtex
@article{RSEMNet2026,
  title={A Next-Generation Multi-Scale CNN with Residual Gradient Stability and Adaptive Channel Recalibration Using RSE-MNet},
  author={Sahibzada Jawad Hadi, Irfan Ahmed, Abid Iqbal, Saad Arif},
  journal={PLOS ONE},
  year={2026}
}
```

---

# Author

**Sahibzada Jawad Hadi**

---

# License

This project is released under the MIT License.

See the LICENSE file for additional details.
