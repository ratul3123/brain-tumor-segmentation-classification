# Brain Tumor Segmentation & Classification

### Using U-Net & Attention U-Net on BRISC 2025 Dataset

---

## Project Overview

This project presents a **multi-task deep learning system** for brain tumor analysis using the **BRISC 2025 dataset**. The system performs:

* **Tumor Segmentation** (pixel-wise prediction)
* **Tumor Classification** (glioma, meningioma, pituitary, no tumor)

Two architectures are implemented and compared:

* **U-Net (Baseline)**
* **Attention U-Net (Enhanced Model)**

The project also explores **joint vs separate training strategies** for multi-task learning.

---

## Key Features

* Multi-task learning (segmentation + classification)
* Attention mechanisms for improved localization
* Efficient data pipeline using `tf.data`
* Reproducible training setup
* Comparative architectural analysis

---

## Model Architectures

### 1️⃣ U-Net (Baseline)

A standard encoder-decoder architecture:

#### 🔻 Encoder (Contracting Path)

* 4 stages of **DoubleConv blocks**
* Each block:
  Conv (3×3) → BatchNorm → ReLU (×2)
* **MaxPooling (2×2)** for downsampling

#### 🔸 Bottleneck + Classification Head

* Deepest feature layer
* **Global Average Pooling (GAP)**
* Fully connected layer for **4-class classification**

#### 🔺 Decoder (Expansive Path)

* **Conv2DTranspose** for upsampling
* **Skip connections** to recover spatial details

---

### 2️⃣ Attention U-Net (Enhanced Model)

Improves U-Net using **Attention Gates (AGs)** to refine skip connections.

#### Motivation

Standard skip connections pass all features, including irrelevant background information.

#### ⚡ Attention Mechanism

Each Attention Gate:

* Takes encoder features (`x`) and decoder features (`g`)
* Applies **1×1 convolutions**
* Combines them and generates an **attention mask (sigmoid)**
* Filters encoder features before passing to decoder

#### Integration

* 4 Attention Gates in the decoder
* Same classification head at bottleneck

---

## Training Strategies

### Joint Training

Both tasks are trained simultaneously.

**Loss Function:**

```
Total Loss = Segmentation Loss (Binary Crossentropy) 
           + Classification Loss (Sparse Categorical Crossentropy)
```

**Advantages:**

* Shared feature learning
* Implicit regularization

---

### Separate Training

Two-stage training approach:

1. Train full model for segmentation
2. Freeze encoder & decoder
3. Train classification head separately

**Advantages:**

* Reduces task interference
* Allows task-specific optimization

---

## Data Pipeline & Preprocessing

* Resize images to **256 × 256**
* Normalize pixel values to **[-1, 1]**
* Optimized `tf.data` pipeline:

  * `AUTOTUNE`
  * Prefetching for performance
* Reproducibility:

  * Fixed random seeds
  * `TF_DETERMINISTIC_OPS` enabled

---

## Evaluation Metrics

### Segmentation

* **Intersection over Union (IoU)**
* **Dice Coefficient**

### Classification

* **Sparse Categorical Crossentropy**
* **Confusion Matrix**

---

## Model Summary

| Component           | Details                 |
| ------------------- | ----------------------- |
| Input Size          | 256 × 256 × 3           |
| Encoder             | DoubleConv Blocks       |
| Bottleneck Filters  | 256                     |
| Classification Head | GAP → Dense (4 classes) |
| Upsampling          | Conv2DTranspose         |
| Segmentation Output | Sigmoid Activation      |

---

## Results & Insights

* Attention U-Net improves:

  * Tumor localization
  * Boundary precision (IoU)
* Faster convergence due to focused feature learning
* Multi-task setup enhances encoder representations

---

## Tech Stack

* TensorFlow / Keras
* Python
* NumPy, Matplotlib
* scikit-learn
