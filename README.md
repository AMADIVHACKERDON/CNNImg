# MNIST Interferometry → Convolutional Filtering → Autoencoder → RCS Classification

## Overview

This project implements an experimental image-classification pipeline that combines **artificial interferometry, convolutional filtering, a convolutional autoencoder, latent feature extraction, and a Residual Convolutional Structure (RCS)** for classification.

The project uses the **MNIST handwritten-digit dataset** as the input data.

The main objective is to investigate whether an image can be transformed into an artificial interferogram, processed through multiple convolutional filters, compressed into a learned hidden representation using an autoencoder, and finally classified using a residual convolutional neural network.

### Overall Pipeline

```text
MNIST Image
     │
     ▼
Artificial Interferometry
     │
     ▼
Artificial Interferogram
     │
     ▼
13 Convolutional Filters
     │
     ▼
13 Filtered Feature Channels
     │
     ▼
Convolutional Autoencoder
     │
     ├── Encoder
     │      │
     │      ▼
     │   Latent Representation
     │
     └── Decoder
            │
            ▼
       Reconstruction
     
Latent Representation
     │
     ▼
Residual Convolutional Structure (RCS)
     │
     ▼
Classification
     │
     ▼
MNIST Digit Prediction (0–9)
```

---

# 1. Project Objective

The purpose of this project is to develop and investigate a multi-stage feature-learning architecture.

Instead of sending the original MNIST image directly into a classifier, the image is first transformed into an **artificial interferogram**.

The interferogram is then processed using a collection of manually defined convolutional kernels.

The resulting filtered images form multiple channels that are given to a **Convolutional Autoencoder**.

The encoder learns a compressed representation of these filtered channels. This representation is then passed to a **Residual Network architecture** for classification.

The approach can therefore be summarized as:

> **Transform → Filter → Encode → Extract Features → Classify**

---

# 2. Technologies Used

The project is implemented in Python using:

* Python
* NumPy
* SciPy
* PyTorch
* Torchvision
* Matplotlib

### Main libraries

```python
import numpy as np
import math
from scipy.signal import convolve as Conv

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms

from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
```

---

# 3. Dataset

The project uses the **MNIST handwritten digit dataset**.

MNIST contains grayscale images of handwritten digits from:

```text
0 → 9
```

Each image has dimensions:

```text
28 × 28
```

and contains a single grayscale channel.

The training and testing datasets are loaded using Torchvision:

```python
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    transform=transform,
    download=True
)
```

The images are converted into PyTorch tensors using:

```python
transform = transforms.Compose([
    transforms.ToTensor()
])
```

---

# 4. Data Loading

The project uses PyTorch `DataLoader` objects.

```python
batch_size = 128

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)
```

The training loader randomly shuffles the training data, while the test loader does not shuffle the testing data.

---

# 5. Artificial Interferometry

One of the important parts of this project is the generation of an **artificial interferogram** from an MNIST image.

The original image is interpreted as an object amplitude.

First, the image is normalized:

```python
amplitude = Image / Image.max()
```

The normalized amplitude is converted into an object phase:

```python
phase_obj = 2 * np.pi * amplitude
```

A complex object wave is then generated:

```python
object_wave = amplitude * np.exp(
    1j * phase_obj
)
```

A reference wave is also generated.

The reference wave uses an angle:

```python
angle = np.pi / 6
```

and a spatial phase:

```python
phase_ref = (
    2 * np.pi *
    (
        filtered_x * np.cos(angle)
        + y * np.sin(angle)
    ) / 8
)
```

The reference wave is:

```python
ref_wave = np.exp(1j * phase_ref)
```

The object and reference waves are then interfered:

```python
interfero = np.abs(
    object_wave + ref_wave
) ** 2
```

The resulting intensity distribution is normalized:

```python
interfer = (
    interfero - interfero.min()
) / (
    interfero.max() - interfero.min()
)
```

This produces the artificial interferogram used by the subsequent filtering stage.

---

# 6. Convolutional Filtering

After the artificial interferogram is generated, it is passed through **13 manually defined convolution kernels**.

The filters are:

```text
K1
K2
K3
K4
K5
K6
K7
K8
K9
K10
K11
K12
K13
```

They include different types of image-processing operations such as:

* Identity filtering
* Edge detection
* Laplacian filtering
* Sharpening
* Smoothing
* Directional filtering
* Pattern extraction

For example:

```python
K3 = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])
```

is a Laplacian-style filter.

The filters are combined into a single array:

```python
K = np.dstack((
    K1, K2, K3, K4, K5,
    K6, K7, K8, K9, K10,
    K11, K12, K13
))
```

Therefore:

```text
Number of filters = 13
```

Each filter produces one output image.

The convolution operation is performed using SciPy:

```python
Out = Conv(
    interfer,
    K[:, :, k],
    mode="same"
)
```

The `same` mode keeps the spatial dimensions at:

```text
28 × 28
```

Therefore, after filtering:

```text
13 filters
       ↓
13 filtered images
       ↓
13 × 28 × 28
```

These 13 filtered images become the input channels of the convolutional autoencoder.

---

# 7. Filtered Image Representation

The filtered images are stacked using:

```python
filter_im = np.stack(
    filter_i,
    axis=0
)
```

The resulting representation is:

```text
13 × 28 × 28
```

where:

```text
13 = number of channels
28 = image height
28 = image width
```

It is then converted into a PyTorch tensor:

```python
filtered_image = torch.tensor(
    filter_im,
    dtype=torch.float32
)
```

---

# 8. Convolutional Autoencoder

The next stage is a **Convolutional Autoencoder**.

The purpose of the autoencoder is to learn a useful internal representation of the 13 filtered channels.

The autoencoder contains two major components:

```text
Autoencoder
│
├── Encoder
│
└── Decoder
```

---

# 9. Encoder

The encoder progressively transforms the 13-channel input into a deeper feature representation.

The first convolution receives:

```text
13 channels
```

and produces:

```text
16 channels
```

```python
nn.Conv2d(
    in_channels=13,
    out_channels=16,
    kernel_size=3,
    padding=1
)
```

A ReLU activation is then applied.

A max-pooling operation reduces the spatial dimensions.

The second convolution changes:

```text
16 channels → 32 channels
```

The third convolution changes:

```text
32 channels → 64 channels
```

The encoder therefore follows approximately:

```text
13 × 28 × 28
        ↓
16 × 28 × 28
        ↓ MaxPool
16 × 14 × 14
        ↓
32 × 14 × 14
        ↓ MaxPool
32 × 7 × 7
        ↓
64 × 7 × 7
```

The final encoder output is therefore approximately:

```text
64 × 7 × 7
```

This is the **latent representation** used later by the classifier.

---

# 10. Decoder

The decoder attempts to reconstruct the original 13 filtered channels from the latent representation.

The decoder uses transposed convolution:

```python
nn.ConvTranspose2d(
    in_channels=64,
    out_channels=32,
    kernel_size=2,
    stride=2
)
```

followed by:

```text
32 → 16
```

and finally:

```text
16 → 13
```

The final output therefore returns to approximately:

```text
13 × 28 × 28
```

The complete autoencoder can be viewed as:

```text
Input
13 × 28 × 28
      │
      ▼
Conv 13 → 16
      │
      ▼
MaxPool
      │
      ▼
Conv 16 → 32
      │
      ▼
MaxPool
      │
      ▼
Conv 32 → 64
      │
      ▼
Latent Representation
64 × 7 × 7
      │
      ▼
ConvTranspose 64 → 32
      │
      ▼
ConvTranspose 32 → 16
      │
      ▼
Conv 16 → 13
      │
      ▼
Reconstructed Image
13 × 28 × 28
```

---

# 11. Autoencoder Training

The autoencoder is trained using **Mean Squared Error (MSE)**:

```python
criterion = nn.MSELoss()
```

The optimizer is Adam:

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3
)
```

The current experiment uses:

```python
epochs = 5
```

During training, the model attempts to minimize:

```text
MSE(Input, Reconstruction)
```

The goal is for:

```text
Reconstructed filtered images
```

to become as similar as possible to:

```text
Original filtered images
```

---

# 12. Latent Feature Extraction

After training, the decoder is no longer required for the classification stage.

The encoder is used independently:

```python
z = model.encoder(filtered_x)
```

The resulting tensor represents the learned hidden features.

For the current architecture:

```text
Input:
13 × 28 × 28

Latent representation:
64 × 7 × 7
```

For multiple images, the latent tensor has the form:

```text
Number of images × 64 × 7 × 7
```

These latent feature maps are stored:

```python
latents.append(z.cpu())
```

The corresponding MNIST labels are stored separately:

```python
labels.append(label)
```

Finally:

```python
latents = torch.stack(latents)
labels = torch.tensor(labels)
```

---

# 13. Feature Maps

The latent representation contains **64 feature maps**.

Each feature map has spatial dimensions:

```text
7 × 7
```

Therefore, for one input image:

```text
64 × 7 × 7
```

can be interpreted as:

```text
64 learned feature maps
```

Each feature map represents information extracted by the convolutional encoder.

The project visualizes these feature maps to inspect what the encoder has learned.

---

# 14. Residual Convolutional Structure (RCS)

The latent features are then passed to a residual convolutional architecture.

The project defines a residual block:

```python
class blk(nn.Module):
```

The block contains three convolutional layers:

```text
1 × 1 convolution
      ↓
3 × 3 convolution
      ↓
1 × 1 convolution
```

This is commonly associated with the **bottleneck residual block** used in deeper ResNet architectures.

The block also contains Batch Normalization and ReLU activation.

Most importantly, the block contains an identity/skip connection:

```python
identity = x
```

and later:

```python
x += identity
```

This creates the residual connection.

Conceptually:

```text
             ┌──────────────────────┐
             │                      │
             │      Identity        │
             │                      │
Input ───────┼──► Conv → BN → ReLU ─┼──► Add → ReLU
             │       ↓              │
             │   Conv → BN → ReLU   │
             │       ↓              │
             │   Conv → BN          │
             │                      │
             └──────────────────────┘
```

---

# 15. ResNet Architecture

The project implements several possible residual network configurations:

```python
ResNet50()
ResNet101()
ResNet152()
```

The current classification experiment uses:

```python
ResNet50()
```

The ResNet-50 configuration contains:

```text
Layer 1: 3 residual blocks
Layer 2: 4 residual blocks
Layer 3: 6 residual blocks
Layer 4: 3 residual blocks
```

Therefore:

```text
[3, 4, 6, 3]
```

The classifier ends with:

```python
nn.Linear(
    512 * 4,
    num_classes
)
```

where:

```text
num_classes = 10
```

because MNIST contains ten digit classes.

---

# 16. Classification

The final classifier receives the latent feature representation.

The classification process is:

```text
MNIST
  ↓
Interferogram
  ↓
13 Filter Channels
  ↓
Autoencoder Encoder
  ↓
64 × 7 × 7 Latent Features
  ↓
Residual Network
  ↓
10 Output Classes
```

The output represents the ten MNIST classes:

```text
0
1
2
3
4
5
6
7
8
9
```

The classification loss is:

```python
criterion = nn.CrossEntropyLoss()
```

The predicted class is obtained using:

```python
prediction = torch.argmax(y, dim=1)
```

---

# 17. Training the RCS

The residual network is trained using Adam:

```python
optimizer = optim.Adam(
    net.parameters(),
    lr=1e-3
)
```

The current experiment uses:

```python
20 epochs
```

At each epoch:

1. The latent features are passed to the RCS.
2. The network produces class predictions.
3. Cross-entropy loss is calculated.
4. Gradients are calculated.
5. Network parameters are updated.
6. The predicted labels are displayed.

The process is:

```text
Latent Features
      ↓
ResNet
      ↓
Prediction
      ↓
Cross Entropy Loss
      ↓
Backpropagation
      ↓
Parameter Update
```

---

# 18. Evaluation Mode

After training, the network is switched to evaluation mode:

```python
net.eval()
```

and gradient calculation is disabled:

```python
with torch.no_grad():
```

The model then performs a final forward pass:

```python
y = net(x)
```

and generates the final predictions:

```python
prediction = torch.argmax(y, dim=1)
```

The final loss and predictions are then displayed.

---

# 19. Current Experimental Dataset Flow

The current implementation extracts latent representations from:

```python
for i in range(32):
```

Therefore, the classification experiment currently uses **32 MNIST samples**.

Their labels are stored in:

```python
labels
```

and their latent representations are stored in:

```python
latents
```

The resulting tensors are approximately:

```text
latents:
32 × 64 × 7 × 7

labels:
32
```

This is currently an experimental subset rather than the complete MNIST dataset.

---

# 20. Project Architecture

The complete architecture can be represented as:

```text
                  MNIST
                    │
                    │ 28 × 28
                    ▼
          ┌─────────────────────┐
          │ Artificial          │
          │ Interferometry      │
          └──────────┬──────────┘
                     │
                     ▼
             Artificial
             Interferogram
              28 × 28
                     │
                     ▼
        ┌────────────────────────┐
        │ 13 Manual Convolution  │
        │ Filters                │
        └────────────┬───────────┘
                     │
                     ▼
              13 × 28 × 28
                     │
                     ▼
        ┌────────────────────────┐
        │ Convolutional          │
        │ Autoencoder            │
        └────────────┬───────────┘
                     │
                     ▼
               64 × 7 × 7
            Latent Features
                     │
                     ▼
        ┌────────────────────────┐
        │ Residual Convolutional │
        │ Structure / ResNet     │
        └────────────┬───────────┘
                     │
                     ▼
                 10 Classes
                     │
                     ▼
              MNIST Prediction
```

---

# 21. Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

### Windows

```bash
venv\Scripts\activate
```

### Linux/macOS

```bash
source venv/bin/activate
```

Install the required packages:

```bash
pip install numpy scipy matplotlib torch torchvision
```

---

# 22. Running the Project

Run the Python script:

```bash
python main.py
```

The program will:

1. Download MNIST if it is not already available.
2. Load the training and testing datasets.
3. Generate artificial interferograms.
4. Apply the 13 convolutional filters.
5. Display the filtered images.
6. Train the convolutional autoencoder.
7. Extract latent representations.
8. Visualize latent feature maps.
9. Create the residual network.
10. Train the classifier.
11. Display predictions and losses.
12. Evaluate the trained network.

---

# 23. Hardware Acceleration

The project automatically checks whether CUDA is available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)
```

Therefore:

```text
CUDA GPU available
        ↓
PyTorch uses GPU

CUDA unavailable
        ↓
PyTorch uses CPU
```

This allows the same code to run on either CPU or NVIDIA CUDA-enabled GPU systems.

---

# 24. Important Tensor Dimensions

Understanding the tensor dimensions is important for this project.

| Stage                         | Representation |
| ----------------------------- | -------------- |
| Original MNIST image          | `1 × 28 × 28`  |
| Artificial interferogram      | `28 × 28`      |
| Number of filters             | `13`           |
| Filtered image representation | `13 × 28 × 28` |
| Encoder first output          | `16 × 28 × 28` |
| First pooled representation   | `16 × 14 × 14` |
| Second convolution            | `32 × 14 × 14` |
| Second pooled representation  | `32 × 7 × 7`   |
| Latent representation         | `64 × 7 × 7`   |
| Number of classes             | `10`           |

---

# 25. Why Use the Autoencoder?

The autoencoder is used to learn a more compact representation of the filtered images.

Instead of directly sending:

```text
13 × 28 × 28
```

into the classifier, the encoder transforms the information into:

```text
64 × 7 × 7
```

The latent representation contains features learned from the filtered interferogram.

The decoder is used during autoencoder training to force the encoder to retain information necessary for reconstruction.

After training, the encoder can be used as a feature extractor.

---

# 26. Why Use Multiple Filters?

Each filter responds differently to structures in the interferogram.

For example, different filters can emphasize:

* Edges
* Fine details
* Smooth regions
* High-frequency structures
* Directional structures
* Local patterns

Using 13 filters produces 13 different views of the same interferogram.

These outputs are treated as separate channels:

```text
Interferogram
     │
     ├── Filter 1 → Channel 1
     ├── Filter 2 → Channel 2
     ├── Filter 3 → Channel 3
     ├── ...
     └── Filter 13 → Channel 13
```

This gives the convolutional autoencoder multiple representations of the same input.

---

# 27. Why Use Residual Connections?

Deep convolutional networks can become difficult to optimize as their depth increases.

Residual connections provide an alternative path through which information and gradients can flow.

The basic residual operation is:

```text
Output = F(x) + x
```

where:

```text
x = original input
F(x) = learned transformation
```

The architecture used in this project therefore allows the classifier to learn residual transformations rather than relying entirely on a sequence of plain convolutions.

---

# 28. Visualization

The project generates several visual outputs.

### Artificial Interferogram

```text
Original MNIST
      ↓
Artificial Interferometry
      ↓
Artificial Interferogram
```

### Filter Responses

The 13 filter outputs are displayed in a grid.

```text
Filter 1   Filter 2   Filter 3
Filter 4   Filter 5   Filter 6
...
Filter 13
```

### Autoencoder Reconstruction

The project compares the filtered inputs against their reconstructed outputs.

```text
Original filtered channels
          vs.
Reconstructed channels
```

### Latent Feature Maps

The encoder produces:

```text
64 feature maps
```

which can be visualized to inspect the learned representation.

---

# 29. Experimental Nature of the Project

This project is currently an **experimental research implementation**.

The primary goal is to understand and evaluate the interaction between:

```text
Interferometry
+
Convolutional Filtering
+
Representation Learning
+
Residual Networks
```

The current implementation is not yet intended to represent a fully optimized production classification system.

In particular, the current experiment uses a relatively small subset of samples during latent feature extraction.

Future experiments should evaluate the complete training and testing datasets.

---

# 30. Future Improvements

Potential improvements include:

### Dataset expansion

Use the complete MNIST training dataset instead of only a small subset.

### Automated filter learning

Compare manually designed filters against learnable convolutional filters.

### Better autoencoder training

Train the autoencoder on a large collection of filtered interferograms instead of a single filtered representation.

### Classification evaluation

Add:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion matrix

### Model comparison

Compare:

```text
Original MNIST → CNN
```

against:

```text
MNIST → Interferometry → Filters → Autoencoder → RCS
```

### Latent-space analysis

Investigate whether the latent representations separate different MNIST classes.

### Hyperparameter optimization

Experiment with:

* Learning rate
* Batch size
* Number of filters
* Number of latent channels
* Number of epochs
* Filter sizes
* Autoencoder architecture
* RCS depth

---

# 31. Research Question

The central research question behind this implementation can be summarized as:

> **Can artificial interferometric representations combined with convolutional filtering and learned latent representations improve or provide useful features for residual convolutional image classification?**

The project provides an experimental framework for investigating this question.

---

# 32. Summary

The project implements the following complete workflow:

```text
MNIST
  ↓
Artificial Interferometry
  ↓
Artificial Interferogram
  ↓
13 Handcrafted Convolution Filters
  ↓
13-Channel Filtered Representation
  ↓
Convolutional Autoencoder
  ↓
64 × 7 × 7 Latent Representation
  ↓
Residual Convolutional Structure
  ↓
10-Class Classification
```

The key idea is to **transform the original image before classification**, allowing the classifier to operate on a learned representation of multiple filtered interferometric channels rather than directly on the original MNIST image.

---

# 33. License

Add your preferred license here.

For example:

```text
MIT License
```

---

# Author

**AMADI VICTORY**

This project is part of an experimental investigation into:

* Artificial Interferometry
* Convolutional Neural Networks
* Autoencoders
* Feature Extraction
* Residual Convolutional Structures
* Image Classification
* Deep Learning
