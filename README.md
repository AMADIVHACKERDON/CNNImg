# CNNImg
Classification of satellite images
# NetworkData_Model

## Artificial Interferogram Generation, Multi-Channel Filtering and Convolutional Autoencoder

## Overview

**NetworkData_Model** is an experimental computer vision project that investigates how image information can be transformed through **artificial interferometry, spatial filtering, and convolutional representation learning**.

The project uses the **MNIST handwritten-digit dataset** as the input source. Each image is transformed into an artificial interferogram by combining an object wave derived from the image with a reference wave. The resulting interferogram is then processed through a set of **13 manually designed convolution kernels**.

The filtered outputs are treated as separate channels and passed into a **Convolutional Autoencoder (CAE)**. The encoder progressively transforms the 13-channel filtered representation into a lower-resolution, higher-dimensional hidden representation.

The main objective is to investigate how image information changes through:

```text
MNIST Image
     │
     ▼
Object Wave
     │
     +─────────────── Reference Wave
     │                       │
     └──────────────┬────────┘
                    ▼
          Artificial Interferogram
                    │
                    ▼
             13 Spatial Filters
                    │
                    ▼
          13-Channel Representation
                    │
                    ▼
       Convolutional Autoencoder
                    │
                    ▼
           Hidden Representation
                    │
                    ▼
              Feature Maps
```

---

# Project Objective

The purpose of this project is to explore a processing pipeline in which a conventional image is transformed into an **interference-based representation**, filtered using multiple spatial kernels, and subsequently encoded into a learned hidden representation.

The project focuses on understanding:

* Artificial interferogram generation
* Wave-based image representation
* Spatial image filtering
* Multi-channel image representations
* Convolutional neural networks
* Autoencoder architectures
* Hidden representations
* Feature-map visualization

This project is primarily an **experimental image-processing and representation-learning study**.

---

# 1. MNIST Dataset

The project uses the **MNIST handwritten-digit dataset** provided through `torchvision`.

MNIST contains grayscale handwritten-digit images with dimensions:

```text
28 × 28 pixels
```

Each image contains a digit from:

```text
0 → 9
```

The images are loaded using PyTorch's `torchvision.datasets.MNIST`.

```python
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True
)
```

The images are converted to PyTorch tensors using:

```python
transforms.ToTensor()
```

---

# 2. Artificial Interferogram Generation

Instead of directly feeding the MNIST image into the filtering stage, the image is first converted into an artificial optical-wave representation.

The image intensity is normalized and used as the amplitude of an object wave.

```python
amplitude = Image / Image.max()
```

A phase term is then generated:

```python
phase_obj = 2 * np.pi * amplitude
```

The object wave is represented as a complex-valued wave:

```python
object_wave = amplitude * np.exp(1j * phase_obj)
```

A reference wave is also generated using a defined propagation angle:

```python
angle = np.pi / 6
```

The reference wave is constructed from the spatial coordinates of the image.

The object and reference waves are then combined:

```python
interfero = np.abs(object_wave + ref_wave) ** 2
```

This produces the artificial interference pattern.

The result is normalized to produce the final artificial interferogram:

```python
interfer =
    (interfero - interfero.min()) /
    (interfero.max() - interfero.min())
```

Conceptually:

```text
Object Wave + Reference Wave
             │
             ▼
       Wave Interference
             │
             ▼
     Artificial Interferogram
```

---

# 3. Spatial Filtering

The artificial interferogram is processed using **13 manually defined 3 × 3 kernels**.

The kernels include different spatial operations such as:

* Identity filtering
* Edge detection
* Laplacian filtering
* Sharpening
* Averaging
* Gaussian-like smoothing
* Directional filtering
* Diagonal structures
* Vertical structures

The filters are defined explicitly rather than learned by the network.

For example:

```python
K1 = np.array([
    [0,0,0],
    [0,1,0],
    [0,0,0]
])
```

Another kernel performs a Laplacian-like operation:

```python
K3 = np.array([
    [0,1,0],
    [1,-4,1],
    [0,1,0]
])
```

The complete collection of kernels is stacked into a single array:

```python
K = np.dstack(
    (K1,K2,K3,K4,K5,K6,K7,
     K8,K9,K10,K11,K12,K13)
)
```

Therefore:

```text
Number of filters = 13
```

Each filter receives the same artificial interferogram and produces a different filtered representation.

---

# 4. Multi-Channel Representation

Each of the 13 filters produces a separate 28 × 28 image.

Therefore, the filtered representation can be interpreted as:

```text
13 channels × 28 × 28
```

Conceptually:

```text
Artificial Interferogram
          │
 ┌────────┼────────┐
 ▼        ▼        ▼
Filter 1 Filter 2 ... Filter 13
 │        │             │
 ▼        ▼             ▼
28×28    28×28         28×28
 │        │             │
 └────────┼─────────────┘
          ▼
      13 × 28 × 28
```

This is an important part of the project because the original MNIST image has only **one channel**, while the filtered representation contains **13 channels**.

---

# 5. Convolutional Autoencoder

The 13-channel filtered representation is used as the input to a convolutional autoencoder.

The autoencoder contains two major components:

```text
Encoder
   │
   ▼
Hidden Representation
   │
   ▼
Decoder
```

The objective of the autoencoder is to reconstruct its input while forcing the encoder to learn a compact representation of the filtered information.

---

# 6. Encoder

The encoder contains three convolutional stages.

### First convolution

```text
13 channels → 16 channels
```

followed by:

* ReLU
* 2 × 2 Max Pooling

### Second convolution

```text
16 channels → 32 channels
```

followed by:

* ReLU
* 2 × 2 Max Pooling

### Third convolution

```text
32 channels → 64 channels
```

followed by:

* ReLU

The overall encoder can therefore be represented as:

```text
13 × 28 × 28
       │
       ▼
Conv2D: 13 → 16
       │
      ReLU
       │
   MaxPool
       │
       ▼
16 × 14 × 14
       │
       ▼
Conv2D: 16 → 32
       │
      ReLU
       │
   MaxPool
       │
       ▼
32 × 7 × 7
       │
       ▼
Conv2D: 32 → 64
       │
      ReLU
       │
       ▼
64 × 7 × 7
```

The resulting tensor is the **latent/hidden representation** produced by the encoder.

---

# 7. Decoder

The decoder attempts to reconstruct the original 13-channel filtered representation.

It uses transposed convolution layers to increase the spatial dimensions.

```text
64 × 7 × 7
      │
      ▼
ConvTranspose2D
      │
      ▼
32 × 14 × 14
      │
      ▼
ConvTranspose2D
      │
      ▼
16 × 28 × 28
      │
      ▼
Conv2D
      │
      ▼
13 × 28 × 28
```

A sigmoid activation is applied to the final output.

---

# 8. Reconstruction Loss

The autoencoder uses **Mean Squared Error (MSE)** as its reconstruction loss:

```python
criterion = nn.MSELoss()
```

The objective is to minimize the difference between:

```text
Original filtered representation
              ↓
        Autoencoder
              ↓
Reconstructed representation
```

Mathematically, the reconstruction objective can be represented as:

$$
L = \frac{1}{N}\sum_{i=1}^{N}(x_i-\hat{x}_i)^2
$$

where:

* \(x\) = original filtered input
* \(\hat{x}\) = reconstructed output
* \(N\) = number of elements

The optimizer used is Adam:

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3
)
```

The network is trained for:

```text
5 epochs
```

---

# 9. Hidden Representation

After training, the decoder is not required when investigating the learned representation.

Instead, the filtered image is passed through the encoder:

```python
z = model.encoder(filtered_x)
```

The resulting tensor represents the learned hidden features.

The project records these representations in:

```python
latents
```

and combines them using:

```python
latents = torch.cat(latents, dim=0)
```

The hidden representation has the structure:

```text
64 × 7 × 7
```

for each encoded sample.

This means that the encoder transforms the original 13-channel filtered representation into **64 learned feature maps**, each with spatial dimensions of **7 × 7**.

---

# 10. Feature Maps

The project attempts to visualize the learned feature maps produced by the encoder.

Each feature map represents a different learned response within the network.

Conceptually:

```text
Input
13 × 28 × 28
      │
      ▼
Convolution
      │
      ▼
16 Feature Maps
      │
      ▼
Convolution
      │
      ▼
32 Feature Maps
      │
      ▼
Convolution
      │
      ▼
64 Feature Maps
      │
      ▼
Hidden Representation
```

The final feature representation is therefore substantially different from the original MNIST image.

---

# Project Pipeline

The complete experimental pipeline is:

```text
                 MNIST
                   │
                   ▼
            28 × 28 Image
                   │
                   ▼
             Normalization
                   │
                   ▼
              Object Wave
                   │
                   │
          Reference Wave
                   │
                   ▼
        Wave Interference
                   │
                   ▼
      Artificial Interferogram
                   │
                   ▼
       ┌─────────────────────┐
       │   13 Fixed Filters  │
       └─────────────────────┘
                   │
                   ▼
          13 × 28 × 28
        Filtered Channels
                   │
                   ▼
        Convolutional Encoder
                   │
                   ▼
          64 × 7 × 7
      Hidden Representation
                   │
                   ▼
         Feature Maps
                   │
                   ▼
       Convolutional Decoder
                   │
                   ▼
          Reconstruction
```

---

# Important Distinction

This project contains **two different types of processing**.

### Fixed image-processing filters

The 13 filters are manually defined kernels.

They are **not learned by the neural network**.

Their purpose is to transform the artificial interferogram into different spatial representations.

### Learned representation

The convolutional autoencoder contains trainable parameters.

During training, the encoder learns how to represent the 13-channel filtered input in a hidden feature space while the decoder learns to reconstruct the input.

Therefore, the project can be understood as:

```text
Fixed Spatial Filtering
          +
Learned Convolutional Representation
```

---

# Technologies

The project uses:

* Python
* NumPy
* SciPy
* PyTorch
* Torchvision
* Matplotlib

### NumPy

Used for numerical operations, wave calculations, and construction of the manually defined filtering kernels.

### SciPy

The `scipy.signal.convolve` function is used to perform spatial convolution between the artificial interferogram and each manually defined kernel.

### PyTorch

Used to construct and train the convolutional autoencoder.

### Torchvision

Used to download and load the MNIST dataset.

### Matplotlib

Used to visualize:

* Artificial interferograms
* Filter outputs
* Reconstruction results
* Hidden representations
* Feature maps

---

# Installation

Clone the repository:

```bash
git clone https://github.com/AMADIVHACKERDON/NetworkData_Model.git
cd NetworkData_Model
```

Install the required dependencies:

```bash
pip install numpy scipy torch torchvision matplotlib
```

---

# Running the Project

Run the Python script containing the implementation:

```bash
python your_script.py
```

The MNIST dataset will automatically be downloaded through `torchvision` if it is not already available.

The program will generate visualizations of the filtering process and the autoencoder representation.

---

# Hardware Acceleration

The project automatically checks whether CUDA is available:

```python
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)
```

If a CUDA-compatible GPU is available, PyTorch can use it for computation. Otherwise, the project runs on the CPU.

---

# Current Experimental Results

The current implementation demonstrates:

* Generation of artificial interferograms from MNIST images.
* Application of 13 manually designed spatial filters.
* Construction of a 13-channel filtered representation.
* Convolutional encoding of the filtered representation.
* Reconstruction using a convolutional decoder.
* Extraction of hidden representations.
* Generation of 64 learned feature maps at a spatial resolution of 7 × 7.

The project is currently an **experimental research implementation**, with further work required to systematically evaluate the learned representation across the complete MNIST dataset.

---

# Future Work

Future development of this project may include:

* Processing the complete MNIST training and test datasets.
* Applying the interferogram generation process to every image.
* Building a complete dataset of filtered interferograms.
* Improving the organization of the filtering pipeline.
* Comparing different manually designed filter banks.
* Visualizing all intermediate representations.
* Quantitatively evaluating reconstruction quality.
* Investigating the latent representation produced by the encoder.
* Connecting the learned representation to subsequent analysis stages.
* Exploring residual convolutional structures after the hidden representation.

---

# Research Direction

The broader direction of this project is to investigate whether **interferometric representations combined with fixed spatial filtering and learned convolutional representations** can provide useful intermediate features for subsequent image analysis.

The current implementation therefore focuses on understanding the transformation of information at each stage rather than performing final classification or prediction.

---

# Author

**Victory Amadi**

GitHub:
https://github.com/AMADIVHACKERDON

---

# License

This project is intended for educational, experimental, and research purposes.
