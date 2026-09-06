Interferometric Image Representation and Residual Convolutional Classification

Overview

This project is an experimental deep-learning and image-processing pipeline that investigates how image information can be transformed through artificial interferometry, multi-channel spatial filtering, convolutional representation learning, and residual convolutional classification.

The project currently uses the MNIST handwritten-digit dataset as the input dataset.

Instead of sending the original MNIST image directly into a classification network, the image passes through several processing stages.

The complete pipeline is:

MNIST Image
     │
     ▼
Artificial Interferometry
     │
     ▼
Artificial Interferogram
     │
     ▼
13 Fixed Spatial Filters
     │
     ▼
13-Channel Filtered Representation
     │
     ▼
Convolutional Autoencoder
     │
     ▼
64 × 7 × 7 Hidden Representation
     │
     ▼
Residual Convolutional Structure
     │
     ▼
10-Class Classification

The main purpose of the project is to investigate how an image can be transformed into a different representation before being used for classification.

---

Project Objective

The project investigates the following processing sequence:

Image
  ↓
Wave Representation
  ↓
Interference
  ↓
Spatial Filtering
  ↓
Multi-Channel Representation
  ↓
Learned Feature Representation
  ↓
Residual Convolutional Classification

The project combines two major approaches:

1. Fixed image processing
   
   - Artificial interferogram generation
   - Manually designed convolution filters

2. Learned deep representation
   
   - Convolutional autoencoder
   - Residual convolutional network

The objective is to understand how information changes at each stage and whether the resulting learned representation can be used for image classification.

---

1. Dataset

The project uses the MNIST handwritten-digit dataset through "torchvision".

MNIST contains:

- 60,000 training images
- 10,000 test images
- 10 digit classes
- Grayscale images
- Image dimensions of "28 × 28"

The classes are:

0  1  2  3  4
5  6  7  8  9

The dataset is loaded using:

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

The images are converted into tensors using:

transform = transforms.Compose([
    transforms.ToTensor()
])

An original MNIST image therefore has the shape:

1 × 28 × 28

where:

1  = grayscale channel
28 = height
28 = width

---

2. Artificial Interferometry

The first major transformation is the generation of an artificial interferogram.

The original MNIST image is treated as the amplitude information of an object wave.

First, the image is normalized:

amplitude = Image / Image.max()

The normalized image is then used to generate an object phase:

phase_obj = 2 * np.pi * amplitude

A complex object wave is constructed:

object_wave = amplitude * np.exp(
    1j * phase_obj
)

A reference wave is then generated.

The reference wave uses an angle:

angle = np.pi / 6

and spatial coordinates to construct its phase:

phase_ref = (
    2 * np.pi *
    (
        filtered_x * np.cos(angle)
        + y * np.sin(angle)
    ) / 8
)

The reference wave is:

ref_wave = np.exp(1j * phase_ref)

The object wave and reference wave are combined:

interfero = np.abs(
    object_wave + ref_wave
) ** 2

This produces an artificial interference intensity pattern.

The interference pattern is then normalized:

interfer = (
    interfero - interfero.min()
) / (
    interfero.max() - interfero.min()
)

The result is the artificial interferogram.

The transformation can be represented as:

MNIST Image
     │
     ▼
Amplitude
     │
     ▼
Object Wave
     │
     │
     ├───────────────┐
     │               │
     │          Reference Wave
     │               │
     └───────┬───────┘
             ▼
       Wave Interference
             │
             ▼
    Artificial Interferogram

The artificial interferogram remains spatially:

28 × 28

---

3. Fixed Spatial Filters

The artificial interferogram is processed using 13 manually designed convolution kernels.

Each kernel has a size of:

3 × 3

The filters are:

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

They are stacked using:

K = np.dstack((
    K1, K2, K3, K4, K5,
    K6, K7, K8, K9, K10,
    K11, K12, K13
))

Therefore:

Number of filters = 13

These filters are fixed filters. Their values are manually defined and are not learned during neural-network training.

For example, the identity-style filter is:

K1 = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])

A Laplacian-style filter is:

K3 = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
])

Another filter performs averaging:

K6 = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]
])

K6 = K6 / 9

---

4. Multi-Channel Representation

Each filter operates on the same artificial interferogram.

The convolution is performed using:

Out = Conv(
    interfer,
    K[:, :, k],
    mode="same"
)

Because "mode="same"" is used, each filtered image remains:

28 × 28

Since there are 13 filters, the resulting representation is:

13 × 28 × 28

This can be interpreted as a 13-channel image.

                 Artificial Interferogram
                         28 × 28
                            │
          ┌─────────────────┼─────────────────┐
          │                 │                 │
          ▼                 ▼                 ▼
       Filter 1          Filter 2          Filter 3
          │                 │                 │
        28×28             28×28             28×28
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │
                           ...
                            │
                            ▼
                        Filter 13
                            │
                          28×28
                            │
                            ▼
                      13 × 28 × 28

The original MNIST image has one channel, while the filtered representation contains 13 channels.

---

5. Convolutional Autoencoder

The 13-channel filtered representation is then provided to a Convolutional Autoencoder (CAE).

The autoencoder contains:

Encoder
   │
   ▼
Hidden Representation
   │
   ▼
Decoder

The encoder learns a representation of the filtered image while the decoder attempts to reconstruct the original filtered representation.

The architecture is:

13 × 28 × 28
      │
      ▼
Conv2D: 13 → 16
      │
      ▼
ReLU
      │
      ▼
MaxPool
      │
      ▼
16 × 14 × 14
      │
      ▼
Conv2D: 16 → 32
      │
      ▼
ReLU
      │
      ▼
MaxPool
      │
      ▼
32 × 7 × 7
      │
      ▼
Conv2D: 32 → 64
      │
      ▼
ReLU
      │
      ▼
64 × 7 × 7

The final encoder representation is:

64 × 7 × 7

---

6. Autoencoder Encoder

The first encoder layer receives the 13 filtered channels:

nn.Conv2d(
    in_channels=13,
    out_channels=16,
    kernel_size=3,
    padding=1
)

The number of channels therefore changes:

13 → 16

A ReLU activation follows the convolution.

A "2 × 2" max-pooling operation then reduces the spatial dimensions:

28 × 28 → 14 × 14

The second convolution changes:

16 → 32 channels

The second max-pooling operation reduces:

14 × 14 → 7 × 7

The third convolution changes:

32 → 64 channels

Therefore, the encoder produces:

64 × 7 × 7

for each processed image.

---

7. Hidden Representation

The output of the encoder is treated as the hidden representation or latent representation.

The encoder is accessed using:

z = model.encoder(filtered_x)

The resulting representation is:

64 × 7 × 7

This means that one image is represented using:

64 feature maps

where each feature map has dimensions:

7 × 7

Therefore:

13 × 28 × 28
        ↓
Convolutional Encoder
        ↓
64 × 7 × 7

This is one of the most important transformations in the project.

---

8. Convolutional Autoencoder Decoder

The decoder reconstructs the 13-channel filtered representation.

It starts from:

64 × 7 × 7

and progressively increases the spatial resolution.

The first transposed convolution produces:

32 × 14 × 14

The second transposed convolution produces:

16 × 28 × 28

The final convolution produces:

13 × 28 × 28

The decoder therefore performs approximately:

64 × 7 × 7
      │
      ▼
32 × 14 × 14
      │
      ▼
16 × 28 × 28
      │
      ▼
13 × 28 × 28

The purpose of this reconstruction stage is to train the encoder to preserve meaningful information from the filtered representation.

---

9. Autoencoder Training

The reconstruction loss is Mean Squared Error:

criterion = nn.MSELoss()

The optimizer is Adam:

optimizer = optim.Adam(
    model.parameters(),
    lr=1e-3
)

The current experiment uses:

Learning rate: 0.001
Epochs: 5

The autoencoder attempts to minimize the difference between the filtered input and its reconstruction.

Conceptually:

Filtered Input
13 × 28 × 28
      │
      ▼
    Encoder
      │
      ▼
64 × 7 × 7
      │
      ▼
    Decoder
      │
      ▼
Reconstruction
13 × 28 × 28

The reconstruction objective is:

$$
L = \frac{1}{N}\sum_{i=1}^{N}(x_i-\hat{x}_i)^2
$$

where:

- "x" is the original filtered representation.
- "x̂" is the reconstructed representation.
- "N" is the number of elements.

---

10. Latent Feature Extraction

After the autoencoder has learned its representation, the encoder output is used as the feature representation.

For the current implementation, latent representations are extracted from:

for i in range(32):

Therefore, the current classification experiment processes 32 MNIST samples.

For each sample:

13 × 28 × 28
       ↓
Encoder
       ↓
64 × 7 × 7

The latent representations are stored in:

latents

and the corresponding MNIST labels are stored in:

labels

The resulting latent tensor is:

32 × 64 × 7 × 7

and the labels have the shape:

32

Therefore:

Latents
32 × 64 × 7 × 7

Labels
32

---

11. Feature-Map Visualization

The latent representation contains 64 feature maps for each sample.

For one sample:

64 × 7 × 7

The project visualizes these feature maps to inspect the information learned by the convolutional encoder.

Conceptually:

13 Filtered Channels
        │
        ▼
Convolutional Encoder
        │
        ▼
64 Learned Feature Maps
        │
        ├── Feature 1
        ├── Feature 2
        ├── Feature 3
        ├── ...
        └── Feature 64

Each feature map represents a learned response generated by the encoder.

---

12. Residual Convolutional Structure

The learned latent representation is subsequently used as input to a Residual Convolutional Structure (RCS).

The implementation uses a bottleneck-style residual block:

class blk(nn.Module):

The block contains three convolutional layers:

1 × 1 convolution
       ↓
3 × 3 convolution
       ↓
1 × 1 convolution

These layers are combined with:

- Batch Normalization
- ReLU activation
- Identity/skip connections

The fundamental residual operation is:

Output = F(x) + x

where:

x = input
F(x) = learned transformation

The skip connection allows the original input to be added to the transformed representation.

Conceptually:

                 ┌──────────────────────┐
                 │                      │
                 │      Identity        │
                 │                      │
Input ───────────┼──────────────────────┤
     │           │                      │
     ▼           │                      │
   Conv 1×1      │                      │
     │           │                      │
    BN           │                      │
     │           │                      │
   ReLU          │                      │
     │           │                      │
   Conv 3×3      │                      │
     │           │                      │
    BN           │                      │
     │           │                      │
   ReLU          │                      │
     │           │                      │
   Conv 1×1      │                      │
     │           │                      │
    BN           │                      │
     │           │                      │
     └───────────┴────────── Add ───────┘
                              │
                             ReLU
                              │
                              ▼
                            Output

---

13. ResNet Architecture

The residual structure in this project follows the design of a ResNet-50-style architecture.

The project also defines:

ResNet50()
ResNet101()
ResNet152()

The current classification experiment uses:

net = ResNet50()

The ResNet-50 configuration contains four major residual stages:

Layer 1 → 3 residual blocks
Layer 2 → 4 residual blocks
Layer 3 → 6 residual blocks
Layer 4 → 3 residual blocks

Therefore:

[3, 4, 6, 3]

The architecture progressively transforms the latent representation into higher-level features.

At the end of the network:

Adaptive Average Pooling
        ↓
Flatten
        ↓
Fully Connected Layer
        ↓
10 Classes

---

14. Classification

The final classification layer contains:

10 output classes

because MNIST has ten digit classes.

The final output corresponds to:

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

The classification loss is:

criterion = nn.CrossEntropyLoss()

The predicted class is obtained from the largest output value:

prediction = torch.argmax(
    y,
    dim=1
)

The classification pipeline is therefore:

Latent Representation
        │
        ▼
     ResNet50
        │
        ▼
Feature Extraction
        │
        ▼
Global Average Pooling
        │
        ▼
Fully Connected Layer
        │
        ▼
10 Class Outputs
        │
        ▼
Digit Prediction

---

15. Complete Architecture

The complete project can be represented as:

                         MNIST
                           │
                           ▼
                     28 × 28 Image
                           │
                           ▼
                 ┌─────────────────┐
                 │ Artificial      │
                 │ Interferometry  │
                 └────────┬────────┘
                          │
                          ▼
                 Artificial
                 Interferogram
                    28 × 28
                          │
                          ▼
               ┌───────────────────┐
               │ 13 Fixed Filters  │
               └─────────┬─────────┘
                         │
                         ▼
                  13 × 28 × 28
               Filtered Channels
                         │
                         ▼
             ┌─────────────────────┐
             │ Convolutional       │
             │ Autoencoder         │
             └──────────┬──────────┘
                        │
                        ▼
                  64 × 7 × 7
              Hidden Representation
                        │
                        ▼
             ┌─────────────────────┐
             │ Residual            │
             │ Convolutional       │
             │ Structure           │
             └──────────┬──────────┘
                        │
                        ▼
                    ResNet50
                        │
                        ▼
                 10 Classification
                      Outputs
                        │
                        ▼
                  MNIST Digit
                   Prediction

---

16. Fixed Filters vs Learned Features

An important distinction in the architecture is the difference between the manually designed filters and the neural-network features.

Fixed Filters

The 13 filters:

K1 → K13

are manually defined.

Their values remain fixed during the neural-network training process.

They perform predetermined spatial transformations on the interferogram.

Artificial Interferogram
          │
          ▼
    Fixed Filter Bank
          │
          ▼
13 Filtered Representations

Learned Features

The convolutional autoencoder contains trainable convolutional layers.

These layers learn feature representations from the filtered data.

13 Filtered Channels
          │
          ▼
Trainable Convolutional Encoder
          │
          ▼
64 Learned Feature Maps

The project therefore combines:

Fixed Feature Transformation
          +
Learned Feature Representation
          +
Residual Classification

---

17. Tensor Dimensions

The main tensor transformations in the project are:

Stage| Shape
Original MNIST image| "1 × 28 × 28"
Artificial interferogram| "28 × 28"
Number of fixed filters| "13"
Filtered representation| "13 × 28 × 28"
Encoder output 1| "16 × 28 × 28"
After first pooling| "16 × 14 × 14"
Encoder output 2| "32 × 14 × 14"
After second pooling| "32 × 7 × 7"
Encoder output 3| "64 × 7 × 7"
Current latent dataset| "32 × 64 × 7 × 7"
Number of classification classes| "10"

---

18. CUDA Support

The project automatically selects between CUDA and CPU:

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

Therefore, when a CUDA-compatible GPU is available, PyTorch can perform the neural-network computations on the GPU.

Otherwise, the project runs on the CPU.

---

19. Visualization

The project uses Matplotlib to visualize intermediate representations.

The visualizations include:

Artificial Interferogram

Shows the result of combining the object and reference waves.

Filter Responses

Displays the outputs generated by the 13 spatial filters.

Autoencoder Reconstruction

Compares filtered inputs with reconstructed outputs.

Latent Feature Maps

Displays the 64 feature maps produced by the encoder.

These visualizations help investigate how information changes throughout the pipeline.

---

20. Current Experimental Configuration

The current implementation uses:

Dataset:              MNIST
Image size:           28 × 28
Original channels:    1
Fixed filters:        13
Filter size:          3 × 3
Filtered channels:    13
Latent channels:      64
Latent spatial size:  7 × 7
Latent samples:       32
Autoencoder epochs:   5
Classifier:           ResNet50
Classes:              10

---

21. Installation

Clone the repository:

git clone https://github.com/AMADIVHACKERDON/NetworkData_Model.git

Move into the project directory:

cd NetworkData_Model

Install the required Python packages:

pip install numpy scipy matplotlib torch torchvision

---

22. Running the Project

Run the Python file containing the implementation:

python your_script.py

The program will:

1. Load the MNIST dataset.
2. Generate artificial interferograms.
3. Apply the 13 fixed filters.
4. Create the 13-channel representation.
5. Train the convolutional autoencoder.
6. Generate the hidden representation.
7. Visualize latent feature maps.
8. Pass the latent representation to the residual network.
9. Train the classification network.
10. Produce digit predictions.

---

23. Current Scope

This implementation is currently an experimental research prototype.

The complete MNIST dataset is loaded, but the current latent-feature extraction and classification experiment uses:

for i in range(32):

Therefore, only 32 samples are currently passed through the complete:

Interferometry
→ Filtering
→ Autoencoder
→ Latent Representation
→ RCS

pipeline.

The architecture can subsequently be extended to process the complete training and testing datasets.

---

24. Future Development

Future work can include:

Full Dataset Processing

Apply the complete pipeline to all training and testing images.

60,000 Training Images
          +
10,000 Testing Images

Automated Dataset Construction

Generate and store the filtered interferometric representations for the complete dataset.

Improved Training

Train the autoencoder using a larger number of samples.

Classification Evaluation

Add quantitative evaluation metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Baseline Comparison

Compare the proposed pipeline agains
