
import numpy as np
import math
from scipy.signal import convolve as Conv
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 128
transform = transforms.Compose([transforms.ToTensor()])

train_dataset = datasets.MNIST(root="./data", train = True, transform = transform, download = True)
test_dataset = datasets.MNIST(root="./data", train = False, transform = transform, download = True)
train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size= batch_size, shuffle=False)

K1 = np.array([[0,0,0], [0,1,0], [0,0,0]])
K2 = np.array([[1,0,-1], [0,0,0], [-1,0,1]])
K3 = np.array([[0,1,0], [1,-4,1], [0,1,0]])
K4 = np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])

K5 = np.array([[0,-1,0], [-1,5,-1], [0,-1,0]])
K6 = np.array([[1,1,1], [1,1,1], [1,1,1]])
K6 = K6/9
K7 = np.array([[1,2,1], [2,4,2], [1,2,1]])

K8 = np.array([[-2,-1,0], [-1,1,1], [0,1,2]])

K9 = np.array([[1,0,1], [0,1,0], [1,0,1]])
K10 = np.array([[1,0,0], [0,0,0], [1,1,1]])
K11 = np.array([[1,0,0], [0,1,0], [0,0,1]])
K12 = np.array([[0,0,1], [0,1,0], [1,0,0]])
K13 = np.array([[1,0,1], [1,0,1], [1,0,1]])
K = np.dstack((K1,K2,K3,K4,K5,K6,K7,K8,K9,K10,K11,K12,K13))


NK = K.shape[2]
NC = NK
SP = math.ceil(NK**0.5)

filter_i = []

for i in range(NC):
    Image, label = train_dataset[i]
    Image = Image.squeeze(0).numpy()
    H,W =Image.shape
    y,filtered_x = np.meshgrid(
        np.arange(H),
        np.arange(W),
        indexing="ij"
    )
    amplitude = Image/ Image.max()
    phase_obj = 2 * np.pi * amplitude
    object_wave = amplitude * np.exp(1j * phase_obj)

    angle = np.pi /6
    phase_ref = (2*np.pi * (filtered_x * np.cos(angle) + y * np.sin(angle))/8)
    ref_wave = np.exp(1j * phase_ref)

    interfero = np.abs(object_wave + ref_wave) ** 2
    interfer = (interfero - interfero.min())/(interfero.max() - interfero.min())
    plt.figure(figsize=(15,12))
    plt.suptitle(f"after convolution of channel {i+1}")
    plt.subplots_adjust(hspace=0.5)
    filter_i = []
    for k in range(NK):
        plt.subplot(SP, SP, k+1)
        Out =Conv(interfer,K[:,:,k], mode="same")
        filter_i.append(Out)
        plt.imshow(filter_i[k], cmap='gray')
        plt.title(f"filter {k+1}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

print("Label:", label)
print("Image shape:", Image.shape)

plt.subplot(1,2,1)
plt.imshow(interfer, cmap='gray')
plt.title("Artificial Interferogram")
plt.axis("off")

plt.tight_layout()
plt.show()


class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=13, out_channels=16,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1
            ),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
             nn.ConvTranspose2d( in_channels=64, out_channels=32, kernel_size=2, stride=2
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=2, stride=2
            ),
            nn.ReLU(),

            # 16 × 28 × 28
            nn.Conv2d(in_channels=16, out_channels=13, kernel_size=3, padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon

filter_im = np.stack(filter_i, axis=0)
print("Number of filters:", NK)
print("Number of channels:", filter_im.shape[0])
print("Filtered image shape:", filter_im.shape)
filtered_image = torch.tensor(filter_im, dtype=torch.float32)
filtered_dataset = TensorDataset(filtered_image)
filtered_loader = DataLoader(filtered_dataset, batch_size=13, shuffle=False)

model = AutoEncoder().to(device)

filtered_image = filtered_image.unsqueeze(0)
(filtered_x,) = next(iter(filtered_loader))
print("filtered_x shape:", filtered_x.shape)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-3)
epochs = 5
filtered_image = filtered_image.to(device)

model.train()
for epoch in range(epochs):
        total_loss = 0
        for (filtered_x,)in filtered_loader:
            filtered_x = filtered_x.to(device)

            optimizer.zero_grad()
            x_recon = model(filtered_image)
            loss = criterion(x_recon, filtered_image)
            loss.backward()
            optimizer.step()

            total_loss = total_loss + loss.item()
        avg_loss = total_loss / len(filtered_loader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:6f}")
model.eval()
with torch.no_grad():
    (filtered_x,) = next(iter(filtered_loader))
    print("filtered_x shape:", filtered_x.shape)
    filtered_x = filtered_x.to(device)

    x_recon = model(filtered_x)
    n=8
    plt.figure(figsize= (12,3))

    for i in range(n): 
        plt.subplot(2,n,i+1)
        plt.imshow(filtered_x[i].view(28,28), cmap="gray")
        plt.axis("off")

        plt.subplot(2,n,i+1+n)
        plt.imshow(x_recon[i].view(28,28), cmap="gray")
        plt.axis("off")

    plt.show()


model.eval()

latents = []
labels = []

with torch.no_grad():
    for (filtered_x,) in filtered_loader:
        filtered_x = filtered_x.to(device)
        z = model.encoder(filtered_x)

        latents.append(z.cpu())
        
latents = torch.cat(latents, dim =0)
print("Latents shape:", latents.shape)
sample_features = latents
print(sample_features.shape)
print("First feature map shape:", sample_features[0].shape)

print("i have reacg plot")
plt.figure(figsize=(12,12))

for i in range(64):
    plt.subplot(8,8,i+1)
    plt.imshow(sample_features[i].detach().numpy(),cmap="gray")

    plt.title(f"f{i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show