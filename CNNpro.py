

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




latents = []
labels = []

model.eval()

latents = []
labels = []

model.eval()

with torch.no_grad():

    for i in range(32):

        Image, label = train_dataset[i]

        # YOUR EXISTING INTERFEROGRAM CODE
        Image = Image.squeeze(0).numpy()

        H, W = Image.shape

        y, filtered_x = np.meshgrid(
            np.arange(H),
            np.arange(W),
            indexing="ij"
        )

        amplitude = Image / Image.max()

        phase_obj = 2 * np.pi * amplitude

        object_wave = amplitude * np.exp(1j * phase_obj)

        angle = np.pi / 6

        phase_ref = (
            2 * np.pi *
            (
                filtered_x * np.cos(angle)
                + y * np.sin(angle)
            ) / 8
        )

        ref_wave = np.exp(1j * phase_ref)

        interfero = np.abs(object_wave + ref_wave) ** 2

        interfer = (
            interfero - interfero.min()
        ) / (
            interfero.max() - interfero.min()
        )

        # YOUR EXISTING 13 FILTERS
        filter_i = []

        for k in range(NK):

            Out = Conv(
                interfer,
                K[:, :, k],
                mode="same"
            )

            filter_i.append(Out)

        # Convert the 13 filtered images
        filter_im = np.stack(filter_i, axis=0)

        filtered_x = torch.tensor(
            filter_im,
            dtype=torch.float32
        )

        filtered_x = filtered_x.unsqueeze(0)

        filtered_x = filtered_x.to(device)

        # SEND THIS IMAGE THROUGH THE AUTOENCODER
        z = model.encoder(filtered_x)

        # Remove batch dimension
        z = z.squeeze(0)

        # Save the latent representation
        latents.append(z.cpu())

        # Save its label
        labels.append(label)


latents = torch.stack(latents)
labels = torch.tensor(labels)

print("Latents shape:", latents.shape)
print("Labels shape:", labels.shape)

print("Latents shape:", latents.shape)
sample_features = latents
print(sample_features.shape)
print("First feature map shape:", sample_features[0].shape)


plt.figure(figsize=(12,12))

for i in range(64):
    plt.subplot(8,8,i+1)
    plt.imshow(sample_features[0][i].detach().numpy(),cmap="gray")

    plt.title(f"f{i+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()

class blk(nn.Module):
    def __init__(self, in_channels, out_channels, identity_downsample= None, stride=1):
        super(blk, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1,padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels*self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self,x):
        identity = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)

        return x

class ResNet (nn.Module):
    def __init__(self, block, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self.make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self.make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self.make_layer(block, layers[3], out_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*4, num_classes)
    def forward(self,x):
            
            x = self.conv1(x)
            
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
    
            x = self.avgpool(x)
            x = x.reshape(x.shape[0], -1)

            x = self.fc(x)
            return x
        

    def make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels !=4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1,stride=stride),nn.BatchNorm2d(out_channels*4))
            layers.append(block(self.in_channels, out_channels,identity_downsample,stride))
            self.in_channels = out_channels*4

            for i in range(num_residual_blocks - 1):
                layers.append(block(self.in_channels, out_channels))

            return nn.Sequential(*layers)

def ResNet50(img_channels=64, num_classes=10):
    return ResNet(blk, [3, 4, 6, 3], img_channels, num_classes)

def ResNet101(img_channels=64, num_classes=10):
    return ResNet(blk, [3, 4, 23, 3], img_channels, num_classes)

def ResNet152(img_channels=64, num_classes=10):
    return ResNet(blk, [3, 8, 36, 3], img_channels, num_classes)

del model
torch.cuda.empty_cache()
def test():

    net = ResNet50()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    
    
    x = sample_features
    x = x.to(device)
    label = labels
    label = label.to(device)


    net.train()
    for epoch in range(20):
        optimizer.zero_grad()
        y = net(x)

        print(y.shape)
        print(label.shape)
        loss = criterion(y, label)

        loss.backward()

        optimizer.step()

        prediction = torch.argmax(y, dim=1)

        print(
            f"Epoch [{epoch+1}/20], "
            f"Loss: {loss.item():.4f}, "
            f"Prediction: {prediction}, "
            f"Label: {label}"
        )
        print(prediction,
            loss.item(),
            label
            )

test()
