# ===== train_self_supervised.py =====
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
#  UNet Autoencoder Architecture
# -----------------------------------


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNetAutoencoder(nn.Module):
    def __init__(self, n_channels=1):
        super().__init__()
        self.encoder1 = DoubleConv(n_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)

        self.up3 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder3 = DoubleConv(512 + 256, 256)
        self.up2 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder2 = DoubleConv(256 + 128, 128)
        self.up1 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder1 = DoubleConv(128 + 64, 64)
        self.final = nn.Conv2d(64, n_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder1(x)
        x2 = self.encoder2(self.pool1(x1))
        x3 = self.encoder3(self.pool2(x2))
        x_bottleneck = self.bottleneck(self.pool3(x3))
        x = self.up3(x_bottleneck)
        x = torch.cat([x, x3], dim=1)
        x = self.decoder3(x)
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.decoder2(x)
        x = self.up1(x)
        x = torch.cat([x, x1], dim=1)
        x = self.decoder1(x)
        return self.final(x)


# -----------------------------------
#  Simple Dataset (You can replace it with MRI data later)
# -----------------------------------
class DummyMRIDataset(Dataset):
    def __init__(self, num_samples=200, img_size=128):
        self.data = torch.rand(num_samples, 1, img_size, img_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# -----------------------------------
#  Image Masking Function for Self-Supervised Learning
# -----------------------------------
def mask_image(img, mask_ratio=0.3):
    mask = (torch.rand_like(img) > mask_ratio).float()
    return img * mask


# -----------------------------------
#  Self-Supervised Model Training
# -----------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetAutoencoder(n_channels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

dataset = DummyMRIDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

num_epochs = 10
print(" Starting Self-Supervised Training ...")
for epoch in range(num_epochs):
    total_loss = 0
    for img in dataloader:
        img = img.to(device)
        corrupted = mask_image(img)
        output = model(corrupted)
        loss = criterion(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(
        f"Epoch [{epoch+1}/{num_epochs}] Loss: {total_loss/len(dataloader):.4f}")

# Save model
torch.save(model.state_dict(), "pretrained_autoencoder.pth")
print("✅ Self-supervised model saved as pretrained_autoencoder.pth")
