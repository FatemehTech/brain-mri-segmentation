import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt

# DoubleConv architecture for U-Net


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

# U-Net with skip connections


class UNetAutoencoder(nn.Module):
    def __init__(self, n_channels):
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
        # Sum of bottleneck + encoder3 channels
        self.decoder3 = DoubleConv(512 + 256, 256)

        self.up2 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder2 = DoubleConv(256 + 128, 128)

        self.up1 = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder1 = DoubleConv(128 + 64, 64)

        self.final = nn.Conv2d(64, n_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.encoder1(x)       # [B,64,H,W]
        x2 = self.encoder2(self.pool1(x1))  # [B,128,H/2,W/2]
        x3 = self.encoder3(self.pool2(x2))  # [B,256,H/4,W/4]

        x_bottleneck = self.bottleneck(self.pool3(x3))  # [B,512,H/8,W/8]

        x = self.up3(x_bottleneck)  # [B,512,H/4,W/4]
        # Concatenate along channel dimension -> [B,768,H/4,W/4]
        x = torch.cat([x, x3], dim=1)
        x = self.decoder3(x)  # -> [B,256,H/4,W/4]

        x = self.up2(x)  # [B,256,H/2,W/2]
        x = torch.cat([x, x2], dim=1)  # [B,384,H/2,W/2]
        x = self.decoder2(x)  # [B,128,H/2,W/2]

        x = self.up1(x)  # [B,128,H,W]
        x = torch.cat([x, x1], dim=1)  # [B,192,H,W]
        x = self.decoder1(x)  # [B,64,H,W]

        return self.final(x)  # [B,1,H,W]

# Corrupt image (masking)


def mask_image(img, mask_ratio=0.3):
    mask = (torch.rand_like(img) > mask_ratio).float()
    if img.is_cuda:
        mask = mask.to(img.device)
    return img * mask

# Synthetic dataset for testing


class DummyMRIDataset(Dataset):
    def __init__(self, num_samples=100, img_size=128):
        self.data = torch.rand(num_samples, 1, img_size, img_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# Settings
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNetAutoencoder(n_channels=1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

dataset = DummyMRIDataset()
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Train the model for several epochs
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0
    for img in dataloader:
        img = img.to(device)
        corrupted = mask_image(img)
        output = model(corrupted)

        loss = loss_fn(output, img)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(
        f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(dataloader):.4f}")

# Display a sample input and output
sample_img = dataset[0].unsqueeze(0).to(device)
corrupted = mask_image(sample_img)
reconstructed = model(corrupted)

original_np = sample_img.cpu().squeeze().detach().numpy()
corrupted_np = corrupted.cpu().squeeze().detach().numpy()
reconstructed_np = reconstructed.cpu().squeeze().detach().numpy()

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(original_np, cmap='gray')
axs[0].set_title("Original")
axs[1].imshow(corrupted_np, cmap='gray')
axs[1].set_title("Corrupted")
axs[2].imshow(reconstructed_np, cmap='gray')
axs[2].set_title("Reconstructed")
for ax in axs:
    ax.axis('off')
plt.tight_layout()
plt.show()
