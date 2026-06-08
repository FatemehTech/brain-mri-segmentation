# ===== train_unet_finetune.py =====
from PIL import Image
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import nibabel as nib
import numpy as np
from unet import UNet
# Import the self-supervised model
from train_self_supervised import UNetAutoencoder

# Data paths
image_dir = "Task01_BrainTumour/imagesTr"
label_dir = "Task01_BrainTumour/labelsTr"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MRISliceDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.files = sorted([f for f in os.listdir(
            image_dir) if f.endswith(".nii.gz")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        img_path = os.path.join(self.image_dir, file)
        lbl_path = os.path.join(self.label_dir, file)
        image = nib.load(img_path).get_fdata()
        label = nib.load(lbl_path).get_fdata()

        if image.ndim == 4:
            image = image[..., 0]

        mid = image.shape[2] // 2
        img = image[:, :, mid]
        lbl = label[:, :, mid]
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)

        return torch.tensor(img).unsqueeze(0).float(), torch.tensor(lbl).unsqueeze(0).float()


# class MRIDataset(Dataset):
#     def __init__(self, image_dir, mask_dir, transform=None):
#         self.image_dir = image_dir
#         self.mask_dir = mask_dir
#         self.images = sorted(os.listdir(image_dir))
#         self.transform = transform

#     def __len__(self):
#         return len(self.images)

#     def __getitem__(self, idx):
#         img_path = os.path.join(self.image_dir, self.images[idx])
#         mask_path = os.path.join(self.mask_dir, self.images[idx])

#         image = np.array(Image.open(img_path).convert("L")) / 255.0
#         mask = np.array(Image.open(mask_path).convert("L")) / 255.0

#         image = torch.tensor(image).unsqueeze(0).float()
#         mask = torch.tensor(mask).unsqueeze(0).float()

#         return image, mask


# ------------------------------
#  Model + Load Pretrained Weights
# ------------------------------
print(" Loading U-Net model with self-supervised weights ...")
pretrained = UNetAutoencoder(n_channels=1)
pretrained.load_state_dict(torch.load(
    "pretrained_autoencoder.pth", map_location=device))

model = UNet(n_channels=1, n_classes=1)

# Copy weights from encoder to the main model
model.dconv_down1.load_state_dict(pretrained.encoder1.state_dict())
model.dconv_down2.load_state_dict(pretrained.encoder2.state_dict())
model.dconv_down3.load_state_dict(pretrained.encoder3.state_dict())
model.dconv_down4.load_state_dict(pretrained.bottleneck.state_dict())

model = model.to(device)


# ------------------------------
#  Segmentation Model Training
# ------------------------------
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.BCEWithLogitsLoss()

dataset = MRISliceDataset(image_dir, label_dir)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

num_epochs = 5
print("🚀 Starting Fine-Tuning Training ...")

model.train()
for epoch in range(num_epochs):
    total_loss = 0
    for img, mask in dataloader:
        img, mask = img.to(device), mask.to(device)
        pred = model(img)
        loss = criterion(pred, mask)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(
        f" Epoch {epoch+1}/{num_epochs} - Loss: {total_loss/len(dataloader):.4f}")

torch.save(model.state_dict(), "trained_model_finetuned.pth")
print("✅ Fine-tuned model saved as trained_model_finetuned.pth")
