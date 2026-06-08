import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import nibabel as nib
from unet import UNet

# ---------- Configuration ----------
image_dir = "Task01_BrainTumour/imagesTr"
label_dir = "Task01_BrainTumour/labelsTr"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------- File List ----------
file_list = sorted([
    f for f in os.listdir(image_dir)
    if f.endswith(".nii.gz") and not f.startswith("._")
])

# ---------- Model ----------
model = UNet(n_channels=1, n_classes=1).to(device)
# # Load fine-tuned trained model
# model.load_state_dict(torch.load(
#     "trained_model_finetuned.pth", map_location=device))
# model.eval()  # Evaluation mode (not training)


model.load_state_dict(torch.load("trained_model.pth", map_location=device))
model.eval()

# ---------- State Variables ----------
current_file_idx = 0
current_slice = 0
image = None
label = None
num_slices = 0
file_name = ""


# ---------- Load an MRI File ----------
def load_file(file_idx):
    global image, label, num_slices, file_name, current_slice
    file_name = file_list[file_idx]
    image_path = os.path.join(image_dir, file_name)
    label_path = os.path.join(label_dir, file_name)

    image = nib.load(image_path).get_fdata()
    label = nib.load(label_path).get_fdata()

    if image.ndim == 4:
        image = image[..., 0]

    num_slices = image.shape[2]
    current_slice = num_slices // 2  # Start from the middle


# ---------- Predict a Slice ----------
def predict_slice(slice_index):
    img_slice = image[:, :, slice_index]
    lbl_slice = label[:, :, slice_index]

    img_slice = (img_slice - np.min(img_slice)) / \
        (np.max(img_slice) - np.min(img_slice) + 1e-8)
    img_tensor = torch.tensor(img_slice, dtype=torch.float32).unsqueeze(
        0).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred = torch.sigmoid(output).squeeze().cpu().numpy()

    return img_slice, lbl_slice, pred


# ---------- Update Display ----------
fig, axs = plt.subplots(1, 3, figsize=(15, 5))


def update_display():
    img, lbl, pred = predict_slice(current_slice)

    axs[0].cla()
    axs[1].cla()
    axs[2].cla()

    axs[0].imshow(img, cmap='gray')
    axs[0].set_title(f"Slice {current_slice}")
    axs[0].axis('off')

    axs[1].imshow(lbl, cmap='hot', alpha=0.6)
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')

    axs[2].imshow(pred, cmap='hot', alpha=0.7)
    axs[2].set_title("Prediction")
    axs[2].axis('off')

    fig.suptitle(f"File: {file_name}", fontsize=14)
    fig.canvas.draw()


# ---------- Key Event Handling ----------
def on_key(event):
    global current_slice, current_file_idx

    if event.key == 'right':
        current_slice = (current_slice + 1) % num_slices
        update_display()
    elif event.key == 'left':
        current_slice = (current_slice - 1) % num_slices
        update_display()
    elif event.key == 'up':
        current_file_idx = (current_file_idx + 1) % len(file_list)
        load_file(current_file_idx)
        update_display()
    elif event.key == 'down':
        current_file_idx = (current_file_idx - 1) % len(file_list)
        load_file(current_file_idx)
        update_display()


# ---------- Initial Execution ----------
load_file(current_file_idx)
update_display()
fig.canvas.mpl_connect('key_press_event', on_key)
plt.tight_layout()
plt.show()
