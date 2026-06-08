# start_viewer()
import os
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Image folder paths
image_dir = "Task01_BrainTumour/imagesTr"
label_dir = "Task01_BrainTumour/labelsTr"

# List of real MRI files (excluding hidden files or ._ files)
image_files = sorted([
    f for f in os.listdir(image_dir)
    if f.endswith(".nii.gz") and not f.startswith("._")
])

# Current index
index = 0

# Display function


def show_image(index):
    image_path = os.path.join(image_dir, image_files[index])
    label_path = os.path.join(label_dir, image_files[index])

    image = nib.load(image_path).get_fdata()
    label = nib.load(label_path).get_fdata()

    mid_slice = image.shape[2] // 2  # Middle slice

    plt.clf()
    plt.suptitle(f"{image_files[index]} - Slice {mid_slice}")

    plt.subplot(1, 2, 1)
    mri_slice = image[:, :, mid_slice]
    mri_slice = (mri_slice - np.min(mri_slice)) / \
        (np.max(mri_slice) - np.min(mri_slice) + 1e-8)
    plt.imshow(mri_slice, cmap="gray")
    plt.title("MRI Image")

    plt.subplot(1, 2, 2)
    plt.imshow(label[:, :, mid_slice], cmap="hot", alpha=0.6)
    plt.title("Tumor Mask")

    plt.draw()


# Keyboard event handler


def on_key(event):
    global index
    if event.key == 'right':
        index = (index + 1) % len(image_files)
        show_image(index)
    elif event.key == 'left':
        index = (index - 1) % len(image_files)
        show_image(index)
    elif event.key == 'escape':
        plt.close()


# Initial execution
fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', on_key)
show_image(index)
plt.show()
