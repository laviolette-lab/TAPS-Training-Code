import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from monai.apps import DecathlonDataset
from monai.networks.nets import SegResNetDS
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    CropForegroundd,
    Lambdad,
)

# 1. Setup Device and Pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
autocast_device = "cuda" if device.type == "cuda" else "cpu"

msd_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    Lambdad(keys="image", func=lambda x: x[0:1, ...]),
    Lambdad(keys="label", func=lambda x: (x > 0).float()),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(0.8, 0.8, 3.0), mode=("bilinear", "nearest")),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    CropForegroundd(keys=["image", "label"], source_key="image", margin=5),
])

# 2. Load Dataset
print("Loading MSD Task05_Prostate dataset...")
dataset = DecathlonDataset(
    root_dir="./data", 
    task="Task05_Prostate", 
    transform=msd_transforms, 
    section="training", 
    val_frac=0.0, 
    download=False, 
    cache_rate=0.0
)

# 3. Initialize Model and Load Weights
print("Loading model weights...")
model = SegResNetDS(
    spatial_dims=3, init_filters=32, in_channels=1, out_channels=1,
    norm=("GROUP", {"num_groups": 8}), act=("MISH", {"inplace": True}), dsdepth=4,
).to(device)

weights_path = "best_segresnet_model.pth"
state_dict = torch.load(weights_path, map_location=device, weights_only=True)
clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
model.load_state_dict(clean_state_dict)
model.eval()

# Helper function for bulletproof RGB overlays
def create_rgb_overlay(img_base, mask, color, alpha=0.5):
    """Blends a specific RGB color into an image where the mask is True."""
    # If image is grayscale [H, W], convert to RGB [H, W, 3]
    if len(img_base.shape) == 2:
        rgb = np.stack([img_base, img_base, img_base], axis=-1)
    else:
        rgb = img_base.copy()
        
    for i in range(3):
        rgb[:, :, i] = np.where(mask, rgb[:, :, i] * (1 - alpha) + color[i] * alpha, rgb[:, :, i])
    return rgb

# 4. Select Random Cases
num_samples = 3
random_indices = random.sample(range(len(dataset)), num_samples)

fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))
fig.suptitle("TAPS Segmentation Comparison & Mismatch", fontsize=16)

# 5. Inference and Plotting Loop
with torch.no_grad():
    for i, idx in enumerate(random_indices):
        data = dataset[idx]
        
        # Add batch dimension and move to device: [1, 1, H, W, D]
        inputs = data["image"].unsqueeze(0).to(device)
        
        # Run Inference
        with torch.amp.autocast(device_type=autocast_device, enabled=(device.type == "cuda")):
            outputs_logits = sliding_window_inference(
                inputs=inputs, roi_size=(128, 128, 32), sw_batch_size=4,
                predictor=model, overlap=0.5, mode="gaussian"
            )
        
        # Threshold prediction to binary
        pred_tensor = (torch.sigmoid(outputs_logits) > 0.5).float()
        
        # Move everything to CPU numpy arrays for plotting
        img_vol = data["image"][0].cpu().numpy()
        lbl_vol = data["label"][0].cpu().numpy()
        pred_vol = pred_tensor[0, 0].cpu().numpy()
        
        # Find the slice with the most prostate tissue in the ground truth
        z_sum = lbl_vol.sum(axis=(0, 1))
        best_z = np.argmax(z_sum)
        
        # Extract 2D slices
        img_slice = img_vol[:, :, best_z]
        lbl_slice = lbl_vol[:, :, best_z]
        pred_slice = pred_vol[:, :, best_z]
        
        # Fix Orientation (Anterior up)
        img_slice = np.rot90(img_slice)
        lbl_slice = np.rot90(lbl_slice)
        pred_slice = np.rot90(pred_slice)
        
        # Fix Contrast & Normalize strictly to [0, 1] for RGB conversion
        p1, p99 = np.percentile(img_slice, (1, 99))
        img_slice_norm = np.clip(img_slice, p1, p99)
        img_slice_norm = (img_slice_norm - p1) / (p99 - p1 + 1e-8)
        
        # Boolean masks
        lbl_mask = lbl_slice > 0.5
        pred_mask = pred_slice > 0.5
        
        # Mismatch masks
        fp_mask = pred_mask & ~lbl_mask  # False Positives (predicted beyond mask)
        fn_mask = lbl_mask & ~pred_mask  # False Negatives (mask beyond prediction)
        
        # Build Overlays
        # Column 1: Ground Truth (Green overlay)
        gt_rgb = create_rgb_overlay(img_slice_norm, lbl_mask, color=(0, 1, 0), alpha=0.4)
        
        # Column 2: Prediction (Red overlay)
        pred_rgb = create_rgb_overlay(img_slice_norm, pred_mask, color=(1, 0, 0), alpha=0.4)
        
        # Column 3: Mismatch (FPs = Red, FNs = Green)
        mismatch_rgb = create_rgb_overlay(img_slice_norm, fp_mask, color=(1, 0, 0), alpha=0.6)
        mismatch_rgb = create_rgb_overlay(mismatch_rgb, fn_mask, color=(0, 1, 0), alpha=0.6)
        
        # -- Plotting --
        axes[i, 0].imshow(gt_rgb)
        axes[i, 0].set_title(f"Case {idx} - Ground Truth (Z={best_z})")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(pred_rgb)
        axes[i, 1].set_title(f"Case {idx} - TAPS Prediction")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(mismatch_rgb)
        axes[i, 2].set_title(f"Mismatch (Red=Overpredict, Green=Underpredict)")
        axes[i, 2].axis("off")

plt.tight_layout()
plt.savefig("msd_mismatch_grid.png", dpi=150, bbox_inches="tight", facecolor="white")
print("Saved comparison to 'msd_mismatch_grid.png'")