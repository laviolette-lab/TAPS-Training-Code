import os
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

# Helper function for RGB overlays
def create_rgb_overlay(img_base, mask, color, alpha=0.5):
    rgb = np.stack([img_base, img_base, img_base], axis=-1)
    for i in range(3):
        rgb[:, :, i] = np.where(mask > 0.5, rgb[:, :, i] * (1 - alpha) + color[i] * alpha, rgb[:, :, i])
    return rgb

# 4. Target 3 random cases
random_indices = random.sample(range(len(dataset)), 3)
slice_targets = ["Base", "Mid-Gland", "Apex"]

fig, axes = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle("Qualitative Performance on External Validation Cohort (MSD Task05)", fontsize=16, y=0.95)

# 5. Inference and Plotting Loop
with torch.no_grad():
    for i, idx in enumerate(random_indices):
        data = dataset[idx]
        
        inputs = data["image"].unsqueeze(0).to(device)
        
        with torch.amp.autocast(device_type=autocast_device, enabled=(device.type == "cuda")):
            outputs_logits = sliding_window_inference(
                inputs=inputs, roi_size=(128, 128, 32), sw_batch_size=4,
                predictor=model, overlap=0.5, mode="gaussian"
            )
        
        pred_tensor = (torch.sigmoid(outputs_logits) > 0.5).float()
        
        img_vol = data["image"][0].cpu().numpy()
        lbl_vol = data["label"][0].cpu().numpy()
        pred_vol = pred_tensor[0, 0].cpu().numpy()
        
        # Find the Z-indices where the prostate actually exists
        z_valid = np.where(lbl_vol.sum(axis=(0, 1)) > 0)[0]
        
        # Target specific anatomical regions
        target_zone = slice_targets[i]
        if target_zone == "Base":
            # Highest Z indices (near bladder/seminal vesicles)
            best_z = z_valid[-2] if len(z_valid) > 2 else z_valid[-1]
        elif target_zone == "Mid-Gland":
            # Center of the gland
            best_z = z_valid[len(z_valid) // 2]
        else: # Apex
            # Lowest Z indices (near pelvic floor)
            best_z = z_valid[1] if len(z_valid) > 2 else z_valid[0]
            
        # Extract 2D slices
        img_slice = img_vol[:, :, best_z]
        lbl_slice = lbl_vol[:, :, best_z]
        pred_slice = pred_vol[:, :, best_z]
        
        # Fix Orientation (Anterior up)
        img_slice = np.rot90(img_slice)
        lbl_slice = np.rot90(lbl_slice)
        pred_slice = np.rot90(pred_slice)
        
        # Fix Contrast & Normalize
        p1, p99 = np.percentile(img_slice, (1, 99))
        img_slice_norm = np.clip(img_slice, p1, p99)
        img_slice_norm = (img_slice_norm - p1) / (p99 - p1 + 1e-8)
        
        # Overlays
        img_rgb = np.stack([img_slice_norm]*3, axis=-1) # Base grayscale image
        gt_rgb = create_rgb_overlay(img_slice_norm, lbl_slice, color=(0, 1, 0), alpha=0.35) # Green GT
        pred_rgb = create_rgb_overlay(img_slice_norm, pred_slice, color=(1, 0, 0), alpha=0.35) # Red Pred
        
        # Row labels
        row_label = f"Case {idx} ({target_zone})"
        
        # Col 1: Raw
        axes[i, 0].imshow(img_rgb)
        if i == 0: axes[i, 0].set_title("Raw T2 Axial", fontweight="bold")
        axes[i, 0].set_ylabel(row_label, fontsize=12, fontweight="bold")
        axes[i, 0].set_xticks([])
        axes[i, 0].set_yticks([])
        
        # Col 2: Ground Truth
        axes[i, 1].imshow(gt_rgb)
        if i == 0: axes[i, 1].set_title("Ground Truth", fontweight="bold")
        axes[i, 1].axis("off")
        
        # Col 3: Prediction
        axes[i, 2].imshow(pred_rgb)
        if i == 0: axes[i, 2].set_title("TAPS Prediction", fontweight="bold")
        axes[i, 2].axis("off")

plt.tight_layout()
plt.subplots_adjust(top=0.90, wspace=0.05, hspace=0.05)
output_filename = "figure_2_taps_qualitative.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Figure successfully saved as '{output_filename}'")