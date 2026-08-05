import os
import json
import numpy as np
import matplotlib.pyplot as plt

import torch
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import SegResNetDS
from monai.inferers import sliding_window_inference
from monai.metrics import (
    DiceMetric,
    MeanIoU,
    HausdorffDistanceMetric,
    SurfaceDistanceMetric,
    ConfusionMatrixMetric
)
from monai.transforms import (
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    Spacingd,
)
from monai.utils import set_determinism

set_determinism(seed=42)

def get_scalar_metric(metric_output) -> float:
    """Safely extracts a scalar float from MONAI metric aggregates."""
    if isinstance(metric_output, (list, tuple)):
        metric_output = metric_output[0]
    if isinstance(metric_output, torch.Tensor):
        return metric_output.item()
    return float(metric_output)

# --- Configuration ---
dataset_path = "data/dataset.json"
data_dir = "data" 
checkpoint_path = "best_segresnet_model.pth"
output_dir = "test_visualizations"

voxel_spacing = (0.8, 0.8, 3.0)
network_input_size = (128, 128, 32)

os.makedirs(output_dir, exist_ok=True)

# 1. Load Data
with open(dataset_path, "r") as f:
    dataset = json.load(f)

test_data = [
    {
        "image": os.path.join(data_dir, item["image"]) if data_dir else item["image"],
        "label": os.path.join(data_dir, item["label"]) if data_dir else item["label"],
    }
    for item in dataset["test"]
]

# 2. Transforms (Mirroring Validation Transforms)
test_transforms = Compose([
    LoadImaged(keys=["image", "label"], image_only=False),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=voxel_spacing, mode=("bilinear", "nearest")),
    AsDiscreted(keys="label", threshold=0.5),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    CropForegroundd(keys=["image", "label"], source_key="image", margin=5),
])

test_dataset = CacheDataset(data=test_data, transform=test_transforms, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

# 3. Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autocast_device = "cuda" if torch.cuda.is_available() else "cpu"

model = SegResNetDS(
    spatial_dims=3,
    init_filters=32,
    in_channels=1,
    out_channels=1,
    norm=("GROUP", {"num_groups": 8}),
    act=("MISH", {"inplace": True}),
    dsdepth=4,
).to(device)

print(f"Loading weights from {checkpoint_path}...")
state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

# Strip compiler prefix if present
clean_state_dict = {
    (k.replace("_orig_mod.", "") if k.startswith("_orig_mod.") else k): v
    for k, v in state_dict.items()
}

model.load_state_dict(clean_state_dict)
model.eval()

# 4. Metrics Suite Setup
metric_dice = DiceMetric(include_background=False, reduction="mean")
metric_iou  = MeanIoU(include_background=False, reduction="mean")
metric_hd95 = HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean")
metric_asd  = SurfaceDistanceMetric(include_background=False, reduction="mean")
metric_sens = ConfusionMatrixMetric(include_background=False, metric_name="sensitivity", reduction="mean")
metric_spec = ConfusionMatrixMetric(include_background=False, metric_name="specificity", reduction="mean")

print(f"Starting inference on {len(test_data)} test cases...")

# 5. Inference and Evaluation Loop
with torch.no_grad():
    for i, batch in enumerate(test_loader):
        test_images = batch["image"].to(device, non_blocking=True)
        test_labels = batch["label"].to(device, non_blocking=True)
        
        filename = os.path.basename(batch["image_meta_dict"]["filename_or_obj"][0])
        case_id = filename.split('.')[0]

        with torch.amp.autocast(device_type=autocast_device, enabled=torch.cuda.is_available()):
            outputs_logits = sliding_window_inference(
                inputs=test_images,
                roi_size=network_input_size,
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
                mode="gaussian"
            )

        test_preds = (torch.sigmoid(outputs_logits) > 0.5).float()
        
        # --- Multi-Channel One-Hot Encoding for Distance & Confusion Matrix Metrics ---
        preds_onehot  = torch.cat([1.0 - test_preds, test_preds], dim=1)
        labels_onehot = torch.cat([1.0 - test_labels, test_labels], dim=1)

        metric_dice(y_pred=preds_onehot, y=labels_onehot)
        metric_iou(y_pred=preds_onehot, y=labels_onehot)
        metric_hd95(y_pred=preds_onehot, y=labels_onehot)
        metric_asd(y_pred=preds_onehot, y=labels_onehot)
        metric_sens(y_pred=preds_onehot, y=labels_onehot)
        metric_spec(y_pred=preds_onehot, y=labels_onehot)

        # --- Visualization ---
        img_np = test_images[0, 0].cpu().numpy()
        lbl_np = test_labels[0, 0].cpu().numpy()
        pred_np = test_preds[0, 0].cpu().numpy()

        z_sum = lbl_np.sum(axis=(0, 1))
        best_slice_idx = np.argmax(z_sum) if z_sum.max() > 0 else img_np.shape[2] // 2

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_np[:, :, best_slice_idx].T, cmap="gray")
        axes[0].set_title(f"Image\n(Slice {best_slice_idx})")
        axes[0].axis("off")
        
        axes[1].imshow(lbl_np[:, :, best_slice_idx].T, cmap="gray")
        axes[1].set_title("Ground Truth")
        axes[1].axis("off")
        
        axes[2].imshow(pred_np[:, :, best_slice_idx].T, cmap="gray")
        axes[2].set_title("Prediction")
        axes[2].axis("off")
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"{case_id}_infer.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
        
        print(f"Processed {case_id} -> Saved visual to {save_path}")

# 6. Aggregate Final Statistics
dice_val = get_scalar_metric(metric_dice.aggregate())
iou_val  = get_scalar_metric(metric_iou.aggregate())
sens_val = get_scalar_metric(metric_sens.aggregate())
spec_val = get_scalar_metric(metric_spec.aggregate())
hd95_val = get_scalar_metric(metric_hd95.aggregate())
asd_val  = get_scalar_metric(metric_asd.aggregate())

print("\n" + "="*50)
print("             INTERNAL TEST SET RESULTS             ")
print("="*50)
print(f"Cases Evaluated:           {len(test_data)}")
print(f"Dice Score:                {dice_val:.4f}")
print(f"Intersection over Union:   {iou_val:.4f}")
print(f"Sensitivity (Recall):      {sens_val:.4f}")
print(f"Specificity:               {spec_val:.4f}")
print(f"95% Hausdorff Distance:    {hd95_val:.4f} mm")
print(f"Average Surface Distance:  {asd_val:.4f} mm")
print("="*50)