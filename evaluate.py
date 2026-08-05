import os
import torch
import numpy as np

from monai.apps import DecathlonDataset
from monai.data import DataLoader
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
from monai.utils import set_determinism

def get_optimal_sw_batch_size(device: torch.device) -> int:
    """Dynamically sets sliding window batch size based on available VRAM."""
    if device.type == "cuda":
        total_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        if total_mem_gb >= 20: return 24
        elif total_mem_gb >= 12: return 16
        elif total_mem_gb >= 8: return 8
        elif total_mem_gb >= 4: return 4
        else: return 2
    return 4 

def get_scalar_metric(metric_output) -> float:
    """Safely extracts a scalar float from MONAI metric aggregates (handles Tensor, list, tuple)."""
    if isinstance(metric_output, (list, tuple)):
        metric_output = metric_output[0]
    if isinstance(metric_output, torch.Tensor):
        return metric_output.item()
    return float(metric_output)

def metric_stats(metric):
    values = metric.aggregate()

    if isinstance(values, (list, tuple)):
        values = values[0]

    values = values.cpu().numpy().flatten()

    # remove NaNs if any (can occur for HD95/ASD on empty masks)
    values = values[~np.isnan(values)]

    metric.reset()

    return np.mean(values), np.std(values), values

def main():
    set_determinism(seed=42)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    autocast_device = "cuda" if device.type == "cuda" else "cpu"
    sw_batch_size = get_optimal_sw_batch_size(device)
    
    weights_path = "best_segresnet_model.pth"

    print(f"Hardware: {device.type.upper()} | Window Batch Size: {sw_batch_size}")
    
    # 1. Define MSD-specific Pipeline
    msd_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),
        
        # --- MSD SPECIFIC TWEAKS ---
        # 1. MSD images have T2 (idx 0) and ADC (idx 1). We only want T2.
        Lambdad(keys="image", func=lambda x: x[0:1, ...]),
        # 2. MSD labels have Central Gland (1) and Peripheral Zone (2). Merge them to 1.
        Lambdad(keys="label", func=lambda x: (x > 0).float()),
        # ---------------------------

        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(0.8, 0.8, 3.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=5),
    ])

    # 2. Download and Load MSD Task05 (Using 'training' section for full ~32 labeled cases)
    print("Fetching/Loading MSD Task05_Prostate dataset...")
    msd_dataset = DecathlonDataset(
        root_dir="./data", 
        task="Task05_Prostate", 
        transform=msd_transforms, 
        section="training",  # Using training split as full external validation set
        download=True,
        cache_rate=1.0,
        val_frac=0.0,  # No validation split needed
    )
    msd_loader = DataLoader(msd_dataset, batch_size=1, shuffle=False)
    print(f"Successfully loaded {len(msd_dataset)} external cases for evaluation.")

    # 3. Model Initialization
    model = SegResNetDS(
        spatial_dims=3, init_filters=32, in_channels=1, out_channels=1,
        norm=("GROUP", {"num_groups": 8}), act=("MISH", {"inplace": True}), dsdepth=4,
    ).to(device)

    # Load weights
    state_dict = torch.load(weights_path, map_location=device, weights_only=True)
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(clean_state_dict)
    model.eval()

    # 4. Define Metrics Suite
    metric_dice = DiceMetric(include_background=False, reduction="none")
    metric_iou = MeanIoU(include_background=False, reduction="none")
    metric_hd95 = HausdorffDistanceMetric(
        include_background=False,
        percentile=95.0,
        reduction="none",
    )
    metric_asd = SurfaceDistanceMetric(include_background=False, reduction="none")
    metric_sens = ConfusionMatrixMetric(
        include_background=False,
        metric_name="sensitivity",
        reduction="none",
    )
    metric_spec = ConfusionMatrixMetric(
        include_background=False,
        metric_name="specificity",
        reduction="none",
    )

    # 5. Inference & Evaluation Loop
    print("Running evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(msd_loader):
            inputs = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=autocast_device, enabled=(device.type == "cuda")):
                outputs_logits = sliding_window_inference(
                    inputs=inputs, roi_size=(128, 128, 32), sw_batch_size=sw_batch_size,
                    predictor=model, overlap=0.5, mode="gaussian"
                )

            preds = (torch.sigmoid(outputs_logits) > 0.5).float()

            # One-hot encoding for multi-channel metric calculation
            preds_onehot = torch.cat([1.0 - preds, preds], dim=1)
            labels_onehot = torch.cat([1.0 - labels, labels], dim=1)

            # Update metrics
            metric_dice(y_pred=preds_onehot, y=labels_onehot)
            metric_iou(y_pred=preds_onehot, y=labels_onehot)
            metric_hd95(
                y_pred=preds_onehot,
                y=labels_onehot,
                spacing=(0.8, 0.8, 3.0),
            )

            metric_asd(
                y_pred=preds_onehot,
                y=labels_onehot,
                spacing=(0.8, 0.8, 3.0),
            )
            metric_sens(y_pred=preds_onehot, y=labels_onehot)
            metric_spec(y_pred=preds_onehot, y=labels_onehot)
            
            print(f"Processed case {i+1}/{len(msd_loader)}")

    # 6. Aggregate Results safely
    dice_mean, dice_std, dice_values = metric_stats(metric_dice)
    iou_mean, iou_std, _ = metric_stats(metric_iou)
    sens_mean, sens_std, _ = metric_stats(metric_sens)
    spec_mean, spec_std, _ = metric_stats(metric_spec)
    hd95_mean, hd95_std, _ = metric_stats(metric_hd95)
    asd_mean, asd_std, _ = metric_stats(metric_asd)

    print("\n" + "="*60)
    print("     EXTERNAL VALIDATION RESULTS (MSD Task05)")
    print("="*60)
    print(f"Cases Evaluated:           {len(msd_loader)}")
    print(f"Dice Score:                {dice_mean:.4f} ± {dice_std:.4f}")
    print(f"Intersection over Union:   {iou_mean:.4f} ± {iou_std:.4f}")
    print(f"Sensitivity (Recall):      {sens_mean:.4f} ± {sens_std:.4f}")
    print(f"Specificity:               {spec_mean:.4f} ± {spec_std:.4f}")
    print(f"95% Hausdorff Distance:    {hd95_mean:.4f} ± {hd95_std:.4f} mm")
    print(f"Average Surface Distance:  {asd_mean:.4f} ± {asd_std:.4f} mm")
    print("="*60)

if __name__ == "__main__":
    main()