import os
import gc
import json
from time import time

import torch
import torch.amp as amp
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

# MONAI Imports
from monai.data import CacheDataset, DataLoader
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet, SegResNetDS
from monai.transforms import (
    AsDiscreted,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandBiasFieldd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandZoomd,
    Spacingd,
    SpatialPadd,
    RandCoarseDropoutd,
)
from monai.utils import set_determinism
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

set_determinism(seed=42)

# Configuration
dataset_path = "data/dataset.json"
data_dir = "data" 

voxel_spacing = (0.8, 0.8, 3.0)
network_input_size = (128, 128, 32) # Tighter crop for the small prostate
batch_size = 4

spatial_dims = 3
in_model_channels = 1
out_model_channels = 1

initial_learning_rate = 2e-4
max_epochs = 500
early_stopping_patience = 75
checkpoint_path = "best_segresnet_model.pth"

# 1. Load Data
with open(dataset_path, "r") as f:
    dataset = json.load(f)

def build_data_list(split):
    return [
        {
            "image": os.path.join(data_dir, item["image"]) if data_dir else item["image"],
            "label": os.path.join(data_dir, item["label"]) if data_dir else item["label"],
        }
        for item in dataset[split]
    ]

train_data = build_data_list("train")
val_data = build_data_list("val")

# 2. Transforms Pipelines
train_transforms = Compose([
    LoadImaged(keys=["image", "label"], image_only=False),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    
    # Standardize space and orientation
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=voxel_spacing, mode=("bilinear", "nearest")),
    AsDiscreted(keys="label", threshold=0.5), # Ensure binary labels after spacing
    
    # NORMALIZATION MUST HAPPEN BEFORE INTENSITY AUGMENTATIONS
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    
    # Strip empty background air to save compute
    CropForegroundd(keys=["image", "label"], source_key="image", margin=5),

    SpatialPadd(keys=["image", "label"], spatial_size=network_input_size),
    
    # Focus heavily on the foreground (prostate)
    RandCropByPosNegLabeld(
        keys=["image", "label"],
        label_key="label",
        spatial_size=network_input_size,
        pos=2, 
        neg=1,
        num_samples=2, 
        # REMOVED: image_key="image", image_threshold=0  <-- These were breaking the crop!
    ),
    
    # Spatial Augmentations
    RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
    RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),
    RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, prob=0.3, mode=("trilinear", "nearest")),
    
    # Intensity Augmentations (Now they actually work!)
    RandScaleIntensityd(keys="image", factors=0.2, prob=0.5),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    RandGaussianNoised(keys="image", prob=0.3, std=0.05),
    RandGaussianSmoothd(keys="image", prob=0.2, sigma_x=(0.5, 1.0), sigma_y=(0.5, 1.0), sigma_z=(0.5, 1.0)),
    RandBiasFieldd(keys="image", prob=0.3, coeff_range=(0.0, 0.3)),
    RandCoarseDropoutd(
        keys=["image", "label"], 
        holes=3, 
        spatial_size=(16, 16, 8), 
        fill_value=0.0, 
        prob=0.2
    ),
])

val_transforms = Compose([
    LoadImaged(keys=["image", "label"], image_only=False),
    EnsureChannelFirstd(keys=["image", "label"]),
    EnsureTyped(keys=["image", "label"], dtype=torch.float32),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=voxel_spacing, mode=("bilinear", "nearest")),
    AsDiscreted(keys="label", threshold=0.5),
    NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
    CropForegroundd(keys=["image", "label"], source_key="image", margin=5),
])

# 3. Datasets and Loaders
train_dataset = CacheDataset(data=train_data, transform=train_transforms, num_workers=8)
val_dataset = CacheDataset(data=val_data, transform=val_transforms, num_workers=8)

train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True, 
    num_workers=8, 
    persistent_workers=True,
    pin_memory=torch.cuda.is_available()
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=1, 
    shuffle=False, 
    num_workers=8, 
    persistent_workers=True,
    pin_memory=torch.cuda.is_available()
)

# 4. Network, Loss, and Optimizer setup
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.fp32_precision = 'tf32'
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
autocast_device = "cuda" if torch.cuda.is_available() else "cpu"

model = SegResNetDS(
    spatial_dims=spatial_dims,
    init_filters=32,
    in_channels=in_model_channels,
    out_channels=out_model_channels,
    norm=("GROUP", {"num_groups": 8}),
    act=("MISH", {"inplace": True}),
    dsdepth=4,
).to(device)
ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.999))

if int(torch.__version__.split('.')[0]) >= 2:
    print("Compiling model for faster training...")
    model = torch.compile(model)
    ema_model = torch.compile(ema_model)

# DiceCELoss is generally more stable than pure Dice for small targets
# dice_loss_calculator = DiceCELoss(sigmoid=True, squared_pred=True)
dice_loss_calculator = DiceFocalLoss(
    sigmoid=True, 
    squared_pred=True, 
    gamma=2.0,       # The focusing parameter (2.0 is standard)
    lambda_dice=1.0, 
    lambda_focal=1.0 # Balance the two losses evenly
)
dice_metric = DiceMetric(include_background=False, reduction="mean")

optimizer = optim.AdamW(model.parameters(), lr=initial_learning_rate, weight_decay=1e-5)
# scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6) # Replaced manual poly decay
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1, eta_min=1e-6)

scaler = amp.GradScaler(enabled=torch.cuda.is_available())
writer = SummaryWriter()

# 5. Training Loop
best_metric = -1.0
best_metric_epoch = -1
epochs_no_improve = 0

for epoch in range(max_epochs):
    start_time = time()
    model.train()
    epoch_loss = 0
    step = 0

    current_lr = scheduler.get_last_lr()[0]
    print(f"\n--- Epoch {epoch + 1}/{max_epochs} (LR: {current_lr:.6e}) ---")

    for batch_data in train_loader:
        step += 1
        inputs = batch_data["image"].to(device, non_blocking=True)
        label_gt = batch_data["label"].to(device, non_blocking=True) 
        if hasattr(inputs, "as_tensor"):
            inputs = inputs.as_tensor()
            label_gt = label_gt.as_tensor()

        optimizer.zero_grad()

        with amp.autocast(device_type=autocast_device, enabled=torch.cuda.is_available()):
            outputs = model(inputs) # Now returns a tuple of 4 tensors
            
            # Deep Supervision weights (Full res gets 1.0, lower res gets progressively less)
            ds_weights = [1.0, 0.5, 0.25, 0.125]
            loss = 0.0
            
            for i, out in enumerate(outputs):
                # Scale the ground truth label down to match the deep supervision output size
                scaled_label = F.interpolate(label_gt, size=out.shape[2:], mode="nearest-exact")
                loss += ds_weights[i] * dice_loss_calculator(out, scaled_label)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        ema_model.update_parameters(model)

        epoch_loss += loss.item()
        
        if step % 20 == 0:
            print(f"Step {step}/{len(train_loader)} | Train Loss: {loss.item():.4f}")

    scheduler.step()
    
    avg_loss = epoch_loss / step
    writer.add_scalar("train/loss", avg_loss, epoch + 1)
    
    # 6. Validation Phase
    model.eval()
    with torch.no_grad():
        for val_batch in val_loader:
            val_images = val_batch["image"].to(device, non_blocking=True)
            val_label_gt = val_batch["label"].to(device, non_blocking=True)
            if hasattr(val_images, "as_tensor"):
                val_images = val_images.as_tensor()
                val_label_gt = val_label_gt.as_tensor()

            with amp.autocast(device_type=autocast_device, enabled=torch.cuda.is_available()):
                # Switched to standard Sliding Window with overlap for smoother edges
                val_outputs_logits = sliding_window_inference(
                    inputs=val_images,
                    roi_size=network_input_size,
                    sw_batch_size=4,
                    predictor=model,  # Use only the full-resolution output for validation
                    overlap=0.5, 
                    mode="gaussian"
                )

            val_pred = (torch.sigmoid(val_outputs_logits) > 0.5).float()
            dice_metric(y_pred=val_pred, y=val_label_gt)

        metric_val = dice_metric.aggregate().item()
        dice_metric.reset()
        writer.add_scalar("val/dice", metric_val, epoch + 1)

        if metric_val > best_metric:
            best_metric = metric_val
            best_metric_epoch = epoch + 1
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            # # 1. Get the state dict from inside the EMA wrapper
            # ema_state = ema_model.module.state_dict()
            
            # # 2. Strip the PyTorch 2.0 compiler prefix if it exists
            # clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in ema_state.items()}
            
            # # 3. Save the clean weights
            # torch.save(clean_state_dict, checkpoint_path)
            # print(f"*** New Best Dice: {best_metric:.4f} (Saved clean EMA to {checkpoint_path}) ***")
            print(f"*** New Best Dice: {best_metric:.4f} (Saved to {checkpoint_path}) ***")
        else:
            epochs_no_improve += 1

        print(f"Val Dice: {metric_val:.4f} | Best: {best_metric:.4f} @ Ep {best_metric_epoch} | Time: {time() - start_time:.1f}s")

        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping triggered.")
            break

    gc.collect()
    torch.cuda.empty_cache()

writer.close()
print("Training completed.")