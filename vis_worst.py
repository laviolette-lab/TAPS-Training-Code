import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from monai.apps import DecathlonDataset
from monai.data import DataLoader
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
from monai.utils import set_determinism
from skimage import measure

def draw_contours(ax, mask, color, linewidth=2):
    """
    Draw contours of a binary mask on an existing matplotlib axis.
    """
    contours = measure.find_contours(mask.astype(float), 0.5)

    for contour in contours:
        ax.plot(
            contour[:, 1],
            contour[:, 0],
            color=color,
            linewidth=linewidth,
        )

def get_optimal_sw_batch_size(device):
    if device.type == "cuda":
        total_mem_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        if total_mem_gb >= 20:
            return 24
        elif total_mem_gb >= 12:
            return 16
        elif total_mem_gb >= 8:
            return 8
        elif total_mem_gb >= 4:
            return 4
        return 2
    return 4


def dice_score(pred, gt):
    pred = pred.astype(bool)
    gt = gt.astype(bool)

    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()

    if denom == 0:
        return 1.0

    return 2.0 * intersection / denom


def overlay(image, gt, pred):
    """
    RGB overlay:
      Green = GT
      Red   = Prediction
      Yellow = overlap
    """
    image = image.astype(np.float32)
    image -= image.min()
    image /= image.max() + 1e-8

    rgb = np.stack([image] * 3, axis=-1)

    rgb[gt] = [0, 1, 0]
    rgb[pred] = [1, 0, 0]
    rgb[np.logical_and(gt, pred)] = [1, 1, 0]

    return rgb


def main():

    set_determinism(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        EnsureTyped(keys=["image", "label"], dtype=torch.float32),

        Lambdad(keys="image", func=lambda x: x[0:1]),
        Lambdad(keys="label", func=lambda x: (x > 0).float()),

        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=(0.8, 0.8, 3.0),
            mode=("bilinear", "nearest"),
        ),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=5),
    ])

    dataset = DecathlonDataset(
        root_dir="./data",
        task="Task05_Prostate",
        transform=transforms,
        section="training",
        download=False,
        cache_rate=1.0,
        val_frac=0.0,
    )

    loader = DataLoader(dataset, batch_size=1)

    model = SegResNetDS(
        spatial_dims=3,
        init_filters=32,
        in_channels=1,
        out_channels=1,
        norm=("GROUP", {"num_groups": 8}),
        act=("MISH", {"inplace": True}),
        dsdepth=4,
    ).to(device)

    weights = torch.load(
        "best_segresnet_model.pth",
        map_location=device,
        weights_only=True,
    )

    weights = {k.replace("_orig_mod.", ""): v for k, v in weights.items()}
    model.load_state_dict(weights)
    model.eval()

    sw_batch_size = get_optimal_sw_batch_size(device)

    results = []

    print("Running inference...")

    with torch.no_grad():

        for idx, batch in enumerate(loader):

            image = batch["image"].to(device)
            label = batch["label"].to(device)

            with torch.amp.autocast(
                device_type=autocast_device,
                enabled=(device.type == "cuda"),
            ):

                logits = sliding_window_inference(
                    image,
                    roi_size=(128, 128, 32),
                    sw_batch_size=sw_batch_size,
                    predictor=model,
                    overlap=0.5,
                    mode="gaussian",
                )

            pred = (torch.sigmoid(logits) > 0.5).float()

            image_np = image.cpu().numpy()[0, 0]
            gt_np = label.cpu().numpy()[0, 0].astype(bool)
            pred_np = pred.cpu().numpy()[0, 0].astype(bool)

            d = dice_score(pred_np, gt_np)

            results.append({
                "index": idx,
                "dice": d,
                "image": image_np,
                "gt": gt_np,
                "pred": pred_np,
            })

            print(f"Case {idx:02d}: Dice = {d:.4f}")

    results.sort(key=lambda x: x["dice"])

    os.makedirs("worst_cases", exist_ok=True)

    N = 5

    print("\nWorst cases:")

    for rank, case in enumerate(results[:N]):

        print(
            f"{rank+1}: Case {case['index']} "
            f"Dice={case['dice']:.4f}"
        )

        gt = case["gt"]

        # Slice with largest GT area
        areas = gt.sum(axis=(0, 1))
        z = np.argmax(areas)

        img = case["image"][:, :, z]
        gt_slice = gt[:, :, z]
        pred_slice = case["pred"][:, :, z]

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        # -------------------------
        # Left: MRI only
        # -------------------------
        axes[0].imshow(img, cmap="gray")
        axes[0].set_title("MRI")
        axes[0].axis("off")

        # -------------------------
        # Right: Overlay
        # -------------------------
        axes[1].imshow(img, cmap="gray")

        draw_contours(axes[1], gt_slice, color="lime", linewidth=2)
        draw_contours(axes[1], pred_slice, color="red", linewidth=2)

        axes[1].set_title("Segmentation")
        axes[1].axis("off")

        legend_elements = [
            Patch(facecolor="lime", edgecolor="lime", label="Ground Truth"),
            Patch(facecolor="red", edgecolor="red", label="Prediction"),
            Patch(facecolor="yellow", edgecolor="yellow", label="Overlap"),
        ]

        axes[1].legend(
            handles=legend_elements,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.08),
            ncol=3,
            frameon=True,
            fontsize=9,
        )

        fig.suptitle(
            f"Worst Case #{rank+1}   |   Dataset Index: {case['index']}   |   Dice = {case['dice']:.3f}",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        plt.savefig(
            f"worst_cases/{rank+1:02d}_case_{case['index']}.png",
            dpi=200,
            bbox_inches="tight",
        )

        plt.close()

    print("\nSaved figures to ./worst_cases/")


if __name__ == "__main__":
    main()