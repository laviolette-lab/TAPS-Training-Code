import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    CropForegroundd,
    Lambdad,
)

def get_msd_sample_images():
    """Loads a real MSD patient and extracts slices before and after preprocessing."""
    base_dir = "./data/Task05_Prostate"
    json_path = os.path.join(base_dir, "dataset.json")
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find MSD dataset at {json_path}. Make sure it is downloaded.")

    with open(json_path, "r") as f:
        dataset_info = json.load(f)
    
    # Grab the first patient
    item = dataset_info["training"][0]
    raw_dict = {
        "image": os.path.join(base_dir, item["image"]),
        "label": os.path.join(base_dir, item["label"])
    }

    # 1. Load Raw (Just extraction, no geometry changes)
    raw_transforms = Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Lambdad(keys="image", func=lambda x: x[0:1, ...]), # Isolate T2
        Lambdad(keys="label", func=lambda x: (x > 0).float()), # Binary mask
    ])
    raw_data = raw_transforms(raw_dict)

    # 2. Apply TAPS Preprocessing
    pre_transforms = Compose([
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(0.8, 0.8, 3.0), mode=("bilinear", "nearest")),
        NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
        CropForegroundd(keys=["image", "label"], source_key="image", margin=5),
    ])
    pre_data = pre_transforms(raw_data)

    # Extract Numpy Arrays
    r_img, r_lbl = raw_data["image"][0].numpy(), raw_data["label"][0].numpy()
    p_img, p_lbl = pre_data["image"][0].numpy(), pre_data["label"][0].numpy()

    # Find the best slice (Center of mass of the mask)
    r_z = np.argmax(r_lbl.sum(axis=(0, 1)))
    p_z = np.argmax(p_lbl.sum(axis=(0, 1)))

    # Format for visual plotting
    def format_slice(img_vol, lbl_vol, z_idx):
        img_slice = np.rot90(img_vol[:, :, z_idx])
        lbl_slice = np.rot90(lbl_vol[:, :, z_idx])
        p1, p99 = np.percentile(img_slice, (1, 99))
        img_slice = np.clip(img_slice, p1, p99)
        img_slice = (img_slice - p1) / (p99 - p1 + 1e-8) # Normalize 0-1
        return img_slice, lbl_slice

    img_raw, lbl_raw = format_slice(r_img, r_lbl, r_z)
    img_pre, lbl_pre = format_slice(p_img, p_lbl, p_z)

    return img_raw, lbl_raw, img_pre, lbl_pre

def create_rgb_overlay(img, mask, color=(1, 0, 0)):
    """Creates a red overlay for the mask."""
    rgb = np.stack([img, img, img], axis=-1)
    for i in range(3):
        rgb[:, :, i] = np.where(mask > 0.5, rgb[:, :, i] * 0.5 + color[i] * 0.5, rgb[:, :, i])
    return rgb

def create_pipeline_flowchart():
    print("Extracting real MSD sample...")
    img_raw, lbl_raw, img_pre, lbl_pre = get_msd_sample_images()
    print("Generating Figure...")

    # Set up the figure canvas
    fig = plt.figure(figsize=(16, 7))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # 1. Draw the encompassing "TAPS Engine" background box
    taps_box = patches.FancyBboxPatch(
        (0.20, 0.05), 0.60, 0.80,
        boxstyle="round,pad=0.02",
        edgecolor="#8c8c8c", facecolor="#f8f9fa", 
        linestyle="--", linewidth=2, zorder=0
    )
    ax.add_patch(taps_box)

    # Clean title placement with a background mask to prevent dashed-line intersection
    ax.text(
        0.5, 0.87, "  TAPS Automated Software Pipeline  ", 
        ha='center', va='center', fontsize=15, fontweight='bold', color="#495057",
        bbox=dict(facecolor='#f8f9fa', edgecolor='none', pad=3.0), zorder=2
    )

    # 2. Define the pipeline nodes
    nodes = [
        {"x": 0.1, "y": 0.23, "color": "#d0ebff", "title": "Input MRI", "text": "Original NIfTI\n(Arbitrary Spacing\n& Orientation)"},
        {"x": 0.3, "y": 0.23, "color": "#d3f9d8", "title": "Preprocessing", "text": "• Orient to RAS\n• Spacing (0.8x0.8x3.0)\n• Intensity Norm.\n• Crop Foreground"},
        {"x": 0.5, "y": 0.23, "color": "#ffe8cc", "title": "SegResNetDS", "text": "Deep Supervision\nResidual U-Net\n(128x128x32 window)"},
        {"x": 0.7, "y": 0.23, "color": "#d3f9d8", "title": "Affine Inversion", "text": "• Restore Padding\n• Restore Spacing\n• Restore Orientation\n(Nearest Neighbor)"},
        {"x": 0.9, "y": 0.23, "color": "#d0ebff", "title": "Output Mask", "text": "Binary NIfTI\n(Matches Original\nPatient Geometry)"}
    ]

    box_width, box_height = 0.15, 0.24

    # 3. Draw the text nodes
    for node in nodes:
        box = patches.FancyBboxPatch(
            (node["x"] - box_width/2, node["y"] - box_height/2),
            box_width, box_height,
            boxstyle="round,pad=0.02",
            edgecolor="#343a40", facecolor=node["color"], linewidth=1.5, zorder=1
        )
        ax.add_patch(box)
        ax.text(node["x"], node["y"] + 0.075, node["title"], ha='center', va='center', fontsize=11, fontweight='bold', zorder=2)
        ax.text(node["x"], node["y"] - 0.035, node["text"], ha='center', va='center', fontsize=10, linespacing=1.5, zorder=2)

    # 4. Draw connecting arrows
    for i in range(len(nodes) - 1):
        start_x = nodes[i]["x"] + box_width/2 + 0.005
        end_x = nodes[i+1]["x"] - box_width/2 - 0.005
        ax.annotate('', xy=(end_x, 0.23), xytext=(start_x, 0.23),
                    arrowprops=dict(arrowstyle="->,head_length=0.8,head_width=0.4", linewidth=2.5, color='#495057'), zorder=0)

    # 5. Add Real Image Visualizations Above the Nodes
    img_y, img_h, img_w = 0.46, 0.34, 0.12
    
    # Raw Image
    ax_raw = fig.add_axes([nodes[0]["x"] - img_w/2, img_y, img_w, img_h])
    ax_raw.imshow(img_raw, cmap="gray")
    ax_raw.axis("off")
    
    # Preprocessed Image
    ax_pre = fig.add_axes([nodes[1]["x"] - img_w/2, img_y, img_w, img_h])
    ax_pre.imshow(img_pre, cmap="gray")
    ax_pre.axis("off")
    
    # Network Graphic
    ax_net = fig.add_axes([nodes[2]["x"] - img_w/2, img_y, img_w, img_h])
    ax_net.axis("off")
    ax_net.text(0.5, 0.5, "[ AI Inference ]\nSliding Window", ha="center", va="center", fontsize=11, fontweight="bold", color="#d9480f", bbox=dict(facecolor='#fff4e6', edgecolor='#d9480f', boxstyle='round,pad=1'))

    # Inverted Mask
    ax_inv = fig.add_axes([nodes[3]["x"] - img_w/2, img_y, img_w, img_h])
    ax_inv.imshow(lbl_raw, cmap="Reds")
    ax_inv.axis("off")
    
    # Final Output
    overlay = create_rgb_overlay(img_raw, lbl_raw)
    ax_out = fig.add_axes([nodes[4]["x"] - img_w/2, img_y, img_w, img_h])
    ax_out.imshow(overlay)
    ax_out.axis("off")

    # Save and display
    output_filename = "figure_1_taps_pipeline_real_images.png"
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure successfully saved as '{output_filename}'")

if __name__ == "__main__":
    create_pipeline_flowchart()