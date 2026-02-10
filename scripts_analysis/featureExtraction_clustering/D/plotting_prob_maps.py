import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from matplotlib.colors import ListedColormap
import os
import math
from z__color_dictionary_27_3_cov import matching_colors
import matplotlib.patches as patches

# this code was not used at the end. we ordered manually

animal_name = 'ISM_31'
stain = 'mye'
cov_type = 'spherical'
image_num = '46'  # ['28', '46', '57']
ae_name = 'AE_model128'
path = '/.../AE/data/'
patch_size = 128


path_mask = f'/.../AE/data/mtbi/{animal_name}/fine_masks/{image_num}_mask.png'
mask = Image.open(path_mask).convert('L')
mask = np.array(mask)
col_size = mask.shape[1]
n_col = col_size // patch_size
row_size = mask.shape[0]
n_row = row_size // patch_size


# n_clusters = 15


def making_plot_more_vert(n_clusters, mask):
    file_probs = f'/.../AE/data/AE_usage/AE_model128/{animal_name}/LS/mye/spherical/k{n_clusters}_latent_img{image_num}_clustered_spherical_predict_prob.npy'
    data = np.load(file_probs)

    # Transpose to get shape: (27, num_pixels)
    data = data.T
    # Reshape each cluster's probabilities to (n_row, n_col)
    reshaped = [np.reshape(prob, (n_row, n_col)) for prob in data]


    mask = (mask // 255).astype(np.uint8)
    mask_resized = np.array(Image.fromarray(mask).resize((n_col, n_row), resample=Image.NEAREST))
    mask_resized = (mask_resized > 0).astype(np.uint8)
    masked_probs = [np.where(mask_resized != 0, prob, np.nan) for prob in reshaped]


    n_cols = 4
    #n_rows = n_clusters//n_cols
    n_rows = 7

    path_save = f'/.../AE/data/AE_usage/AE_n9_mye_w128_LS256_batch64_batchnorm/{animal_name}/prob_maps_new/'
    os.makedirs(path_save, exist_ok=True)

    # --------------------------------------------------------------------------------------------------
    # lets try another thing:

    color_dict = {'black': '#000000', 'grey': '#D3D3D3', 'grey2': '#808080', 'white': '#FFFFFF', 'p4': '#3C194E',
                  'p3': '#6A2C89', 'p2': '#AD48E0', 'p1': '#BB80DF',
                  'r2': '#73001F', 'r1': '#CA0036', 'o2': '#ED6529', 'o1': '#FF9439', 'y': '#FFF266', 'b4': '#221B6E', # canvi y #FFD548
                  'b3': '#4638E1', 'b2': '#4F9EFF', 'b1': '#46C7E1',
                  'g4': '#225F36', 'g3': '#379958', 'g2': '#49CC75', 'g1': '#B9E884', 'deep_orange': '#C2470E',
                  'rose_pink': '#EF3676', 'amber': '#FFC107', # he canviat rose_pink, anterior: #EF5350
                  'teal': '#009688', 'brown': '#795548', 'light_coral': '#EDA2CF', 'olive': '#808000'} # he canviat lightcoral, abans: #F08080


    color_names = [
        'black', 'grey', 'grey2', 'rose_pink', 'amber', 'p1', 'b2', 'g2', 'o2', 'g4',
        'b1', 'p2', 'b3', 'b4', 'g3', 'brown', 'p4', 'r1', 'teal', 'y', 'g1',
        'deep_orange', 'white', 'r2', 'o1', 'p3', 'light_coral', 'olive'
    ]

    ordenats = [10, 13, 4,
                5, 2, 14, 11,
                3, 1, 15, 6,
                23, 16, 7, 20,
                17, 8, 21, 19,
                24, 22, 18, 12,
                9, 25, 27, 26]


    def hex_to_rgb(hex_color):
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    all_color_mapping = matching_colors('ISM_31')
    new_color_mapping = all_color_mapping[n_clusters//3-1]
    cluster_ids = sorted(new_color_mapping.keys())
    color_list = [color_dict[new_color_mapping[cl]] for cl in cluster_ids]

    # # -------------------# -------------------# -------------------# -------------------# -------------------
    # lets compute the probs:
    # Flatten each masked cluster probability map
    masked_flat = [p.flatten() for p in masked_probs]
    masked_array = np.vstack(masked_flat)
    sums = np.nansum(masked_array, axis=1)
    percentages = 100 * sums / np.nansum(sums)
    # --------------------------------------------------------------------------------------------
    # lets order by percentage

    # Sort indices by descending percentage
    sorted_indices = np.argsort(-percentages)  # Negative for descending

    # Reorder masked_probs and percentages
    masked_probs_sorted = [masked_probs[i] for i in sorted_indices]
    percentages_sorted = percentages[sorted_indices]
    color_list_sorted = [color_list[i] for i in sorted_indices]  # If colors are per-cluster
    percentages = 100 * sums / np.nansum(sums)
    # --------------------------------------------------------------------------------------------

    # Plotting
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2 * n_cols, 1.2 * n_rows))
    axes = axes.flatten()

    # --- Legend in first subplot ---
    cbar_ax = axes[0]
    cbar_ax.axis('off')

    # Create horizontal gradient from 0 to 1, shaped like mask_resized
    grad = np.linspace(0, 1, mask_resized.shape[1]).reshape(1, -1)
    grad = np.tile(grad, (mask_resized.shape[0], 1))

    # Mask it — only show inside tissue
    grad_masked = np.where(mask_resized == 1, grad, np.nan)

    # Plot the masked gradient
    cmap = plt.colormaps.get_cmap('CMRmap_r').copy()
    cmap.set_bad(color='white')
    cbar_ax.imshow(grad_masked, cmap=cmap, vmin=0, vmax=1)

    # Add labels and title
    cbar_ax.text(-0.07, 0.5, "0", va='center', ha='left', fontsize=8, transform=cbar_ax.transAxes)
    cbar_ax.text(1.07, 0.5, "1", va='center', ha='right', fontsize=8, transform=cbar_ax.transAxes)
    cbar_ax.text(0.5, 1.15, "Tissue probability per cluster",
                 va='top', ha='center', fontsize=8, transform=cbar_ax.transAxes)

    # --- Cluster probability maps ---
    for i in range(1, len(axes)):  # Start from 1
        ax = axes[i]
        cluster_idx = i - 1  # Adjust index for masked_probs_sorted

        if cluster_idx < n_clusters:
            prob = masked_probs_sorted[cluster_idx]
            prob_nan = np.where(mask_resized == 0, np.nan, prob)

            cluster_color = color_list_sorted[cluster_idx]
            cmap = plt.colormaps.get_cmap('CMRmap_r').copy()
            cmap.set_bad(color=cluster_color)

            im = ax.imshow(prob_nan, cmap=cmap, vmin=0, vmax=1)
            # ax.set_title(f"{percentages_sorted[cluster_idx]:.1f}%", fontsize=8)

            # Top-left: Probability %
            ax.text(0.01, 1.15, f"{percentages_sorted[i-1]:.1f}%",
                    va='top', ha='left', fontsize=7,
                    color='black', transform=ax.transAxes)

            # Top-right: Cluster ID from `ordenats`
            ax.text(0.99, 1.15, f"{ordenats[i-1]}",
                    va='top', ha='right', fontsize=7,
                    color='black', transform=ax.transAxes)

        else:
            ax.axis('off')
        ax.axis('off')

    plt.tight_layout()
    plt.tight_layout(h_pad=1, w_pad=0.5)
    # plt.colorbar(im, ax=axes.tolist(), shrink=0.5)
    plt.savefig(f"{path_save}im{image_num}_{n_clusters}_cluster_probs_order_v2_CMRmap_reordered_tight_clust_.png", dpi=300, bbox_inches='tight')
    plt.close()

for n_clusters in [27]:
    making_plot_more_vert(n_clusters, mask)
