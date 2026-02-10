import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import os
from skimage.measure import block_reduce

stain = 'mye'
cov_type = 'spherical'
animal_names = ['ISM_26', 'ISM_30', 'ISM_31', 'ISM_37']
# image_nums = [['27', '45', '57'], ['28', '46', '57'], ['28', '46', '57'], ['23', '41', '53']]
image_nums = [['45'], ['46'], ['46'], ['41']]
ae_names = ['AE_model128', 'AE_model256']
path = '/.../AE/data/'
patch_sizes = [128, 256]
# klusters = [3, 9, 21]
klusters = [9]


def plot_probability_map(prob_map, mask_down, n_row, n_col, cluster_id, animal_name, im, path_save, n_clusters):
    """Plot probability map with white background and black text showing presence %"""

    # Reshape probability map to spatial dimensions
    prob_spatial = np.reshape(prob_map, (n_row, n_col))

    # Calculate presence percentage
    n_tissue_pixels = np.sum(mask_down > 0)
    tissue_probs = prob_spatial[mask_down > 0]
    total_presence = 100 * np.sum(tissue_probs) / n_tissue_pixels

    # Set non-tissue to grey, tissue keeps probability values
    prob_display = np.where(mask_down, prob_spatial, 0.9)  # 0.5 = medium grey in normalized scale

    # Create plot
    plt.figure(figsize=(8, 8), dpi=100, facecolor='#E8E8E8')
    ax = plt.gca()
    ax.set_axis_off()
    ax.set_facecolor('#E8E8E8')  # Grey background

    # Use gist_stern_r: black for high prob, white for low prob
    # Only plot tissue areas (mask=1), leave non-tissue transparent
    prob_tissue_only = np.where(mask_down, prob_spatial, np.nan)
    cmap = plt.cm.gist_stern_r
    im_obj = ax.imshow(prob_tissue_only, cmap=cmap, vmin=0, vmax=1)

    # Add presence percentage text in black at top-left corner
    ax.text(0.02, 0.98, f'{total_presence:.1f}%',
            transform=ax.transAxes, fontsize=18, color='black',
            verticalalignment='top', fontweight='bold')

    # Save figure
    plt.savefig(path_save, bbox_inches='tight', pad_inches=0, dpi=800, transparent=False)
    plt.close()


# Main loop
p = -1
for ae_name in ae_names:
    p += 1
    s = -1
    patch_size = patch_sizes[p]

    for animal_name in animal_names:
        s += 1
        image_num = image_nums[s]

        path_masks = f'{path}mtbi/{animal_name}/fine_masks/mye/'

        for im in image_num:
            # Load and process mask
            path_mask = f'{path_masks}{im}_mask.png'
            mask = np.array(Image.open(path_mask).convert('L'))
            n_col = mask.shape[1] // patch_size
            n_row = mask.shape[0] // patch_size
            mask_down = block_reduce(mask > 0, block_size=(patch_size, patch_size), func=np.max)

            for k in klusters:
                # Load probability predictions
                path_probs = f'{path}AE_usage/{ae_name}/4animals/gmm/{cov_type}/{animal_name}_k{k}_num{im}_clustered_predict_prob.npy'
                data_probs = np.load(path_probs)
                # data_probs shape: (num_pixels, n_clusters)

                n_clusters = data_probs.shape[1]

                # Create folder structure: k{k}/cluster_{i}/
                # base_save_dir = f'{path}AE_usage/{ae_name}/4animals/gmm/{cov_type}_plots/k{k}'
                base_save_dir = f'{path}AE_usage/{ae_name}/4animals/gmm/{cov_type}_plots/manuscript/k{k}'
                os.makedirs(base_save_dir, exist_ok=True)

                # Plot each cluster
                for cluster_id in range(n_clusters):
                    cluster_dir = f'{base_save_dir}/cluster_{cluster_id}'
                    os.makedirs(cluster_dir, exist_ok=True)

                    # Extract probability for this cluster
                    prob_map = data_probs[:, cluster_id]

                    # Create save path with animal name and image number
                    file_name = f'{animal_name}_{im}_cluster{cluster_id}_prob.png'
                    save_path = f'{cluster_dir}/{file_name}'

                    # Plot and save
                    plot_probability_map(prob_map, mask_down, n_row, n_col,
                                         cluster_id, animal_name, im, save_path, n_clusters)

                    print(f'Saved: {save_path}')