import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# ae_name = 'AE_model128'
pca_name = 'PCA_model128'

animal_name = 'ISM_31'
# folder = f'/.../AE/data/AE_usage/{ae_name}/{animal_name}/LS/spherical_plots/'
folder = f'/.../AE/data/massivePCA/{pca_name}/data/{animal_name}/pca/spherical_plots/'

# Cluster values to use
clusters = list(range(3, 28, 3))

# Set up plot
fig, axes = plt.subplots(3, 3, figsize=(10, 10))
axes = axes.flatten()

for i, k in enumerate(clusters):
    # filename = f"k{k}.png"
    filename = f'k{k}_section46_gmm_clusters.png'
    # filename = f"k{k}_46.png" # ISM_30 , ISM_31
    # filename = f"k{k}_45.png"   # ISM_26
    # filename = f'k{k}_section45_gmm_clusters.png'
    # filename = f"k{k}_41.png"  # ISM_37
    # filename = f'k{k}_section41_gmm_clusters.png'
    path = os.path.join(folder, filename)

    # Load image
    img = Image.open(path).convert("RGB")

    # Convert image to array for Matplotlib
    img_array = np.array(img)

    # Plot image
    ax = axes[i]
    ax.imshow(img_array)
    ax.axis("off")

    # Add text in top-left corner
    # ax.text(0.10, 0.90, f"{k} clusters", color="black", fontsize=12,
    ax.text(0.15, 0.97, f"{k} clusters", color="black", fontsize=12,
            ha='center', va='bottom', transform=ax.transAxes,
            bbox=dict(facecolor='white', edgecolor='none', alpha=1, boxstyle='round'))

plt.subplots_adjust(
    wspace=0.01,  # space between columns
    hspace=-0.4,  # space between rows
    left=0.01, right=0.99,
    top=0.99, bottom=0.01
)

plt.tight_layout()
plt.savefig(folder + f'{animal_name}_altogether.png', dpi=1000, bbox_inches='tight', pad_inches=0) #, box_inches='tight', dpi=800)
plt.close()
