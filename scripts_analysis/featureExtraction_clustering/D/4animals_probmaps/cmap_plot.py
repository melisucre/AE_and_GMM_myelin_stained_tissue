import numpy as np
import matplotlib.pyplot as plt

# Create a rectangular gradient from 0 to 1
gradient = np.linspace(0, 1, 256)
gradient_2d = np.tile(gradient, (50, 1))  # 50 = height, 256 = width

# Create figure
fig, ax = plt.subplots(figsize=(12, 2), dpi=100)
ax.set_axis_off()

# Plot with gist_stern_r colormap
cmap = plt.cm.gist_stern_r
im = ax.imshow(gradient_2d, cmap=cmap, aspect='auto', vmin=0, vmax=1)

# Add colorbar
# cbar = plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.1, fraction=0.9)
# cbar.set_label('Probability', fontsize=12)

plt.tight_layout()
plt.savefig('/.../AE/data/AE_usage/AE_model128/4animals/gmm/spherical_plots/manuscript/cmap_visualization.png', dpi=300, bbox_inches='tight')
# plt.show()