import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import os

# Parameters

animals = ['ISM_31', 'ISM_37', 'ISM_26', 'ISM_30']

ae_names = ['AE_model128', 'AE_model256']
LS = [128, 256]
linestyles = ['-', ':']
model_labels = ['model128', 'model256']

reg_covar_value = 0.001
num_patches = 100000
N = np.array([3, 6, 9, 12, 15, 18, 21, 24, 27])
cov_type = 'spherical'

colors = ['#FF9654', '#FF515D', '#526FFF', '#52B4FF']

# Plot setup
plt.figure(figsize=(14, 8))
custom_labels = ['sham 1', 'sham 2', 'mTBI 1', 'mTBI 2']

for model_idx, ae_name in enumerate(ae_names):
    linestyle = linestyles[model_idx]

    for i, a in enumerate(animals):
        dir = f'/...AE/data/AE_usage/{ae_name}'
        base_dir = f'{dir}/{a}/gmm/'
        bic_path = os.path.join(base_dir, f'{num_patches}_bic_{cov_type}_regcove-{reg_covar_value}.npy')

        try:
            bic_values = np.load(bic_path)
        except FileNotFoundError:
            print(f"BIC file not found for {a} in {ae_name}")
            bic_values = [np.inf] * len(N)

        # Plot (no label in the main plot command)
        plt.plot(N, bic_values, marker='o', color=colors[i], linestyle=linestyle)

        if np.isfinite(bic_values).any():
            min_idx = np.argmin(bic_values)
            min_val = bic_values[min_idx]
            plt.plot(N[min_idx], min_val, 'k*', markersize=12)

# Create custom legend with two sections
# Animal colors
animal_handles = [Line2D([0], [0], color=colors[i], marker='o', linestyle='-', label=custom_labels[i] if i < len(custom_labels) else animals[i])
                  for i in range(len(animals))]

# Model linestyles
model_handles = [Line2D([0], [0], color='black', linestyle=linestyles[i], label=model_labels[i])
                 for i in range(len(ae_names))]

# Combine legends
first_legend = plt.legend(handles=animal_handles, title='animals',
                          loc='upper right', fontsize=18, title_fontsize=18)
plt.gca().add_artist(first_legend)  # Keep first legend when adding second

plt.legend(handles=model_handles, title='models',
           loc='upper right', fontsize=18, title_fontsize=18, bbox_to_anchor=(0.85, 1.0))

plt.xlabel('number of clusters', fontsize=18)
plt.ylabel('BIC score', fontsize=18)
plt.title(f'BIC score comparison across number of clusters, animals and AE models', fontsize=24)
plt.grid(axis='y', linestyle=':', alpha=0.7)
plt.xticks(N, fontsize=18)
plt.tight_layout()

# Save
output = '/.../AE/data/AE_usage/BIC'
os.makedirs(output, exist_ok=True)
output_path = f'{output}/BIC_AEmodels.png'    # animals_{num_patches}_BIC_comparison_{ae_name}_{cov_type}_regcove-{reg_covar_value}.png'
plt.savefig(output_path, format='png', dpi=300)
plt.show()
