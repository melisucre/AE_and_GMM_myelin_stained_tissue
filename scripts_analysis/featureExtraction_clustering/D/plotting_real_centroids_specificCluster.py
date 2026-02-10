import numpy as np
from scipy.spatial.distance import cdist
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
import os

main_path = '.../AE/data/AE_usage/AE_model128/'
cluster_id = 10
k = 15
threshold = 0.95
N = 4
window = 128
ims_range = range(19, 67)  # inclusive upper range

# Load centroid for cluster 10
Centroids = np.load(main_path + 'ISM_31/gmm/100000_means_spherical_regcove-0.001.npy', allow_pickle=True)
centroid = Centroids[k // 3 - 1][cluster_id]

# Store info from all sections
all_latents = []
all_indices = []
all_image_ids = []

for im in ims_range:
    im_str = str(im)
    try:
        LS_path = main_path + f'ISM_31/LS/all/latent/latent_img{im_str}.npy'
        Prob_path = main_path + f'ISM_31/LS/all/gmm/k{k}_latent_img{im_str}_clustered_spherical_predict_prob.npy'

        LS = np.load(LS_path)
        Prob = np.load(Prob_path)

        probs = Prob[:, cluster_id]
        high_prob_idx = np.where(probs >= threshold)[0]

        if len(high_prob_idx) == 0:
            continue

        latents = LS[high_prob_idx]
        all_latents.append(latents)
        all_indices.append(high_prob_idx)
        all_image_ids.extend([im_str] * len(high_prob_idx))

    except Exception as e:
        print(f"Skipping image {im_str} due to error: {e}")
        continue

# Combine
if not all_latents:
    raise ValueError("No high-probability points found in any image!")

all_latents_combined = np.vstack(all_latents)
all_indices_combined = np.concatenate(all_indices)

# Compute distances
dists = cdist(all_latents_combined, centroid[None, :])[:, 0]
top_n_idx = np.argsort(dists)[:N]

selected_image_ids = [all_image_ids[i] for i in top_n_idx]
selected_indices = [all_indices_combined[i] for i in top_n_idx]

# --- Plotting setup ---
def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

color_dict = {'black': '#000000', 'grey': '#D3D3D3', 'grey2': '#808080', 'white': '#FFFFFF', 'p4': '#3C194E',
              'p3': '#6A2C89', 'p2': '#AD48E0', 'p1': '#BB80DF', 'r2': '#73001F', 'r1': '#CA0036',
              'o2': '#ED6529', 'o1': '#FF9439', 'y': '#FFF266', 'b4': '#221B6E', 'b3': '#4638E1',
              'b2': '#4F9EFF', 'b1': '#46C7E1', 'g4': '#225F36', 'g3': '#379958', 'g2': '#49CC75',
              'g1': '#B9E884', 'deep_orange': '#C2470E', 'rose_pink': '#EF3676', 'amber': '#FFC107',
              'teal': '#009688', 'brown': '#795548', 'light_coral': '#EDA2CF', 'olive': '#808000'}

color_names = ['black', 'grey', 'grey2', 'rose_pink', 'amber', 'p1', 'b2', 'g2', 'o2', 'g4',
               'b1', 'p2', 'b3', 'b4', 'g3', 'brown', 'p4', 'r1', 'teal', 'y', 'g1',
               'deep_orange', 'white', 'r2', 'o1', 'p3', 'light_coral', 'olive']

color_hex = color_dict[color_names[cluster_id]]
color_rgb = hex_to_rgb(color_hex)

# Image canvas settings
pad = 5
spacing = 5
bar_height = 5

# Load and crop windows
windows = []
for im_str, idx in zip(selected_image_ids, selected_indices):
    orig_path = f'/.../AE/data/mtbi/ISM_31/renormalized/mye/{im_str}.png'
    image = np.array(Image.open(orig_path).convert('L'))
    n_col = image.shape[1] // window

    row = idx // n_col
    col = idx % n_col
    x_start = col * window
    y_start = row * window
    window_img = image[y_start:y_start + window, x_start:x_start + window]

    windows.append(Image.fromarray(window_img).convert('RGB'))

# Pad windows to exact N if needed
while len(windows) < N:
    windows.append(windows[-1])

# Create canvas
win_size = windows[0].size[0]
padded_win_size = win_size + 2 * pad
total_width = N * padded_win_size + 34
total_height = 2 * bar_height + padded_win_size

canvas = Image.new('RGB', (total_width, total_height), color=color_rgb)

# Draw color bar
draw = ImageDraw.Draw(canvas)
draw.rectangle([(0, 0), (total_width, bar_height)], fill=color_rgb)

# Paste windows
for j, win in enumerate(windows):
    x = j * (padded_win_size + spacing) + 10
    y = bar_height
    padded_bg = Image.new('RGB', (padded_win_size, padded_win_size), color=color_rgb)
    padded_bg.paste(win, (pad, pad))
    canvas.paste(padded_bg, (x, y))

# Save
out_dir = main_path + 'ISM_31/LS/mye/cluster_visualizations/'
os.makedirs(out_dir, exist_ok=True)
canvas.save(os.path.join(out_dir, f'cluster{cluster_id}_combined.png'))
print(f"Saved cluster {cluster_id} combined example to: {out_dir}")
