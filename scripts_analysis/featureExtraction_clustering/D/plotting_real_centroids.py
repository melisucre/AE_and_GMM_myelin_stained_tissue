import numpy as np
from scipy.spatial.distance import cdist
import os
import numpy as np
from PIL import Image, ImageDraw
Image.MAX_IMAGE_PIXELS = None
import matplotlib
matplotlib.use('TkAgg')
from color_dictionary_27_3_cov import matching_colors

im = '42' # 21, 36, 47 ['28', '46', '57']
main_path = '.../AE/data/AE_usage/AE_n9_mye_w128_LS256_batch64_batchnorm/'
# Load latent space vectors and GMM outputs
LS = np.load(main_path+f'ISM_31/LS/all/latent/latent_img{im}.npy')

k = 21 # number of clusters i want
Prob = np.load(main_path+f'ISM_31/LS/all/gmm[{k}]/k{k}_latent_img{im}_clustered_spherical_predict_prob.npy')
Centroids = np.load(main_path+'ISM_31/gmm/100000_means_spherical_regcove-0.001.npy', allow_pickle=True)

centroids = Centroids[k//3 -1]

N = 3  # number of top examples to keep per cluster
# Probability threshold
threshold = 0.10

selected_clusters = []  # tuples of (cluster_index, top_n_indices)

for i in range(centroids.shape[0]):
    probs = Prob[:, i]
    high_prob_idx = np.where(probs >= threshold)[0]

    if len(high_prob_idx) == 0:
        print(f"No high-prob points for cluster {i}")
        continue

    high_prob_latents = LS[high_prob_idx]
    dists = cdist(high_prob_latents, centroids[i:i + 1])[:, 0]

    top_n_idx_in_subset = np.argsort(dists)[:N]
    top_n_indices = high_prob_idx[top_n_idx_in_subset]

    selected_clusters.append((i, top_n_indices.tolist()))

# there's something wrong with cluster 6, i just get one position
# len(selected_indices)

original = f'/.../AE/data/mtbi/ISM_31/renormalized/mye/{im}.png'
window = 128

image = np.array(Image.open(original).convert('L'))
n_col = image.shape[1] // window
n_row = image.shape[0] // window

windows_by_cluster = []

for cluster_index, cluster_indices in selected_clusters:
    cluster_windows = []
    for idx in cluster_indices:
        row = idx // n_col
        col = idx % n_col
        x_start = col * window
        y_start = row * window
        window_img = image[y_start:y_start + window, x_start:x_start + window]
        cluster_windows.append(window_img)
    windows_by_cluster.append((cluster_index, cluster_windows))

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

def hex_to_rgb(hex_color):
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

new_color_mappings = matching_colors('ISM_31')
new_color_mapping = new_color_mappings[k//3 -1]
cluster_ids = sorted(new_color_mapping.keys())
color_list = [color_dict[new_color_mapping[cl]] for cl in cluster_ids]

out_dir = main_path +'ISM_31/LS/mye/cluster_visualizations_21_vertical_3/'
os.makedirs(out_dir, exist_ok=True)


# making it vertical:

pad = 5  # padding around each window
spacing = 5  # spacing between windows
bar_height = 5  # optional top bar height

for cluster_index, cluster_windows in windows_by_cluster:
    while len(cluster_windows) < N:
        cluster_windows.append(cluster_windows[-1])
    cluster_windows = cluster_windows[:N]

    rgb_windows = [Image.fromarray(w).convert('RGB') for w in cluster_windows]
    win_size = rgb_windows[0].size[0]
    color_rgb = hex_to_rgb(color_list[cluster_index])  # NOTE: use actual index

    padded_win_size = win_size + 2 * pad
    total_width = padded_win_size + 20  # optional side padding
    total_height = N * (padded_win_size + spacing) + bar_height + spacing

    canvas = Image.new('RGB', (total_width, total_height), color=color_rgb)

    draw = ImageDraw.Draw(canvas)
    draw.rectangle([(0, 0), (total_width, bar_height)], fill=color_rgb)

    for j, win in enumerate(rgb_windows):
        x = 10  # optional left margin
        y = bar_height + spacing + j * (padded_win_size + spacing)
        padded_bg = Image.new('RGB', (padded_win_size, padded_win_size), color=color_rgb)
        padded_bg.paste(win, (pad, pad))
        canvas.paste(padded_bg, (x, y))

    canvas.save(os.path.join(out_dir, f'cluster{cluster_index}_{im}_{threshold}.png'))

