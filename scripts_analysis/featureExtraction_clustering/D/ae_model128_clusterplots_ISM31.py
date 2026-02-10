import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2
from matplotlib.colors import ListedColormap
import os
# from color_dictionary import matching_colors
from color_dictionary_27_3_cov import matching_colors
from skimage.measure import block_reduce


animal_name = 'ISM_31'
stain = 'mye'
cov_type = 'spherical'
image_num = ['28', '46', '57']
ae_name = 'AE_model128'
path = '.../AE/data/'

patch_size = 128

# --------------------------------------------
path_clusters = f'.../AE/data/AE_usage/{ae_name}/{animal_name}/LS/{stain}/{cov_type}_plots'
os.makedirs(path_clusters, exist_ok=True)
path_save = path_clusters


# plot:
def plotting(n, colors, im, mask_down, n_row, n_col):
    # -----------------------

    # path to data prepared for clustering
    path_labels = f'{path}AE_usage/{ae_name}/{animal_name}/LS/{stain}/{cov_type}/k{n}_latent_img{im}_clustered_{cov_type}.npy'

    # loading both image and data with labels
    data = np.load(path_labels)
    data = np.reshape(data, (n_row, n_col))

    # np.unique(data)

    # paleta = ListedColormap(sns.color_palette(colors))
    paleta = ListedColormap(colors)
    masked_data = np.ma.masked_where(~mask_down, data)

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_axis_off()

    img = ax.imshow(masked_data, vmin=0, vmax=len(paleta.colors), cmap=paleta)

    where = f'{path_save}/k{n}_{im}.png'
    plt.savefig(where, bbox_inches='tight', pad_inches=0, dpi=800, transparent=True)
    plt.close()


color_dict = {'black': '#000000', 'grey': '#D3D3D3', 'grey2': '#808080', 'white': '#FFFFFF', 'p4': '#3C194E',
              'p3': '#6A2C89', 'p2': '#AD48E0', 'p1': '#BB80DF',
              'r2': '#73001F', 'r1': '#CA0036', 'o2': '#ED6529', 'o1': '#FF9439', 'y': '#FFD548', 'b4': '#221B6E',
              'b3': '#4638E1', 'b2': '#4F9EFF', 'b1': '#46C7E1',
              'g4': '#225F36', 'g3': '#379958', 'g2': '#49CC75', 'g1': '#B9E884', 'deep_orange': '#C2470E',
              'rose_pink': '#EF3676', 'amber': '#FFC107', # he canviat rose_pink, anterior: #EF5350
              'teal': '#009688', 'brown': '#795548', 'light_coral': '#EDA2CF', 'olive': '#808000'} # he canviat lightcoral, abans: #F08080


color_names = [
    'black', 'grey', 'grey2', 'rose_pink', 'amber', 'p1', 'b2', 'g2', 'o2', 'g4',
    'b1', 'p2', 'b3', 'b4', 'g3', 'brown', 'p4', 'r1', 'teal', 'y', 'g1',
    'deep_orange', 'white', 'r2', 'o1', 'p3', 'light_coral', 'olive'
]



klusters = [3, 6, 9, 12, 15, 18, 21, 24, 27]


for im in image_num:
    # this needs to be opened as image!!
    path_mask = f'{path}mtbi/{animal_name}/fine_masks/{im}_mask.png'
    mask = np.array(Image.open(path_mask).convert('L'))

    n_col = mask.shape[1] // patch_size
    n_row = mask.shape[0] // patch_size

    mask_down = block_reduce(mask > 0, block_size=(patch_size, patch_size), func=np.max)

    all_new_color_mapping = matching_colors(animal_name)
    i=-1
    for k in klusters:
        i+=1
        # new_color_mapping = matching_colors(animal_name), n_clusters=k)
        new_color_mapping = all_new_color_mapping[i]
        cluster_ids = sorted(new_color_mapping.keys())
        color_list = [color_dict[new_color_mapping[cl]] for cl in cluster_ids]
        plotting(k, color_list, im, mask_down, n_row, n_col)

