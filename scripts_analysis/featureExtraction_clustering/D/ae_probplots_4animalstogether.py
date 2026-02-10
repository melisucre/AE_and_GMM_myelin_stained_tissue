import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2
from matplotlib.colors import ListedColormap
import os
# from zz_color_dictionary_2steps_4animals import all_matching_colors
from colorDictionary_4animals import matching_colors
from skimage.measure import block_reduce


stain = 'mye'
cov_type = 'spherical'

animal_names = ['ISM_26', 'ISM_30', 'ISM_31', 'ISM_37']
image_nums = [['27', '45', '57'], ['28', '46', '57'], ['28', '46', '57'], ['23', '41', '53']]


ae_names = ['AE_model128', 'AE_model256']
path = '/.../AE/data/'

patch_sizes = [128, 256]

# plot:
def plotting(n, colors, im, mask_down, n_row, n_col, path_save):

    # path to data prepared for clustering
    path_labels = f'/.../AE/data/AE_usage/{ae_name}/4animals/gmm/{cov_type}/{animal_name}_k{n}_num{im}_clustered.npy'

    # loading both image and data with labels
    data = np.load(path_labels)
    data = np.reshape(data, (n_row, n_col))

    # paleta = ListedColormap(sns.color_palette(colors))
    paleta = ListedColormap(colors)
    masked_data = np.ma.masked_where(~mask_down, data)

    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    ax.set_axis_off()
    img = ax.imshow(masked_data, vmin=0, vmax=len(paleta.colors), cmap=paleta)

    where = f'{path_save}/{animal_name}_k{n}_{im}.png'
    plt.savefig(where, bbox_inches='tight', pad_inches=0, dpi=800, transparent=True)
    plt.close()

# --------------------------------------------


color_dict = {'black': '#000000', 'grey': '#D3D3D3', 'grey2': '#808080', 'white': '#FFFFFF', 'p4': '#3C194E',
              'p3': '#6A2C89', 'p2': '#AD48E0', 'p1': '#BB80DF',
              'r2': '#73001F', 'r1': '#CA0036', 'o2': '#ED6529', 'o1': '#FF9439', 'y': '#FFD548', 'b4': '#221B6E',
              'b3': '#4638E1', 'b2': '#4F9EFF', 'b1': '#46C7E1',
              'g4': '#225F36', 'g3': '#379958', 'g2': '#49CC75', 'g1': '#B9E884', 'deep_orange': '#C2470E',
              'rose_pink': '#EF3676', 'amber': '#FFC107', 
              'teal': '#009688', 'brown': '#795548', 'light_coral': '#EDA2CF', 'olive': '#808000'} 

color_names = [
    'black', 'grey', 'grey2', 'rose_pink', 'amber', 'p1', 'b2', 'g2', 'o2', 'g4',
    'b1', 'p2', 'b3', 'b4', 'g3', 'brown', 'p4', 'r1', 'teal', 'y', 'g1',
    'deep_orange', 'white', 'r2', 'o1', 'p3', 'light_coral', 'olive'
]

klusters = [3 , 9, 21]

p=-1
for ae_name in ae_names:
    p+=1
    s=-1
    patch_size = patch_sizes[p]
    for animal_name in animal_names:
        s+=1
        path_clusters = f'/.../AE/data/AE_usage/{ae_name}/4animals/gmm/{cov_type}_plots'
        os.makedirs(path_clusters, exist_ok=True)
        path_save = path_clusters

        path_masks = f'/.../AE/data/mtbi/{animal_name}/fine_masks/mye/'
        path_images = f'/.../AE/data/mtbi/{animal_name}/renormalized/mye/'

        image_num = image_nums[s]

        for im in image_num:
            # this needs to be opened as image!!
            path_mask = f'{path_masks}/{im}_mask.png'
            mask = np.array(Image.open(path_mask).convert('L'))

            n_col = mask.shape[1] // patch_size
            n_row = mask.shape[0] // patch_size

            mask_down = block_reduce(mask > 0, block_size=(patch_size, patch_size), func=np.max)

            all_color_mapping = all_matching_colors(animal_name, ae_name)
            # i = -1
            for k in klusters:
                i = 1
                color_mapping = all_color_mapping[i]
                print(f'{animal_name}\n{color_mapping}\n\n')
                cluster_ids = sorted(color_mapping.keys())
                color_list = [color_dict[color_mapping[cl]] for cl in cluster_ids]
                plotting(k, color_list, im, mask_down, n_row, n_col, path_save)

