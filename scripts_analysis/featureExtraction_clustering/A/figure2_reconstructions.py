import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import cv2
import random
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
# import matplotlib.cm as cm
# import matplotlib.colors as mcolors


ae_names = ['AE_model128']

stain = 'mye'
animal_name = 'ISM_31' 
slice = 46  # slices = ['28', '46', '57']


base_dir = '.../AE/data'
base_save = '.../AE/reconstructions/'
os.makedirs(base_save, exist_ok=True)

image_path = f'{base_dir}/mtbi/{animal_name}/renormalized/{stain}/{slice}.png'
original_image = np.array(Image.open(image_path).convert('L'))

# mask
mask_path = f'{base_dir}/mtbi/{animal_name}/fine_masks/{stain}/{slice}_mask.png'
mask = np.array(Image.open(mask_path).convert('L'))
# apply the mask to the original:
original_image[mask == 0] = 255

# now let's save the original:
name_details = 'original'
image = original_image

# # first plot needs resize because it's too large:
# tros = image[:, 0:32000]
# tros = Image.fromarray(tros)
# new_size = (tros.size[0]//2, tros.size[1]//2)
# resized_image = tros.resize(new_size, Image.LANCZOS)
# path_save = f'{base_save}/{animal_name}-{slice}_{name_details}_downsampledby2.png'
# resized_image.save(path_save)

# # second plot:
# tros = image[:, 0:32000]
# tros_tros = tros[5000:18000, 20000:28000]
# tros_tros = Image.fromarray(tros_tros)
# path_save = f'{base_save}/{animal_name}-{slice}_{name_details}_tros1_mesgran.png'
# tros_tros.save(path_save)
#
# # third plot:
# tros = image[:, 0:32000]
# tros_tros = tros[5000:15000, 23000:28000]
# tros_troset = tros_tros[3500:6500, 2000:3500]
# tros_troset = tros_tros[3500:6500, 1500:4000]
# path_save = f'{base_save}/{animal_name}-{slice}_{name_details}_tros2.png'
# tros_troset = Image.fromarray(tros_troset)
# tros_troset.save(path_save)
#
# # small pieces:
# tros_troset1 = image[8960:9472, 25600:26112]
# tros_troset2 = image[10496:11008, 25600:26112]
#
# path_save = f'{base_save}/{animal_name}-{slice}_{name_details}_tros2a.png'
# tros_troset1 = Image.fromarray(tros_troset1)
# tros_troset1.save(path_save)
# path_save = f'{base_save}/{animal_name}-{slice}_{name_details}_tros2b.png'
# tros_troset2 = Image.fromarray(tros_troset2)
# tros_troset2.save(path_save)

for ae_name in ae_names:

    reconstr_path = f'{base_dir}/AE_usage/{ae_name}/{animal_name}/RC/{stain}/reconstr_img{slice}.png'
    reconstr_image = np.array(Image.open(reconstr_path).convert('L'))
    reconstr_image = reconstr_image.astype(np.uint8)

    # apply the mask to the reconstructed:
    reconstr_image[mask == 0] = 255

    image = reconstr_image
    name_details = f'reconstr_{ae_name}'

# ---------------------------

    original_a = original_image[8960:9472, 25600:26112].copy()
    original_b = original_image[10496:11008, 25600:26112].copy()
    reconstr_a = reconstr_image[8960:9472, 25600:26112].copy()
    reconstr_b = reconstr_image[10496:11008, 25600:26112].copy()

    plt.imshow(reconstr_a, cmap='grey', vmin=0, vmax=255)
    plt.axis('off')
    path = f'{base_save}/{animal_name}-{slice}_{name_details}_tros_a.png'
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)

    plt.imshow(reconstr_b, cmap='grey', vmin=0, vmax=255)
    plt.axis('off')
    path = f'{base_save}/{animal_name}-{slice}_{name_details}_tros_b.png'
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)


    original_a = original_a.astype(np.float32)
    original_b = original_b.astype(np.float32)
    reconstr_a = reconstr_a.astype(np.float32)
    reconstr_b = reconstr_b.astype(np.float32)



    error_a = (original_a - reconstr_a) ** 2
    error_b = (original_b - reconstr_b) ** 2

    maxim_error_a = np.max(error_a)
    error_a_norm = error_a / maxim_error_a
    maxim_error_b = np.max(error_b)
    error_b_norm = error_b / maxim_error_b

    mse_a = np.mean((original_a - reconstr_a) ** 2)
    mse_b = np.mean((original_b - reconstr_b) ** 2)


    name_details = f'reconstr_{ae_name}'
    plt.imshow(error_a_norm, cmap='magma', vmin=0, vmax=1)
    plt.axis('off')
    plt.text(
        error_a_norm.shape[1] // 2,  # Center X position
        error_a_norm.shape[0] - 10,  # Near the bottom Y position
        f"{mse_a:.2f}",
        color='white',
        fontsize=30,
        ha='center',  # Center horizontally
        va='bottom',  # Align text at the bottom
    )
    path = f'{base_save}/{animal_name}-{slice}_{name_details}_tros_a_norm_mse.png'
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
    plt.close()

    name_details = f'reconstr_{ae_name}'
    plt.imshow(error_b_norm, cmap='magma', vmin=0, vmax=1)
    plt.axis('off')
    plt.text(
        error_b_norm.shape[1] // 2,  # Center X position
        error_b_norm.shape[0] - 10,  # Near the bottom Y position
        f"{mse_b:.2f}",
        color='white',
        fontsize=30,
        ha='center',  # Center horizontally
        va='bottom',  # Align text at the bottom
    )
    path = f'{base_save}/{animal_name}-{slice}_{name_details}_tros_b_norm_mse.png'
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=300, transparent=True)
    plt.close()

# # ---------
#
# # lets compute the errors of the whole half section:
# # for ae_name in ae_names:
# #
# #     reconstr_path = f'{base_dir}/AE_usage/{ae_name}/{animal_name}/RC/{stain}/reconstr_img{slice}.png'
# #     reconstr_image = np.array(Image.open(reconstr_path).convert('L'))
# #     reconstr_image = reconstr_image.astype(np.uint8)
# #
# #     name_details = f'reconstr_{ae_name}'
# #     # apply the mask to the reconstructed:
# #
# #     reconstr_image[mask == 0] = 255
# #     original_image[mask == 0] = 255
# #
# #     orig = original_image[:, 0:32000].copy()
# #     recon = reconstr_image[:, 0:32000].copy()
# #     mask_half = mask[:, 0:32000].copy()
# #
# #     orig = orig.astype(np.float32)
# #     recon = recon.astype(np.float32)
# #
# #     error = (orig - recon) ** 2
# #
# #     maxim_error = np.max(error)
# #     error_norm = error / maxim_error
# #
# #     mse = np.mean((orig - recon) ** 2)
# #     mse_masked = np.mean((orig[mask_half != 0] - recon[mask_half != 0]) ** 2)
# #
# #     resized_image = cv2.resize(error_norm, (error_norm.shape[1] // 2, error_norm.shape[0] // 2), interpolation=cv2.INTER_LINEAR)
# #
# #     plt.imshow(resized_image, cmap='magma', vmin=0, vmax=0.2)
# #     plt.axis('off')
# #     plt.text(
# #         resized_image.shape[1] // 2,  # Center X position
# #         resized_image.shape[0] - 10,  # Near the bottom Y position
# #         f"{mse_masked:.2f}",
# #         color='white',
# #         fontsize=30,
# #         ha='center',  # Center horizontally
# #         va='bottom',  # Align text at the bottom
# #     )
# #     path = f'{base_save}/{animal_name}-{slice}_{name_details}_half_slice_norm_mse_02.png'
# #     plt.savefig(path, bbox_inches='tight', dpi=300, transparent=True)
# #     plt.close()
#
#
