import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import cv2
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import glob

path_data = '...AE/data/mtbi/'

animal_names = ['ISM_37']
slices = ['all']
stain = 'mye'


def mask(array, d=64):
    """Apply a mask, take the biggest component and fill it"""
    # Ensure input is grayscale
    if len(array.shape) > 2:
        array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    height, width = array.shape
    new_width = width // d
    new_height = height // d

    # Convert to PIL Image for resizing
    pil_image = Image.fromarray(array)
    dws_image = pil_image.resize((new_width, new_height))

    # Convert back to NumPy array
    array_image = np.array(dws_image)

    # Create the binary mask using thresholding
    _, binary_image = cv2.threshold(array_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)

    # Resize mask back to original dimensions
    resized_mask = cv2.resize(binary_image, (width, height), interpolation=cv2.INTER_NEAREST)

    # Remove small connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(resized_mask)

    areas = stats[:, 4]  # Area column
    neg_indices = np.where(areas < 0)[0]

    if len(neg_indices) > 0:
        # Pick the first label with a negative area
        selected_label = neg_indices[0]
    else:
        # Pick the largest label (excluding background label 0)
        selected_label = 1 + np.argmax(areas[1:])

    component_mask = (labels == selected_label).astype(np.uint8) * 255

    # Find contours and hierarchy
    contours, hierarchy = cv2.findContours(component_mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    # Create an empty image to draw the filled component
    filled_mask = np.zeros_like(component_mask)

    # Draw filled contours
    for i in range(len(contours)):
        # Only fill external and hole contours if hierarchy is available
        if hierarchy[0][i][3] == -1:  # Outer contour
            cv2.drawContours(filled_mask, contours, i, 255, thickness=cv2.FILLED)
            # Also fill any holes (children)
            child_idx = hierarchy[0][i][2]
            while child_idx != -1:
                cv2.drawContours(filled_mask, contours, child_idx, 255, thickness=cv2.FILLED)
                child_idx = hierarchy[0][child_idx][0]

    return filled_mask


for animal_name in animal_names:

    path_save = f'{path_data}{animal_name}/fine_masks/{stain}/'
    os.makedirs(path_save, exist_ok=True)

    for slice in slices:

        if slice == 'all':
            path_normalized = f'{path_data}{animal_name}/normalized/{stain}*'
            all_imgs = sorted(glob.glob(os.path.join(path_normalized, "*.png")))
            slice_names = [os.path.splitext(os.path.basename(imgs))[0] for imgs in all_imgs]

            path_originals = f'{path_data}{animal_name}/ordered/{stain}'
            for number in slice_names:
                section = f'{path_originals}/{number}.png'
                im = np.array(Image.open(section).convert('L'))

                mm = mask(im)
                mask_img = Image.fromarray(mm)
                mask_img.save(os.path.join(path_save, f"{number}_mask.png"))


        else:
            im_path = f'{path_data}{animal_name}/ordered/{stain}/{slice}.png'
            im = np.array(Image.open(im_path).convert('L'))

            path_mask = f'{path_save}/{slice}_mask.png'

            mm = mask(im)
            Image.fromarray(mm).save(path_mask)



