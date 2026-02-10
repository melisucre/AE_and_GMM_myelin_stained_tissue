import os
import re
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import glob
import mask_def as mask
import numpy as np
import cv2
from skimage.exposure import match_histograms


def renorm(path, animal_name, animal_ref, stain, slice_ref):

    print(animal_name)
    animal_path = path + animal_name + '/normalized/'

    # list the folders in animal_path, then select the folder with the sections
    subfolders = [os.path.join(animal_path, d) for d in os.listdir(animal_path) if os.path.isdir(os.path.join(animal_path, d))]
    sections2normalize_path = [d for d in subfolders if os.path.basename(d).startswith(stain)]

    # where to store the normalized images
    save_path = path + animal_name + '/renormalized/'
    os.makedirs(save_path, exist_ok=True)

    save_path = save_path + str(stain) + '/'
    os.makedirs(save_path, exist_ok=True)
    print('folder created')

    # list the images in that folder
    pattern = os.path.join(animal_path, f"{stain}*-norm{animal_ref}.png")
    slice_ref_path = glob.glob(pattern)

    # pattern = sorted(glob.glob(os.path.join(sections2normalize_path[0], "*.png")))
    # slice_names = [os.path.splitext(os.path.basename(imgs))[0] for imgs in all_imgs]

    # read and mask reference
    image_reference = Image.open(slice_ref_path[0]).convert('L')
    image_mask_reference = mask.mask(slice_ref_path[0])
    # image_mask_reference = mask.mask(image_reference)
    image_reference = np.array(image_reference, dtype=np.float16)
    image_reference[image_mask_reference == 0] = np.nan
    del image_mask_reference

    print('image_ref_loaded')
    # let's save the ref_image that will not be normalized:
    to_save = save_path + str(slice_ref) + '.png'
    save_ref_image = image_reference.copy()
    save_ref_image[np.isnan(save_ref_image)] = 255
    save_ref_image = (save_ref_image).astype(np.uint8)
    save_ref_image = Image.fromarray(save_ref_image).convert('L')
    save_ref_image.save(to_save)
    del save_ref_image

    # let's check what are the number sections, I need the min and max
    image_numbers = os.listdir(sections2normalize_path[0])
    numbers = []
    for i in image_numbers:
        if i.endswith('.png'):
            match = re.match(r'(\d+)\.png', i)
            if match:
                numbers.append(int(match.group(1)))
    if numbers:
        min_num = min(numbers)
        max_num = max(numbers)
    print(min_num)
    print(max_num)

    # now let's normalize using that:
    reference = image_reference.copy()
    n = slice_ref

    # first the lower part:
    while n > min_num:
        print('inside first loop')
        n -= 1
        to_convert = sections2normalize_path[0] + '/' + str(n) + '.png'
        to_save = save_path + str(n) + '.png'

        n_mask = mask.mask(to_convert)
        to_convert = Image.open(to_convert).convert('L')

        to_convert = np.array(to_convert, dtype=np.float16)
        to_convert[n_mask == 0] = np.nan
        del n_mask

        match = match_histograms(to_convert, reference)
        reference = match

        match[np.isnan(match)] = 255
        match = Image.fromarray(match).convert('L')  # aqui en principi ja l'estic fent de 8bits
        match.save(to_save)
        print("convertida\n")

    # second, upper part:

    reference = image_reference.copy()
    n = slice_ref

    while n < max_num:
        print('inside second loop')
        n += 1
        to_convert = sections2normalize_path[0] + '/' + str(n) + '.png'
        to_save = save_path + str(n) + '.png'

        n_mask = mask.mask(to_convert)
        to_convert = Image.open(to_convert).convert('L')

        to_convert = np.array(to_convert, dtype=np.float16)
        to_convert[n_mask == 0] = np.nan
        del n_mask

        match = match_histograms(to_convert, reference)
        reference = match

        match[np.isnan(match)] = 255
        match = Image.fromarray(match).convert('L')  # aqui en principi ja l'estic fent de 8bits
        match.save(to_save)







