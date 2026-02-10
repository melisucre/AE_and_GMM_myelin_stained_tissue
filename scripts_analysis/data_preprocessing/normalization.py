import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2
import numpy as np
from skimage.exposure import match_histograms
import time


def mask(image, d=1024):

    # image = Image.open(path_image).convert('L')
    width, height = image.size
    new_width = width // d
    new_height = height // d

    # down_sample the image
    dws_image = image.resize((new_width, new_height))
    array_image = np.array(dws_image)

    # create the mask
    _, binary_image = cv2.threshold(array_image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)# cv2.THRESH_OTSU)
    # binary_image = binary_image.astype('uint8')

    # back to original size
    resized_mask = cv2.resize(binary_image, (width, height))

    return resized_mask

# ---------
start_time = time.time()

stain = 'mye'
animal = '37'
path = '.../AE/data/mtbi/ISM_'+animal+'/ordered/'+stain+'/'

save_path = '.../AE/data/mtbi/ISM_'+animal+'/normalized/'
os.makedirs(save_path, exist_ok=True)
print('carpeta creada\n')
# mye:    first_slice = 15    last_slice = 58      reference = 41
Ref = 41
first_slice = 15
last_slice = 58
save_path_stain = save_path + stain + str(Ref) + '/'
os.makedirs(save_path_stain, exist_ok=True)

reference_path = path + str(Ref) + '.png'

reference = Image.open(reference_path).convert('L')
reference_mask = mask(reference)
reference = np.array(reference, dtype=np.float16)
reference[reference_mask == 0] = np.nan


n = Ref
while n > first_slice:
    n -= 1
    to_convert = path + str(n) + '.png'
    to_save = save_path_stain + str(n) + '.png'

    to_convert = Image.open(to_convert).convert('L')
    n_mask = mask(to_convert)

    to_convert = np.array(to_convert, dtype=np.float16)
    to_convert[n_mask == 0] = np.nan

    match = match_histograms(to_convert, reference)

    reference = match

    match[np.isnan(match)] = 255
    match = Image.fromarray(match).convert('L') # in this way it's already 8bits
    match.save(to_save)
    print("convertida\n")


n = Ref
reference = Image.open(reference_path).convert('L')
reference_mask = mask(reference)
reference = np.array(reference, dtype=np.float16)
reference[reference_mask == 0] = np.nan

while n < last_slice:
    n += 1
    to_convert = path + str(n) + '.png'
    to_save = save_path_stain + str(n) + '.png'

    to_convert = Image.open(to_convert).convert('L')
    n_mask = mask(to_convert)

    to_convert = np.array(to_convert, dtype=np.float16)
    to_convert[n_mask == 0] = np.nan

    match = match_histograms(to_convert, reference)

    reference = match

    match[np.isnan(match)] = 255
    match = Image.fromarray(match).convert('L') # in this way it's already 8bits
    match.save(to_save)


end_time = time.time()
execution_time = end_time - start_time

print(f"The code took {execution_time:.2f} seconds to run.")

