import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import cv2
import numpy as np
from skimage.exposure import match_histograms
import time
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


animal_data_references = [
    {'animal': 'ISM_24', 'stain_ref': [('mye', 44), ('nis', 39)]},
    {'animal': 'ISM_26', 'stain_ref': [('mye', 42), ('nis', 43)]},
    {'animal': 'ISM_27', 'stain_ref': [('mye', 44), ('nis', 44)]},
    {'animal': 'ISM_29', 'stain_ref': [('mye', 44), ('nis', 45)]},
    {'animal': 'ISM_30', 'stain_ref': [('mye', 44), ('nis', 45)]},
    {'animal': 'ISM_31', 'stain_ref': [('mye', 43), ('nis', 44)]},
    {'animal': 'ISM_32', 'stain_ref': [('mye', 38), ('nis', 38)]},
    {'animal': 'ISM_33', 'stain_ref': [('mye', 45), ('nis', 45)]},
    {'animal': 'ISM_34', 'stain_ref': [('mye', 46), ('nis', 47)]},
    {'animal': 'ISM_35', 'stain_ref': [('mye', 43), ('nis', 44)]},
    {'animal': 'ISM_36', 'stain_ref': [('mye', 45), ('nis', 45)]},
    {'animal': 'ISM_37', 'stain_ref': [('mye', 40), ('nis', 40)]},
    {'animal': 'ISM_38', 'stain_ref': [('mye', 37), ('nis', 38)]}]

animal_dictionary = {}

for entry in animal_data_references:
    animal_name = entry['animal']
    stain_and_ref = entry['stain_ref']

    animal_dictionary[animal_name] = {}

    for staining, reference in stain_and_ref:
        animal_dictionary[animal_name][staining] = reference

# now I can call things like this: animal_dictionary['ISM_24']['mye']

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

# ----------------------------------------------------------------------------------------------------------------------

path = '.../AE/data/mtbi/'
animal_ref = 'ISM_27'
stain = 'mye'
path_animal_ref = path + animal_ref + '/normalized'

directories = os.listdir(path_animal_ref)

for directory in directories:
    if directory.startswith(stain):
        path_animal_ref = os.path.join(path_animal_ref, directory)
        break

image_reference_path = path_animal_ref + '/' + str(animal_dictionary[animal_ref][stain]) + '.png'

image_reference = Image.open(image_reference_path).convert('L')
image_mask_reference = mask(image_reference)
image_reference = np.array(image_reference, dtype=np.float16)
image_reference[image_mask_reference == 0] = np.nan
del image_mask_reference

# ----------------------------------------------------------------------------------------------------------------------
# let's normalize per animals:
for entry in animal_data_references:
    animal_name = entry['animal']

    if animal_name != animal_ref:

        path_animal = path + animal_name + '/normalized'

        directories = os.listdir(path_animal)
        for directory in directories:
            if directory.startswith(stain):
                path_animal = os.path.join(path_animal, directory)
                break

        image_path = path_animal + '/' + str(animal_dictionary[animal_name][stain]) + '.png'

        path_save = path + animal_name + '/normalized/' + directory + '_' + str(animal_dictionary[animal_name][stain]) + '-norm' + animal_ref + '.png'
        # we read the image, create the mask, convolve them and delete mask
        image2convert = Image.open(image_path).convert('L')
        # print the image to visualize it
        plt.imshow(image2convert, cmap='jet', vmin=0, vmax=255)
        plt.axis('off')
        output_path = path + animal_name + '/normalized/' + directory + '_' + str(animal_dictionary[animal_name][stain]) + '-vis.png'
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

        # create the mask, convolve them and delete mask
        image_mask_2convert = mask(image2convert)
        image2convert = np.array(image2convert, dtype=np.float16)
        image2convert[image_mask_2convert == 0] = np.nan
        del image_mask_2convert


        # now histo matching:
        start_time = time.time()
        match = match_histograms(image2convert, image_reference)
        match[np.isnan(match)] = 255
        match = Image.fromarray(match).convert('L') # aqui en principi ja l'estic fent de 8bits
        match.save(path_save)
        print("convertida\n")
        end_time = time.time()
        execution_time = end_time - start_time

        print(f"The code took {execution_time:.2f} seconds to run.")

        # print the image to visualize it
        plt.imshow(match, cmap='jet', vmin=0, vmax=255)
        plt.axis('off')
        output_path = path + animal_name + '/normalized/' + directory + '_' + str(animal_dictionary[animal_name][stain]) + '-norm' + animal_ref + '-vis.png'
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()


