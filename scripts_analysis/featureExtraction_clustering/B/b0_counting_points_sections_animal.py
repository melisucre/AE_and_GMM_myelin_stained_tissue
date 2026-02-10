import re
import numpy as np
import cv2
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import json


animal_names = ['ISM_31', 'ISM_30', 'ISM_26', 'ISM_37']


def counting_points(animal_name):
    main_path = '.../AE/data/mtbi/'
    stain = 'mye'

    animal_path = f'{main_path}/{animal_name}'
    images2correct_folder = f'{animal_path}/renormalized/{stain}'

    # Get list of .png files that are named as numbers and extract their numeric part
    numbers = [int(re.search(r'(\d+)\.png$', f).group(1)) for f in os.listdir(images2correct_folder) if re.match(r'^\d+\.png$', f)]

    # Sort the numbers and save them in the dictionary
    numbers = sorted(numbers)
    print(f"Let's count in animal '{animal_name}'")
    window_sizes = [64, 128, 256]

    results = {}

    for n in numbers:

        # original_image_path = f'{animal_path}/ordered/{stain}/{n}.png'
        mask_path = f'{animal_path}/fine_masks/{n}_mask.png'
        mask = Image.open(mask_path).convert('L')

        image_path = f'{images2correct_folder}/{n}.png'
        image = Image.open(image_path).convert('L')

        mask_np = np.array(mask)
        image_np = np.array(image)
        # Set pixels where the result is 0 to 255

        together = np.copy(image_np)
        together[mask_np == 0] = 255

        # result_image = Image.fromarray(together.astype(np.uint8))
        results[n] = {}
        # as_image = Image.fromarray(together)

        # lets run every window size:
        for window in window_sizes:
            count = 0
            coordinates = []

            # iterate over the image with the window size:
            height, width = together.shape
            for i in range(0, height - window + 1, window):
                for j in range(0, width - window + 1, window):
                    tile = together[i:i+window, j:j+window]

                    # let's use that tile if the number of non-255 pixels if more than half of the area of the tile:
                    non_255_count = np.sum(tile != 255)
                    if non_255_count > (window*window/2):
                        count += 1
                        coordinates.append((i, j))

            results[n][window] = {'count': count, 'coordinates': coordinates}


    output_path = f'{animal_path}/counting_patches.json'
    with open(output_path, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print(f"Results saved to {output_path}")

for animal_name in animal_names:
    counting_points(animal_name)