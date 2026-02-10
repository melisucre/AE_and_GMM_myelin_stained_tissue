import os
import re
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from multiprocessing import Pool
import sys


path_glasses = sys.argv[1]
path_cut = sys.argv[2]

os.makedirs(path_cut, exist_ok=True)

conv_factor = 16 * 64

path_coord = path_glasses + '/coordinates'

def crop_and_save(values):
    try:
        x0, y0, x1, y1 = map(int, values[0].strip().split(', '))
        n = values[1]

        X0 = x0 * conv_factor
        Y0 = y0 * conv_factor
        X1 = x1 * conv_factor
        Y1 = y1 * conv_factor

        cropped_image = image.crop((Y0, X0, Y1, X1))

        output_file = new_folder + '/' + str(n) + ".png"
        cropped_image.save(output_file, format="PNG")

    except Exception as e:
        print(f"An error occurred for glass {values[1]} (when cropping): {e}")


# iterate through the text files in the directory
for filename in os.listdir(path_coord):
    if filename.endswith(".txt"):
        coordinates = os.path.join(path_coord, filename)
        # extract the number from the file name
        match = re.match(r'positions_glass_(\d+)\.txt', filename)

        if match:
            number = int(match.group(1))
            new_folder = path_cut + '/glass_' + str(number)
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)

                path_image = path_glasses + '/glass_' + str(number)
                file_list = os.listdir(path_image)

                for file in file_list:
                    if file.endswith("x40_z0.tif"):
                        desired_file_path = os.path.join(path_image, file)
                        break  # exit the loop once the file is found

                try:
                    image = Image.open(desired_file_path).convert('L')
                except Exception as e:
                    print(f"An error occurred for glass {number} (opening the image): {e}")

                n = 1
                pairs = []
                with open(coordinates, 'r') as file:
                    for line in file:
                        pairs.append([line, n])
                        n += 1

                print('pooling\n')
                pool = Pool(processes=6)
                pool.map(crop_and_save, pairs)

                pool.close()
                pool.join()
