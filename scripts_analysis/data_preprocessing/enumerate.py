import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import sys


def custom_sort_key(directory):
    return int(directory.split('_')[1])


path = sys.argv[1]      # path+'ISM_{animal_numbers}/sliced/{stain}'
stain = sys.argv[2]     # stain
animal_folder = path[:path.index('/sliced')]

new_folder_path = os.path.join(animal_folder, 'ordered')
if not os.path.exists(new_folder_path):
    os.makedirs(new_folder_path)

new_folder_path2 = os.path.join(new_folder_path, stain)
if not os.path.exists(new_folder_path2):
    os.makedirs(new_folder_path2)


# list of all subdirectories (glass_i folders)
subdirectories = [i for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))]

# sort the subdirectories using the custom sorting key
sorted_subdirectories = sorted(subdirectories, key=custom_sort_key)

image_number = 1
# iterating:
for folder in sorted_subdirectories:
    folder_path = os.path.join(path, folder)

    # Check if the subdirectory exists
    if os.path.exists(folder_path):
        image_files = sorted([f for f in os.listdir(folder_path)])  # if f.lower().endswith(('.png', '.jpg', '.jpeg')]))

        for image_file in image_files:
            new_image_name = f"{image_number}.png"
            image_number += 1

            # Copy the image to the destination directory with the new name
            with open(os.path.join(folder_path, image_file), 'rb') as src_file:
                with open(os.path.join(new_folder_path2, new_image_name), 'wb') as dest_file:
                    dest_file.write(src_file.read())
            print(f"Copying {image_file} to {new_image_name}")

print("Image copying complete.")

