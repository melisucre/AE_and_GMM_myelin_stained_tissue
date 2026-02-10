import os
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np
import cv2
import skimage as ski
import matplotlib.pyplot as plt


def down_sampling(path):

    # list of all subdirectories (glass_i folders)
    subdirectories = [i for i in os.listdir(path) if os.path.isdir(os.path.join(path, i))]

    # down_sampling factor
    d = 16

    # iterating
    for folder in subdirectories:
        folder_path = os.path.join(path, folder)

        # list the files in the folder
        file_list = os.listdir(folder_path)

        # filter the list to include only TIFF files with specific endings
        x0625 = [file for file in file_list if file.lower().endswith("x0.625_z0.tif")]

        if x0625:
            path_image_x0625 = os.path.join(folder_path, x0625[0])
            image_x0625 = Image.open(path_image_x0625).convert('L')

            # new width and height for downsampling
            width, height = image_x0625.size
            new_width = width // d
            new_height = height // d

            # down_sample the image
            downsampled_image = image_x0625.resize((new_width, new_height))

            # Save the downsized image in the same folder
            downsampled_image.save(os.path.join(folder_path, "downsampled_by_16.tif"))


def connected_components(path):
    # new folder for the masks
    new_folder_path = os.path.join(path, 'masks')
    if not os.path.exists(new_folder_path):
        os.makedirs(new_folder_path)

    subfolder_path = os.path.join(new_folder_path, 'some_issue')
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)

    # coordinates
    final_folder_path = os.path.join(path, 'coordinates')
    if not os.path.exists(final_folder_path):
        os.makedirs(final_folder_path)

    # list of all subdirectories (glass_i folders)
    subdirectories = [i for i in os.listdir(path) if os.path.isdir(os.path.join(path, i)) and i.startswith("glass")]

    def custom_sort_key(directory):
        return int(directory.split('_')[1])

    # sort the subdirectories using the custom sorting key
    sorted_subdirectories = sorted(subdirectories, key=custom_sort_key)

    kernel_sizes = [(1, 1), (1, 2), (2, 1), (2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3), (4, 4), (5, 5)]
    j = 1
    for folder in sorted_subdirectories:
        folder_path = os.path.join(path, folder)
        file_path = os.path.join(folder_path, 'downsampled_by_16.tif')

        object_positions = []

        # Read image and make it binary
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        y1, x1 = image.shape[:2]
        _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        # fuckkk xd, this function might work better, at least detecting a single section
        # if used in the future, check it:
        # _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_TRIANGLE)  # cv2.THRESH_OTSU)

        # labeling the pieces & counting them;
        labeled_image, count = ski.measure.label(binary_image, connectivity=2, return_num=True)
        object_features = ski.measure.regionprops(labeled_image)
        # extract area from features:
        # ******************
        # later on possible to extract centroids*****
        # object_areas = [objf["area"] for objf in object_features]

        # discard small areas:
        min_area = 80
        for object_id, objf in enumerate(object_features, start=1):
            if objf["area"] < min_area:
                labeled_image[labeled_image == objf["label"]] = 0
        binary_image = np.where(labeled_image != 0, 1, 0)
        binary_image = binary_image.astype('uint8')

        for kernel_size in kernel_sizes:
            kernel = np.ones(kernel_size, np.uint8)
            dilated_image = cv2.dilate(binary_image, kernel)
            labeled_image, count = ski.measure.label(dilated_image, connectivity=2, return_num=True)

            if count == 6:
                plt.imshow(labeled_image)
                plt.axis("off")
                plt.savefig(os.path.join(new_folder_path, f'mask_{j}.png'))
                plt.close()

                object_features = ski.measure.regionprops(labeled_image)
                for objf in object_features:
                    x, y, width, height = objf["bbox"]
                    # to have a little more margin
                    # since I had problems, I change margin bit more, it will need to be checked if used with new data
                    if (x - 2) >= 0:
                        x = x - 2
                    if (x - 2) < 0:
                        if (x - 1) >= 0:
                            x = x - 1
                    if (y - 2) >= 0:
                        y = y - 2
                    if (y - 2) < 0:
                        if (y - 1) > 0:
                            y = y - 1
                    if (width + 2) <= y1:
                        width = width + 2
                    if (width + 2) > y1:
                        if (width + 1) <= y1:
                            width = width + 1

                    if (height + 2) <= x1:
                        height = height + 2
                    if (height + 2) > x1:
                        if (height + 1) <= x1:
                            height = height + 1

                    object_positions.append((x, y, width, height))

                break  # exit the loop if count is 6

        # if count is not 6 after trying all kernel sizes, save the labeled image
        if count != 6:
            kernel = np.ones((2, 2), np.uint8)
            dilated_image = cv2.dilate(binary_image, kernel)
            labeled_image, count = ski.measure.label(dilated_image, connectivity=2, return_num=True)
            plt.imshow(labeled_image)
            plt.axis("off")
            plt.savefig(os.path.join(subfolder_path, f'mask_{j}.png'))
            plt.close()

            object_features = ski.measure.regionprops(labeled_image)
            for objf in object_features:
                x, y, width, height = objf["bbox"]
                # to have a little more margin
                if (x - 1) > 0:
                    x = x - 1
                if (y - 1) > 0:
                    y = y - 1
                if (width + 1) < y1:
                    width = width + 1
                if (height + 1) < x1:
                    height = height + 1

                object_positions.append((x, y, width, height))

        # order the boxes as they are ordered in glasses:
        sorted_positions = sorted(object_positions)
        sorted_positions_first = sorted(sorted_positions[0:3], key=lambda k: k[1])
        sorted_positions_second = sorted(sorted_positions[3:6], key=lambda k: k[1])

        # Write the object properties to a text file
        with open(os.path.join(final_folder_path, f'positions_glass_{j}.txt'), 'w') as file:
            for x, y, width, height in sorted_positions_first:
                file.write(f'{x}, {y}, {width}, {height}\n')
            for x, y, width, height in sorted_positions_second:
                file.write(f'{x}, {y}, {width}, {height}\n')

        j += 1

