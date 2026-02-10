import matplotlib.pyplot as plt
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from multiprocessing import Pool
import os
import sys


folder_input = sys.argv[1]
folder_output = sys.argv[2]

def dali(values):
    image_folder = values[0]
    folder_num = values[1]
    images = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    images.sort()

    # Create a subplot for each image
    plt.figure(figsize=(10, 8))
    for i, img_name in enumerate(images, start=1):
        img_path = os.path.join(image_folder, img_name)
        img = Image.open(img_path)
        plt.subplot(2, 3, i)
        plt.imshow(img, cmap='jet', vmin=0, vmax=255)
        plt.title(img_name)
        plt.axis('off')

    # Save the plotted images in the 'prova' folder
    output_path = os.path.join(folder_output, f'{folder_num}.png')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Plotted images from '{folder_num}'")  # and saved as '{output_path}'")

if not os.path.exists(folder_output):
    os.makedirs(folder_output)

# List all directories (folders) starting with 'glass_str'
folders = [f for f in os.listdir(folder_input) if os.path.isdir(os.path.join(folder_input, f)) and f.startswith('glass_')]
# folders.sort()

images_folder = []
for folder in folders:
    images_folder.append([os.path.join(folder_input, folder), folder])

print('pooling\n')
pool = Pool(processes=6)
pool.map(dali, images_folder)

pool.close()
pool.join()

