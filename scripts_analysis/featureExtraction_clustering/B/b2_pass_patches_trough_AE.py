import json
import os
import re
import sys
import torch
import importlib
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


main_path = '.../AE/data/mtbi'
stain = 'mye'
animal_names = ['ISM_26', 'ISM_37', 'ISM_30', 'ISM_31']
num_patches = 100000
LS = 256
windows = [128, 256]
ae_names = ['AE_model128', 'AE_model256']

def passing_patches(animal_name, num_patches, LS, window, ae_name):

    animal_path = f'{main_path}/{animal_name}'
    file_path = f'{animal_path}/w_{window}_{num_patches}_selected_patches.json'
    with open(file_path, 'r') as json_file:
        selected_patches = json.load(json_file)

    data = {}

    # now I'm in a concrete window
    for patch in selected_patches:
        image_num = patch['image']
        coord = patch['coordinates']
        if image_num not in data:
            data[image_num] = {'coordinates': []}
            # append the coordinates to the image entry for this window size
        data[image_num]['coordinates'].append(coord)
        # now coordinates can be accessed like data[window_size][slice]; e.g. data['32'][33] gives the coord to be used in slice 33
        # get the list of .png files that are named as numbers and extract their numeric part
    animal_slices = f'{animal_path}/renormalized/{stain}'
    numbers = [int(re.search(r'(\d+)\.png$', f).group(1)) for f in os.listdir(animal_slices)
               if re.match(r'^\d+\.png$', f)]

        ## ------

    module_path = f'.../AE/scripts/architectures/AE_model{window}/'
    sys.path.insert(0, module_path)
    ae_architecture = f'AE_model{window}'
    ae = importlib.import_module(ae_architecture)

    ae_pth = ae_name+'.pth'

    # now let's create an instance of the ae architecture and load the trained model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AE = ae.instance()
    path_to_saved_model = '...AE/trained_models/'+ae_pth
    AE.load_state_dict(torch.load(path_to_saved_model, map_location=device))
    AE.eval()  # set the model to evaluation mode (important for batch normalization and dropout)

    base_dir = f'.../AE/data/AE_usage/{ae_name}/'
    dirs2create = [
        base_dir,
        f'{base_dir}{animal_name}/',
        f'{base_dir}{animal_name}/gmm/',
    ]

    for directory in dirs2create:
        os.makedirs(directory, exist_ok=True)

    to_save = f'{base_dir}{animal_name}/gmm/'

    d = data # [str(window)]

    LatSpace = []
    for num in numbers:
        path_section = f'{animal_slices}/{num}.png'
        image = Image.open(path_section).convert('L')
        image = np.array(image)/255.0

            # n_col = image.shape[1]
            # n_row = image.shape[0]

        coords_image = d[num]['coordinates']
            # now easy to access the coordinates, just by coords_image[0] and so on

        max = len(coords_image)
        tiles = [image[coords_image[i][0]:coords_image[i][0]+window, coords_image[i][1]:coords_image[i][1]+window] for i in range(max)]
            # to change the format as [max, 32, 32]:
        tiles = np.array(tiles)
        tiles = torch.from_numpy(tiles).unsqueeze(1)  # this function keep the float32 dtype & add a channel dimension

        with torch.no_grad():
            latent = AE.encoder(tiles)
            # Convert tensors to numpy arrays:
            latent = latent.numpy()

            LatSpace.append(latent)

        del tiles, latent

    LatSpace = np.concatenate(LatSpace, axis=0)
    np.save(f'{to_save}/{num_patches}_LS_gmm.npy', LatSpace)


for animal_name in animal_names:
    for i in range(0, 2):
        passing_patches(animal_name, num_patches, LS, windows[i], ae_names[i])

