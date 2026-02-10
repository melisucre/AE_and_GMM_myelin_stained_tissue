import json
import os
import re
import sys
import torch
import importlib
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import joblib
import pickle


main_path = '.../AE/data/mtbi'
stain = 'mye'
animal_names = ['ISM_26', 'ISM_30', 'ISM_31', 'ISM_37']
num_patches = 100000
LS = 256
windows = [128, 256]
pca_names = ['w128pc256', 'w256pc256']

def passing_patches(animal_name, num_patches, LS, window, pca_name):

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

        data[image_num]['coordinates'].append(coord)
    # now coordinates can be accessed like data[window_size][slice]; e.g. data['32'][33] gives the coord to be used in slice 33
    # get the list of .png files that are named as numbers and extract their numeric part
    animal_slices = f'{animal_path}/renormalized/{stain}'
    numbers = [int(re.search(r'(\d+)\.png$', f).group(1)) for f in os.listdir(animal_slices)
               if re.match(r'^\d+\.png$', f)]
    numbers.sort()

    all_tiles = []
    for num in numbers:
        path_section = f'{animal_slices}/{num}.png'
        image = Image.open(path_section).convert('L')
        image = np.array(image) / 255.0
        coords_image = data[num]['coordinates']
        tiles = [
            image[c[0]:c[0] + window, c[1]:c[1] + window].flatten()
            for c in coords_image]
        all_tiles.extend(tiles)

    all_tiles = np.array(all_tiles, dtype=np.float32)
    print(f"Extracted {all_tiles.shape[0]} patches of shape {all_tiles.shape[1]}")

    ## ------

    model_path = f'/.../PCA/trained_models/PCA_model{window}.pkl'

    with open(model_path, 'rb') as f:
        pca = pickle.load(f)
    print(f"PCA model loaded from: {model_path}")
    print(f"Components: {pca.n_components_}")
        # return ipca

    base_dir = f'...AE/data/massivePCA/{pca_name}/data/'

    dirs2create = [
        base_dir,
        f'{base_dir}{animal_name}/',
        f'{base_dir}{animal_name}/gmm/',
    ]

    for directory in dirs2create:
        os.makedirs(directory, exist_ok=True)

    to_save = f'{base_dir}{animal_name}/gmm/'

        # Apply pre-trained PCA
    print("Applying pre-trained PCA...")
    pca_features = pca.transform(all_tiles)
    print(f"PCA transformed to shape: {pca_features.shape}")

    # Save the transformed features
    save_filename = f'{num_patches}_PCA_gmm.npy'
    np.save(f'{to_save}{save_filename}', pca_features)


for animal_name in animal_names:
    for i in range(0, 2):
        passing_patches(animal_name, num_patches, LS, windows[i], pca_names[i])

