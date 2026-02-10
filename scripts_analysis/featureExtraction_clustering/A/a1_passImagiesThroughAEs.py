import os
import sys
import glob
import torch
import importlib
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def getting_LS_and_reconstruction(ae_name, LS, window, animal_name, stain, slices):

    # add the architecture path:
    # module_path = f'.../AE/architectures/AE_model{window}/'
    # sys.path.insert(0, module_path)
    # ae_architecture = f'AE_w{window}_LS{LS}_batchnorm'
    
    ae_architecture =  f'.../AE/architectures/AE_model{window}'

    # import ae_architecture as ae
    ae = importlib.import_module(ae_architecture)
    ae_pth = ae_name+'.pth'


    path_animal = f'.../AE/data/mtbi/{animal_name}/renormalized/{stain}/'

    if slices[0] == 'all':
        # read all images:
        paths = glob.glob(os.path.join(path_animal, '*.png'))
        paths.sort(key=lambda f: int(os.path.basename(f).replace('.png', '')))
        slices = [os.path.basename(f).replace('.png', '') for f in paths]
        paths_mask = [f'/research/groups/Multiscale_Imaging/users/Melina_Estela/AE/data/mtbi/{animal_name}/fine_masks/{stain}/{slice}_mask.png' for slice in slices]

    # else:
        # paths = [path_animal + slice + '.png' for slice in slices]
    #    paths = [path_animal + slice + '.png' for slice in slices]
    #    paths_mask = [f'.../AE/data/mtbi/{animal_name}/fine_masks/{slice}_mask.png' for slice in slices]


    # now let's create an instance of the ae architecture and load the trained model
    AE = ae.instance()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    path_to_saved_model = '.../AE/data/AE_trained/batchnorm/'+ae_pth
    AE.load_state_dict(torch.load(path_to_saved_model, map_location=device, weights_only=True))

    AE.to(device)
    AE.eval()  # set the model to evaluation mode (important for batch normalization and dropout)

    base_dir = f'.../AE/data/AE_usage/{ae_name}/'
    dirs2create = [
        base_dir,
        f'{base_dir}{animal_name}/',
        f'{base_dir}{animal_name}/LS/',
        f'{base_dir}{animal_name}/LS/all/',
        f'{base_dir}{animal_name}/LS/all/latent'
    ]

    for directory in dirs2create:
        os.makedirs(directory, exist_ok=True)

    path_LS = f'{base_dir}{animal_name}/LS/all/latent/'
    path_RC = f'{base_dir}{animal_name}/RC/{stain}/'

    count = -1
    for path_im, path_m in zip(paths, paths_mask):
        count += 1
        img = np.array(Image.open(path_im).convert('L')) / 255.0
        mask = np.array(Image.open(path_m).convert('L')) / 255.0

        # images = [np.array(Image.open(path).convert('L')) / 255.0 for path in paths]
        # masks = [np.array(Image.open(path).convert('L')) / 255.0 for path in paths_mask]

        image = img * mask

        n_col = image.shape[1] // window
        n_row = image.shape[0] // window

        # slice = slice.replace('_mask', '')
        LS = []
        RC = []

        for i in range(n_row):
            tiles = [image[i * window:(i + 1) * window, j * window:(j + 1) * window] for j in range(n_col)]
            tiles = np.array(tiles)
            tiles = torch.from_numpy(tiles).unsqueeze(1)  # this function keep the float32 dtype & add a channel dimension

            with torch.no_grad():
                latent = AE.encoder(tiles)
                reconstr = AE.decoder(latent)

            # Convert tensors to numpy arrays:
            latent = latent.numpy()
            reconstr = reconstr.numpy()
            reconstr = (reconstr * 255.0).astype(np.uint8)

            LS.append(latent)
            RC.append(reconstr)

            del tiles, latent, reconstr

        LS = np.concatenate(LS, axis=0)
        np.save(path_LS + 'latent_img' + slices[count] + '.npy', LS)

        RC = np.concatenate(RC, axis=0)

        reconstructed_image = np.zeros((n_row * window, n_col * window))
        patch_idx = 0
        for i in range(n_row):
            for j in range(n_col):
                reconstructed_image[i * window:(i + 1) * window, j * window:(j + 1) * window] = RC[
                    patch_idx, 0]  # Remove the channel dimension
                patch_idx += 1

        # Rescale the pixel values from range [0,1] to [0,255]
        reconstructed_image = (reconstructed_image * 255).astype(np.uint8)
        # black and white scale is flipped
        reconstructed_image = (255 - reconstructed_image)
        # Save the reconstructed image using PIL
        reconstructed_image_array = Image.fromarray(reconstructed_image)

        # reconstructed_image_array = reconstructed_image_array.astype(np.uint8)
        reconstructed_image_array.save(path_RC + 'reconstr_img' + slices[count] + '.png')



getting_LS_and_reconstruction('model128', 256, 128, 'ISM_31', 'mye', ['all'])
# getting_LS_and_reconstruction('model128', 256, 128, 'ISM_37', 'mye', ['all'])
# getting_LS_and_reconstruction('model128', 256, 128, 'ISM_26', 'mye', ['all'])
# getting_LS_and_reconstruction('model128', 256, 128, 'ISM_37', 'mye', ['all'])

