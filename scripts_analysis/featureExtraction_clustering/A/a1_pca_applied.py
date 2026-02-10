import numpy as np
import os
import joblib
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import pickle
import glob


def load_pca_model(model_path):
    """Load saved PCA model"""
    with open(model_path, 'rb') as f:
        ipca = pickle.load(f)
    print(f"PCA model loaded from: {model_path}")
    print(f"Components: {ipca.n_components_}")
    return ipca


def extract_and_reconstruct_pca(
    base_dir,
    animal_name,
    stain,
    slices,
    model_path,
    output_dir,
    patch_size,
    stride,
    num_components
):
    """
    Extract patches and reconstruct PCA for multiple slices
    1. Extract patches from an image (only masked areas)
    2. Transform to PCA features
    3. Reconstruct image from PCA
    4. Save features, coordinates, and reconstructed image
    """

    # Handle 'all' slices case
    if slices[0] == 'all':
        # Path to renormalized images
        path_animal = f'{base_dir}/mtbi/{animal_name}/renormalized/{stain}'
        paths = glob.glob(os.path.join(path_animal, '*.png'))
        paths.sort(key=lambda f: int(os.path.basename(f).replace('.png', '')))
        slices = [os.path.basename(f).replace('.png', '') for f in paths]
        print(f"Found {len(slices)} slices to process: {slices}")


    # === Load PCA model ===
    pca = load_pca_model(model_path)

    for slice in slices:

        print(f"\nProcessing slice: {slice}")

        image_path = f'{base_dir}/mtbi/{animal_name}/renormalized/{stain}/{slice}.png'
        mask_path = f'{base_dir}/mtbi/{animal_name}/fine_masks/{stain}/{slice}_mask.png'

        # === Load image & mask ===
        image = Image.open(image_path).convert('L')
        image = np.array(image) / 255.0
        mask = Image.open(mask_path).convert('L')
        mask = np.array(mask)
        assert image.shape == mask.shape
        h, w = image.shape
        patches = []
        coords = []

        # === Extract patches ===
        for i in range(0, h - patch_size + 1, stride):
            for j in range(0, w - patch_size + 1, stride):
                patch_mask = mask[i:i+patch_size, j:j+patch_size]
                if patch_mask.mean() > 0.:
                    patch = image[i:i+patch_size, j:j+patch_size].flatten()
                    patches.append(patch)
                    coords.append((i, j))

        patches = np.array(patches)

        # === PCA transform ===
        pca_features = pca.transform(patches)

        # === Save features/coords ===
        image_id = os.path.splitext(os.path.basename(image_path))[0]
        base_name = f"{animal_name}_{stain}_{image_id}_PCA{num_components}"
        ooutput_dir = output_dir + '/pca/all'
        os.makedirs(ooutput_dir, exist_ok=True)
        np.save(os.path.join(ooutput_dir, f"{base_name}_features.npy"), pca_features)
        np.save(os.path.join(ooutput_dir, f"{base_name}_coords.npy"), np.array(coords))
        np.save(os.path.join(ooutput_dir, f"{base_name}_shape.npy"), np.array([h, w]))

        # === Inverse PCA ===
        patch_recon = pca.inverse_transform(pca_features)
        patch_reconstructions = patch_recon.reshape(-1, patch_size, patch_size)

        # === Place patches back ===
        recon_image = np.zeros((h, w), dtype=np.float32)
        count_image = np.zeros((h, w), dtype=np.float32)
        for patch, (pi, pj) in zip(patch_reconstructions, coords):
            recon_image[pi:pi+patch_size, pj:pj+patch_size] += patch
            count_image[pi:pi+patch_size, pj:pj+patch_size] += 1
        valid_mask = count_image > 0
        recon_image[valid_mask] /= count_image[valid_mask]

        # === Save reconstructed image ===
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir+'/RC', exist_ok=True)

        recon_image_uint8 = (recon_image * 255.0).clip(0, 255).astype(np.uint8)
        recon_path = f'{output_dir}/RC/{base_name}_reconstructed.png'
        Image.fromarray(recon_image_uint8).save(recon_path)

    print(f"Reconstructed image saved.")

    return 0 #pca_features, coords, recon_image_uint8
