 import numpy as np
from sklearn.decomposition import IncrementalPCA
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from patchify import patchify
from torch.utils.data import Dataset
import pickle
import os


def patchifying(file_path, patch_size, num_patches):

    image = Image.open(file_path).convert('L')
    image = np.array(image)

    mask_path = file_path.replace("renormalized", "fine_masks").replace(".png", "_mask.png")
    mask = Image.open(mask_path).convert('L')
    mask = np.array(mask)

    # Patchify image and mask
    image_patches = patchify(image, (patch_size, patch_size), step=patch_size)
    mask_patches = patchify(mask, (patch_size, patch_size), step=patch_size)

    # Flatten both sets of patches
    image_patches = image_patches.reshape(-1, patch_size, patch_size)
    mask_patches = mask_patches.reshape(-1, patch_size, patch_size)

    # Classify patches
    foreground_indices = [i for i, m in enumerate(mask_patches) if np.any(m > 0)]
    background_indices = [i for i, m in enumerate(mask_patches) if not np.any(m > 0)]

    # Shuffle both for randomness
    np.random.shuffle(foreground_indices)
    np.random.shuffle(background_indices)

    # Select from both sets
    selected_indices = foreground_indices[:num_patches]
    if len(selected_indices) < num_patches:
        n_needed = num_patches - len(selected_indices)
        if len(background_indices) < n_needed:
            raise ValueError(f"Not enough total patches to reach {num_patches}")
        selected_indices += background_indices[:n_needed]

    selected_patches = image_patches[selected_indices]
    return selected_patches


class PCADataset(Dataset):

    def __init__(self, file_paths, patch_size, num_patches_per_section):
        self.file_paths = file_paths
        self.patch_size = patch_size
        self.num_patches_per_section = num_patches_per_section
        self.total_patches = num_patches_per_section * len(file_paths)

        # Current state
        self.current_patches = None
        self.current_patch_count = 0
        self.current_image_idx = -1

        print(f'PCA Dataset initialized with {len(file_paths)} images')
        print(f'Expected total patches: {self.total_patches}')

    def __len__(self):
        return self.total_patches

    def load_one_image(self):
        """Load patches from the next image"""
        print(
            f"Loading image {self.current_image_idx + 1}/{len(self.file_paths)}: {os.path.basename(self.file_paths[self.current_image_idx])}")
        self.current_patches = patchifying(
            self.file_paths[self.current_image_idx],
            self.patch_size,
            self.num_patches_per_section
        )
        self.current_patch_count = len(self.current_patches)

    def __getitem__(self, idx):
        if self.current_patch_count == 0:
            self.current_image_idx += 1
            if self.current_image_idx >= len(self.file_paths):
                self.current_image_idx = 0
                print('Wrapped around to first image')
            self.load_one_image()

        if self.current_patch_count > 0:
            patch = self.current_patches[self.current_patch_count - 1]
            self.current_patch_count -= 1
            # Return flattened patch normalized to [0,1] for PCA
            return patch.flatten() / 255.0


def create_patch_generator(dataset, batch_size=1024):
    """
    Generator that yields batches of flattened patches for incremental PCA
    """
    batch = []

    for i in range(len(dataset)):
        patch = dataset[i]  # This returns a flattened, normalized patch
        batch.append(patch)

        # Yield batch when it's full
        if len(batch) >= batch_size:
            yield np.array(batch)
            batch = []
            print(f"Yielded batch of {batch_size} patches (processed {i + 1}/{len(dataset)} total patches)")

    # Yield remaining patches if any
    if batch:
        yield np.array(batch)
        print(f"Yielded final batch of {len(batch)} patches")


def train_incremental_pca(file_paths, patch_size, num_patches_per_section, path_save,
                          n_components=256, batch_size=1024):
    """
    Train incremental PCA on image patches
    """
    print("=== Starting Incremental PCA Training ===")
    print(f"Images: {len(file_paths)}")
    print(f"Patches per image: {num_patches_per_section}")
    print(f"Patch size: {patch_size}x{patch_size}")
    print(f"PCA components: {n_components}")
    print(f"Batch size: {batch_size}")

    # Create dataset
    dataset = PCADataset(file_paths, patch_size, num_patches_per_section)

    # Initialize IncrementalPCA
    feature_dim = patch_size * patch_size
    ipca = IncrementalPCA(n_components=n_components)

    print(f"\nFeature dimension: {feature_dim}")
    print("Training started...")

    # Train incrementally
    batch_count = 0
    total_samples = 0

    for batch in create_patch_generator(dataset, batch_size):
        batch_count += 1
        total_samples += batch.shape[0]

        print(f"Training on batch {batch_count}, shape: {batch.shape}")
        ipca.partial_fit(batch)

        # Print progress every some batches
        if batch_count % 500 == 0:
            print(f"Completed {batch_count} batches, {total_samples} total samples processed")

    print(f"\n=== Training Complete ===")
    print(f"Total batches processed: {batch_count}")
    print(f"Total samples processed: {total_samples}")
    print(f"Explained variance ratio (first 10 components): {ipca.explained_variance_ratio_[:10]}")
    print(f"Total explained variance: {ipca.explained_variance_ratio_.sum():.4f}")

    # Save the model if path provided
    if path_save:
        with open(path_save, 'wb') as f:
            pickle.dump(ipca, f)
        print(f"Model saved to: {path_save}")

    return 0 # ipca


def transform_patches_with_pca(ipca, file_paths, patch_size, num_patches_per_section,
                               batch_size=1000, save_transformed=None):
    """
    Transform patches using trained PCA model
    """
    print("=== Transforming patches with trained PCA ===")

    dataset = PCADataset(file_paths, patch_size, num_patches_per_section)
    transformed_batches = []

    for batch in create_patch_generator(dataset, batch_size):
        transformed_batch = ipca.transform(batch)
        transformed_batches.append(transformed_batch)
        print(f"Transformed batch shape: {batch.shape} -> {transformed_batch.shape}")

    # Combine all transformed batches
    all_transformed = np.vstack(transformed_batches)
    print(f"Final transformed data shape: {all_transformed.shape}")

    if save_transformed:
        np.save(save_transformed, all_transformed)
        print(f"Transformed data saved to: {save_transformed}")

    return all_transformed


def load_pca_model(model_path):
    """Load saved PCA model"""
    with open(model_path, 'rb') as f:
        ipca = pickle.load(f)
    print(f"PCA model loaded from: {model_path}")
    print(f"Components: {ipca.n_components_}")
    return ipca


