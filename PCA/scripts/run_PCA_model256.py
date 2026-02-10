from PCA_model256 import train_incremental_pca
from pathlib import Path


def load_and_process_file_paths(txt_file_path, new_base_path):
    """
    Load file paths from txt file and convert them to new base path structure.

    Args:
        txt_file_path: Path to the .txt file containing original paths
        new_base_path: New base directory to combine with extracted parts

    Example:
        Original: /.../AE/data/ISM_24/mye/18.png
        New path: {new_base_path}/ISM_24/renormalized/mye/18.png
    """
    with open(txt_file_path, 'r') as f:
        original_paths = [line.strip() for line in f if line.strip()]

    new_paths = []

    for original_path in original_paths:
        # Convert to Path object for easier manipulation
        path_obj = Path(original_path)

        # Extract filename without extension (e.g., "18")
        filename_no_ext = path_obj.stem

        # Get parent directories and find the one that looks like ISM_XX
        parts = path_obj.parts

        # Find ISM_XX directory (assuming it starts with ISM_)
        ism_dir = None
        for part in parts:
            if part.startswith('ISM_'):
                ism_dir = part
                break

        if ism_dir is None:
            print(f"Warning: Could not find ISM_ directory in path: {original_path}")
            continue

        # Reconstruct the path: new_base_path/ISM_XX/renormalized/mye/filename.png
        new_path = Path(new_base_path) / ism_dir / "renormalized" / "mye" / f"{filename_no_ext}.png"
        new_paths.append(str(new_path))

    print(f"Processed {len(new_paths)} file paths")
    print(f"Example conversion:")
    if original_paths and new_paths:
        print(f"  Original: {original_paths[0]}")
        print(f"  New:      {new_paths[0]}")

    return new_paths

txt_file_path = '/.../AE/data/datasets_training/AE_model256_png_files_train.txt'
new_base_path = '/.../AE/data/mtbi'

png_files_train = load_and_process_file_paths(txt_file_path, new_base_path)

# Parameters
patch_size = 256
num_patches_per_section = 25600 # ...
n_components = 256  # Number of PCA components
batch_size = 1024  # Adjust based on memory

train_incremental_pca(
    file_paths=png_files_train,
    patch_size=patch_size,
    num_patches_per_section=num_patches_per_section,
    path_save=f'/.../AE/data/massivePCA/w{patch_size}pc{n_components}/PCA_model256.pkl',
    n_components=n_components,  # adjust as needed
    batch_size=batch_size  # adjust based on memory
)

