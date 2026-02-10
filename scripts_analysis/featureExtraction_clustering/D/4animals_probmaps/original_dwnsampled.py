from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import numpy as np

animal_names = ['ISM_26', 'ISM_30', 'ISM_31', 'ISM_37']
image_nums = ['45', '46', '46', '41']  # Middle image for each animal
stain = 'mye'
path = '/.../AE/data/'

# Define output resolution (scale factor)
# Adjust as needed: 0.5 = 1/4 file size, 0.25 = 1/16 file size
scale_factor = 0.4

for animal_name, image_num in zip(animal_names, image_nums):
    # Load image
    path_image = f'{path}mtbi/{animal_name}/renormalized/{stain}/{image_num}.png'
    image = Image.open(path_image)

    # Load mask
    path_mask = f'{path}mtbi/{animal_name}/fine_masks/{stain}/{image_num}_mask.png'
    mask = Image.open(path_mask).convert('L')
    mask = np.array(mask) > 0  # Convert to boolean

    # Apply mask: set non-masked areas to black (0)
    image_array = np.array(image)
    image_array[~mask] = 255 # 0 Black for non-tissue

    # Convert back to image
    masked_image = Image.fromarray(image_array)

    # Reduce resolution
    new_size = tuple(int(dim * scale_factor) for dim in masked_image.size)
    masked_image_resized = masked_image.resize(new_size, Image.Resampling.LANCZOS)

    # Save
    output_dir = f'{path}AE_usage/AE_model128/4animals/gmm/spherical_plots/manuscript/'
    output_path = f'{output_dir}{animal_name}_{image_num}_masked_lowres.png'
    masked_image_resized.save(output_path, quality=95)

    original_size = image.size
    file_size_mb = masked_image_resized.getbbox()  # Just for reference
    print(f'Saved: {output_path}')
    print(f'  Original size: {original_size} -> Resized to: {new_size}')
    print()