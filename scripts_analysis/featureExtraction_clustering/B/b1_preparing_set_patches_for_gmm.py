import json
import random
import os
import re

# from all sections included of a single animal, we pick randomly num_random_points from those sections 
# we do this for the different patch sizes, and we save them in the animal's folder

animal_names = ['ISM_26', 'ISM_30', 'ISM_31', 'ISM_37']

main_path = '.../AE/data/mtbi/'
stain = 'mye'

num_random_points = 100000

def patches4gmm(animal_name):

    animal_path = f'{main_path}/{animal_name}'
    images_included = f'{animal_path}/renormalized/{stain}'
    file_path = f'{animal_path}/counting_patches.json'

    # get the list of .png files that are named as numbers and extract their numeric part
    numbers = [int(re.search(r'(\d+)\.png$', f).group(1)) for f in os.listdir(images_included)
               if re.match(r'^\d+\.png$', f)]

    # open and read the JSON file
    with open(file_path, 'r') as f:
        results = json.load(f)

    # cumulative counts for each window size
    # window_sizes = [64, 128, 256]
    window_sizes = [128, 256]
    cumulative_counts = {size: sum(results[str(n)][str(size)]['count'] for n in numbers) for size in window_sizes}

    # generate random indices for each window size
    random_indices = {w: sorted(random.sample(range(cumulative_counts[w]), num_random_points)) for w in window_sizes}

    # dictionary to store coordinates of randomly selected patches
    selected_patches = {w: [] for w in window_sizes}


    for w in window_sizes:
        total_count = 0
        current_index = 0
        L = len(random_indices[w])

        for n in numbers:
            image_count = results[str(n)][str(w)]['count']
            coordinates = results[str(n)][str(w)]['coordinates']

            while (current_index < L) and (random_indices[w][current_index] < (total_count + image_count)):
                idx_image = random_indices[w][current_index] - total_count
                selected_coordinates = coordinates[idx_image]
                selected_patches[w].append({'image': n, 'coordinates': selected_coordinates})

                # Move to the next random index
                current_index += 1

            total_count += image_count

    # Save the selected patches and coordinates to a new JSON file
    for w in window_sizes:
        output_selected_patches = f'{animal_path}/w_{w}_{num_random_points}_selected_patches.json'
        with open(output_selected_patches, 'w') as json_file:
            json.dump(selected_patches[w], json_file, indent=4)

        print(f"Selected patches saved to {output_selected_patches}")

for animal_name in animal_names:
    patches4gmm(animal_name)
