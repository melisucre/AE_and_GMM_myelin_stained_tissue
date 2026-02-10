import os
import numpy as np
import joblib

# Load the saved GMM model

animal_names = ['ISM_26', 'ISM_30', 'ISM_31', 'ISM_37']

image_nums = [['27', '45', '57'], ['28', '46', '57'], ['28', '46', '57'], ['23', '41', '53']]

ae_names = ['AE_model128', 'AE_model256']

windows = [128, 256]
reg_covar = 1e-3
num_patches = 100000

cov_type = 'spherical'

# N = [3, 6, 9, 12, 15, 18, 21, 24, 27]

N = [3, 9, 21]
s = -1
for ae_name in ae_names:
    base_dir = f'.../AE/data/AE_usage/{ae_name}/4animals/gmm/'

    for n in N:
        model_path = os.path.join(base_dir, f'gmm_{cov_type}_n{n}_regcove-{reg_covar}.pkl')
        gmm_model = joblib.load(model_path)
        s = -1
        for animal_name in animal_names:
            s+=1
            image_num = image_nums[s]
            directory = f'.../AE/data/AE_usage/{ae_name}/{animal_name}/LS/all/latent/'
            files = os.listdir(directory)
            latent_files = [file for file in files if "latent" in file and file.endswith(".npy") and any(img_num in file for img_num in image_num)]


            directory_save = base_dir + cov_type
            os.makedirs(directory_save, exist_ok=True)

            for ls in latent_files:
                file_path = os.path.join(directory, ls)
                data = np.load(file_path)
                labels = gmm_model.predict(data)
                probabilities = gmm_model.predict_proba(data)

                num = ls.replace('latent_img', '').replace('.npy', '')

                output_dir = directory_save + f'/{animal_name}_k{n}_num{num}_clustered.npy'
                np.save(output_dir, labels)
                output_dir2 = directory_save + f'/{animal_name}_k{n}_num{num}_clustered_predict_prob.npy'
                np.save(output_dir2, probabilities)
