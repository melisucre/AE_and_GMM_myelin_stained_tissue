import os
import numpy as np
import sys
from sklearn.mixture import GaussianMixture
import joblib

# Define the animals, AE names, and respective reg_covar values
animals = ['ISM_26', 'ISM_30', 'ISM_31', 'ISM_37']
ae_names = ['AE_model128', 'AE_model256']
windows = [128, 256]

num_patches = 100000
reg_covar_values = [1e-3, 1e-3]

main_path = '.../AE/data/AE_usage'
s = -1

for ae_name in ae_names:

    s+=1
    window = windows[s]
    reg_covar = reg_covar_values[s]

    base_dir = f'{main_path}/{ae_name}/4animals/gmm/'
    os.makedirs(base_dir, exist_ok=True)

    LatSpace_all = []
    
    for animal in animals:
        LatSpace_path = f'{main_path}/{ae_name}/{animal}/gmm/{num_patches}_LS_gmm.npy'
        LatSpace = np.load(LatSpace_path)
        LatSpace_all.append(LatSpace)

    LatSpace_all = np.vstack(LatSpace_all)

    l = len(LatSpace_all)
    print(f"just checkin: lentgh is {l} and should be 100000")

    N = [3, 6, 9, 12, 15, 18, 21, 24, 27]
    # cov_types = ['full', 'spherical', 'diag', 'tied']
    cov_types = ['spherical']

    for cov_type in cov_types:
        bic_values = []
        models = []
        means = []
        covariances = []
        weights = []

        for n in N:
            try:
                gmm = GaussianMixture(n_components=n, covariance_type=cov_type, reg_covar=reg_covar, warm_start=False)
                gmm.fit(LatSpace_all)
                bic_values.append(gmm.bic(LatSpace_all))
                models.append(gmm)
                means.append(gmm.means_)
                covariances.append(gmm.covariances_)
                weights.append(gmm.weights_)

                # Save the models using joblib
                joblib.dump(gmm, os.path.join(base_dir, f'gmm_{cov_type}_n{n}_regcove-{reg_covar}.pkl'))


            except Exception as e:
                print(f"Error with {cov_type} and {n} components: {e}")
                bic_values.append(np.inf)

        # Save results
        np.save(os.path.join(base_dir, f'bic_{cov_type}_regcove-{reg_covar}.npy'), bic_values)
        np.save(os.path.join(base_dir, f'means_{cov_type}_regcove-{reg_covar}.npy'), means)
        np.save(os.path.join(base_dir, f'covariances_{cov_type}_regcove-{reg_covar}.npy'), covariances)
        np.save(os.path.join(base_dir, f'weights_{cov_type}_regcove-{reg_covar}.npy'), weights)

        print(f"Finished {ae_name} with {cov_type} covariance")

