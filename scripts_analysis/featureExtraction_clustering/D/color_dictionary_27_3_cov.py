import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def matching_colors(animal_name):

    cluster_color_mapping = {2: 'g3', 8: 'g1', 6: 'grey2', 13: 'rose_pink', 10: 'p1', 23: 'o1', 1: 'b2', 5: 'grey',
                         21: 'light_coral', 11: 'g4', 15: 'b1', 26: 'p2', 20: 'r2', 14: 'b4', 25: 'o2', 24: 'brown',
                         18: 'p4', 16: 'r1', 0: 'teal', 12: 'y', 4: 'g2', 17: 'deep_orange', 9: 'black', 3: 'amber',
                         22: 'b3', 7: 'p3', 19: 'white'}

    def sym_kl(mu1, s1, mu2, s2):
        d = mu1.shape[0]
        kl = 0.5 * ((d * (s1*s1)/(s2*s2)+(s2*s2)/(s1*s1)) - 2*d + np.sum((mu1 - mu2)**2)/(s1*s1) + np.sum((mu1 - mu2)**2)/(s2*s2))
        return kl

    # reference
    centroids_ISM_31 = np.load('.../AE/data/AE_usage/AE_n9_mye_w128_LS256_batch64_batchnorm/ISM_31/gmm/100000_means_spherical_regcove-0.001.npy', allow_pickle=True)
    cov_ISM_31 = np.load('.../AE/data/AE_usage/AE_n9_mye_w128_LS256_batch64_batchnorm/ISM_31/gmm/100000_covariances_spherical_regcove-0.001.npy', allow_pickle=True)

    # colors needed:
    centroids_ISM_x = np.load(f'.../AE/data/AE_usage/AE_n9_mye_w128_LS256_batch64_batchnorm/{animal_name}/gmm/100000_means_spherical_regcove-0.001.npy', allow_pickle=True)
    cov_ISM_x = np.load(f'.../AE/data/AE_usage/AE_n9_mye_w128_LS256_batch64_batchnorm/{animal_name}/gmm/100000_covariances_spherical_regcove-0.001.npy', allow_pickle=True)


    # Step 1: Match ISM_31[8] → ISM_x[8]
    mus_1 = centroids_ISM_31[8]
    sigmas_1 = cov_ISM_31[8]
    mus_2 = centroids_ISM_x[8]
    sigmas_2 = cov_ISM_x[8]

    # Compute symmetrized KL divergence matrix
    D = np.zeros((len(mus_1), len(mus_2)))
    for i in range(len(mus_1)):
        for j in range(len(mus_2)):
            D[i, j] = sym_kl(mus_1[i], sigmas_1[i], mus_2[j], sigmas_2[j])

    r, c = linear_sum_assignment(D)
    color_mapping = {int(c[i]): cluster_color_mapping[int(r[i])] for i in range(len(r))}
    new_color_mapping = {8: color_mapping}  # Start with level 8

    # Step 2: Iteratively match ISM_x[p+1] → ISM_x[p]
    for p in range(7, -1, -1):
        mus_1 = centroids_ISM_x[p + 1]
        sigmas_1 = cov_ISM_x[p + 1]
        mus_2 = centroids_ISM_x[p]
        sigmas_2 = cov_ISM_x[p]

        D = np.zeros((len(mus_1), len(mus_2)))
        for i in range(len(mus_1)):
            for j in range(len(mus_2)):
                D[i, j] = sym_kl(mus_1[i], sigmas_1[i], mus_2[j], sigmas_2[j])

        r, c = linear_sum_assignment(D)

        next_mapping = {}
        for i in range(len(r)):
            from_idx = r[i]
            to_idx = c[i]
            color = color_mapping.get(from_idx, None)
            if color:
                next_mapping[to_idx] = color
        color_mapping = next_mapping
        new_color_mapping[p] = color_mapping

    return new_color_mapping