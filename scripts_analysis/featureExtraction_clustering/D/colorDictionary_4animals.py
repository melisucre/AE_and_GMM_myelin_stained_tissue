import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def matching_colors(animal_name, ae_name):

    cluster_color_mapping = {0: 'o2', 1: 'brown', 2: 'p4', 3: 'r1', 4: 'amber', 5: 'g2', 6: 'b1', 7: 'p3', 8: 'p1', 9: 'g4', 10: 'light_coral', 11: 'p2', 12: 'g3', 13: 'rose_pink', 14: 'o1', 15: 'b4', 16: 'g1', 17: 'b2', 18: 'b3', 19: 'grey', 20: 'grey2'}

    def sym_kl(mu1, s1, mu2, s2):
        d = mu1.shape[0]
        kl = 0.5 * ((d * (s1*s1)/(s2*s2)+(s2*s2)/(s1*s1)) - 2*d + np.sum((mu1 - mu2)**2)/(s1*s1) + np.sum((mu1 - mu2)**2)/(s2*s2))
        return kl

    # reference
    centroids_ISM_31 = np.load(f'/.../AE/data/AE_usage/{ae_name}/ISM_31/gmm/100000_means_spherical_regcove-0.001.npy', allow_pickle=True)
    cov_ISM_31 = np.load(f'/.../AE/data/AE_usage/{ae_name}/ISM_31/gmm/100000_covariances_spherical_regcove-0.001.npy', allow_pickle=True)



    # colors needed:
    centroids_ISM_x = np.load(f'/...AE/data/AE_usage/{ae_name}/4animals/gmm/means_spherical_regcove-0.001.npy', allow_pickle=True)
    cov_ISM_x = np.load(f'/.../AE/data/AE_usage/{ae_name}/4animals/gmm/covariances_spherical_regcove-0.001.npy', allow_pickle=True)


    # changed 8 to 6, since i do now 21 clusters
    # Step 1: Match ISM_31[6] → ISM_x[6]
    mus_1 = centroids_ISM_31[6]
    sigmas_1 = cov_ISM_31[6]
    mus_2 = centroids_ISM_x[6]
    sigmas_2 = cov_ISM_x[6]

    # Compute symmetrized KL divergence matrix
    D = np.zeros((len(mus_1), len(mus_2)))
    for i in range(len(mus_1)):
        for j in range(len(mus_2)):
            D[i, j] = sym_kl(mus_1[i], sigmas_1[i], mus_2[j], sigmas_2[j])

    r, c = linear_sum_assignment(D)
    color_mapping = {int(c[i]): cluster_color_mapping[int(r[i])] for i in range(len(r))}
    new_color_mapping = {6: color_mapping}  # Start with level 8

    # Step 2: Iteratively match ISM_x[p+1] → ISM_x[p]
    for p in range(5, -1, -1):
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