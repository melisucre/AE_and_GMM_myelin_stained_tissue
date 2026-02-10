explanation on what each folder contains (codes)

A. here we check the training performances and pass the data through the trained models
	a0_plotErrorsAltogether.py (plot errors from AEs training)
	a1_passImagesThroughAEs.py (pass sections through the AE and PCA models)
	a1_pca_applied.py

B. here we extract randomly (but fixed to compare methods) patches, that will be used for the clustering
	counting_points_sections_animals.py (this goes through all sections per animal, making masks and there, counting and taking coordinates of all possible non-overlapping tiles (patch sizes = 128, 256))
	preparing_set_patches_for_gmm.py (this selects per animal random points from all possibilities counted previously, for the different window sizes. tested two numbers of points = 10000, 100000 (used))
	pass_patches_through_AE.py (for clustering)
	pass_patches_through_PCA.py (for clustering)

C. here we cluster (gmm) the compressed representations of the set of patches extracted in B
	c0_clustering_ae_4animals.py
	c0_clustering_pca_4animals.py
	c1_pass_4animals_gmmmodel_ae_to_sections.py
	c1_pass_4animals_gmmmodel_pca_to_sections.py
	c2_plot_gmm_bic_animals_together_ae.py
	c2_plot_gmm_bic_animals_together_pca.py

D. plots of clusters (matching colors)

	ae_model128_clusterplots_4animalstogether.py & colorDictionary_4animals.py
	ae_model128_clusterplots_ISM31.py & color_dictionary_27_3.py

	figure_k3_9_21_3x3.py (plots 3 sections from same animal for clusters 3,9,21)

	4animals_probmaps/ codes for the gmm of 4 animals together

