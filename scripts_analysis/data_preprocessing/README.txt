# note that SNAKEMAKE file was not fully implemented.

1. original data is ndpi format. by running ndpi2tiff.sh in mac, ndpi transformed to tif in both original resolution and downsampled (paths not updated at all)
2. more downsampled is needed & detecting masks: coordinates.py (which calls two functions: down_sampling & connected_components, in functions4finding_coordinates.py)
3. need to check manually if there is any issue in some glass from previous step, check folder: some_issue
4. if issues detected, manually search for the coordinates in down_sampled images and write them in coordinates folder. useful file: manual_inspection.py

5. proceed to cut the slices: bounding_rectangle.py
6. recheck the cuts: in order to visualize, check_cuts.py
    -> borders were not super ok in some cases (modified to wider borders), so if needed, change coordinates again (in coordinates folder, downsampled) and go back to 5.

7. rename the slices from 1 to N, by enumerate.py (folder ordered is created)

8. next step is normalize. for this, two things should be decided: which sections will be normalized (we are discarding frontal and posterior slices. I have notes on ISM animals sections discarded in notebook)
	this is done by inspection. 
	other thing to decide, from central part of the brain, what is the slice we use as Reference. (this has been decided too, notes in my notebook).

	then go to: normalization.py

as we need to normalize among animals (we need a Reference number section for each animal):
9. normalization_among_animals.py ( this has not been added to the snakefile)
then normalize again each animal using that Reference corrected among animals:
10. then check renormalization_def.py and run runing_renorm.py



