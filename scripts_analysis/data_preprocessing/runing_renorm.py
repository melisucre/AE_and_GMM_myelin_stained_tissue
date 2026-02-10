import renormalization_def as renorm
from multiprocessing import Pool


path = '.../AE/data/mtbi/'
animal_ref = 'ISM_27'

animal_data_references = [
    {'animal': 'ISM_24', 'stain_ref': [('mye', 44), ('nis', 39)]},
    {'animal': 'ISM_26', 'stain_ref': [('mye', 42), ('nis', 43)]},
    {'animal': 'ISM_27', 'stain_ref': [('mye', 44), ('nis', 44)]},
    {'animal': 'ISM_29', 'stain_ref': [('mye', 44), ('nis', 45)]},
    {'animal': 'ISM_30', 'stain_ref': [('mye', 44), ('nis', 45)]},
    {'animal': 'ISM_31', 'stain_ref': [('mye', 43), ('nis', 44)]},
    {'animal': 'ISM_32', 'stain_ref': [('mye', 38), ('nis', 38)]},
    {'animal': 'ISM_33', 'stain_ref': [('mye', 45), ('nis', 45)]},
    {'animal': 'ISM_34', 'stain_ref': [('mye', 46), ('nis', 47)]},
    {'animal': 'ISM_35', 'stain_ref': [('mye', 43), ('nis', 44)]},
    {'animal': 'ISM_36', 'stain_ref': [('mye', 45), ('nis', 45)]},
    {'animal': 'ISM_37', 'stain_ref': [('mye', 40), ('nis', 40)]},
    {'animal': 'ISM_38', 'stain_ref': [('mye', 37), ('nis', 38)]}]

animal_dictionary = {}

for entry in animal_data_references:
    animal_name = entry['animal']
    stain_and_ref = entry['stain_ref']

    animal_dictionary[animal_name] = {}

    for staining, reference in stain_and_ref:
        animal_dictionary[animal_name][staining] = reference


animal_name = 'ISM_31'
stain = 'mye'
slice_ref = animal_dictionary[animal_name][stain]

renorm.renorm(path, animal_name, animal_ref, stain, slice_ref)

# def submiting_job(args):
#     animal_name, stain, slice_ref = args
#     renorm.renorm(path, animal_name, animal_ref, stain, slice_ref)
#
# animal_name = 'ISM_24'
# jobs = []
#
# for stain, reference in animal_dictionary[animal_name].items():
#     jobs.append((animal_name, stain, reference))
#
# with Pool(processes=len(jobs)) as pool:
#     pool.map(submiting_job, jobs)
#
# print('all jobs done!')


# renorm.renorm(path, animal_name, animal_ref, stain, animal_dictionary[animal_name][stain])
# def submit_job(animal_name, stain, slice_ref):
#     renorm.renorm(path, animal_name, animal_ref, stain, slice_ref)
#
# with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
#     jobs = []
#
#     for animal_name, stains in animal_dictionary.items():
#         for stain, reference in stains.items():
#             job = pool.apply_async(submit_job, args=(animal_name, stain, reference))
#             jobs.append(job)
#
#     for job in jobs:
#         job.get()




