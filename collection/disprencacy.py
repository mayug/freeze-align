import os
import tarfile
from tqdm import tqdm
import multiprocessing as mp
root_dir = '/home/storage/classwise_collection/laion_4k_cls_5k_samples/'
n_parts = 1363

# check if a jpg, json and txt file exists inside the tar file



def check_tar_files(i):
    discrepancies = []
    part_name = f'{i:05d}.tar'
    part_tar = tarfile.open(os.path.join(root_dir, part_name))
    members = part_tar.getmembers()
    
    current_dict = {}
    for member in part_tar.getmembers():
        key = member.name.split('.')[0]
        try:
            current_dict[key].append(member)
        except KeyError:
            current_dict[key] = [member]
    for key, value in current_dict.items():
        if len(value) != 3:
            discrepancies.append((i, key, value))
    return discrepancies





pool = mp.Pool(mp.cpu_count())
results = pool.map(check_tar_files, range(n_parts))
pool.close()
pool.join()

discrepancies = []
for r in results:
    discrepancies.extend(r)

print(discrepancies)