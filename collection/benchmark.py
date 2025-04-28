laion_root = '/home/storage/laion400m-data-50M/'


import tarfile
import os
import pandas as pd 
import time

current_og_part = pd.read_parquet(f'{laion_root}/00000.parquet')

start_time = time.time()

subset_parquet = pd.read_parquet('./collection_4k_classes_5k_each.snappy.parquet')

end_time = time.time()
execution_time = end_time - start_time

print("Execution time for reading subset parquet file :", execution_time, "seconds")

def select_stuff_from_tar(current_og_part_selected, part):
    part = int(part)
    part_name = f'{part:05d}.tar'
    part_tar = tarfile.open(os.path.join(laion_root, part_name))

    save_files = []

    for i in current_og_part_selected['key']:
        tar_info = part_tar.getmember(f'{i}.jpg')
        file_obj  = part_tar.extractfile(f'{i}.jpg')
        save_files.append((tar_info, file_obj))

        tar_info = part_tar.getmember(f'{i}.json')
        file_obj  = part_tar.extractfile(f'{i}.json')
        save_files.append((tar_info, file_obj))

        tar_info = part_tar.getmember(f'{i}.txt')
        file_obj  = part_tar.extractfile(f'{i}.txt')
        save_files.append((tar_info, file_obj))
    
    return save_files




urls_set = set(subset_parquet['URL'])

start_time = time.time()

current_og_part_selected = current_og_part[current_og_part['url'].apply(lambda x: x in urls_set)]

end_time = time.time()
execution_time = end_time - start_time

print("Execution time for url is in :", execution_time, "seconds")


start_time = time.time()

current_og_part_selected = current_og_part_selected[current_og_part_selected['status']=='success']

end_time = time.time()
execution_time = end_time - start_time

print("Execution time for success filtering :", execution_time, "seconds")


start_time = time.time()

part = 0
selected_fs = select_stuff_from_tar(current_og_part_selected, part)

end_time = time.time()
execution_time = end_time - start_time

print("Execution time for select stff from tar:", execution_time, "seconds")