import argparse
import pandas as pd
import os
import multiprocessing
from tqdm.contrib.concurrent import process_map as tqdm_process_map
from multiprocessing import Queue as multi_queue
import time
from tqdm import tqdm
import heapq
import torch
import pickle
import multiprocessing
import subprocess
import s3fs
import s3fs
fs = s3fs.S3FileSystem()

import subprocess

LAION_LOCATION = 's3://vlm-datasets-us-east1/data/laion_full/'
N_PARTS = 31



import tarfile


def select_stuff_from_tar(current_og_part_selected, part):
    part = int(part)
    part_name = f'{part:05d}.tar'
    file= fs.open(os.path.join(laion_root, part_name), 'rb')
    # Use tarfile to read the tar file from S3
    part_tar = tarfile.open(fileobj=file, mode='r:*')

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


def create_tar(selected, target_dir, part):
    part_name = f'{part:05d}.tar'
    with tarfile.open(os.path.join(target_dir, f'./{part_name}'), 'w') as tar:
        # with tarfile.open(f) as tar:
        for i, item in enumerate(selected):
            tar.addfile(*item)
    return part_name

def parse_args():
    """
    Parse the following arguments for a default parser
    """
    parser = argparse.ArgumentParser(
        description="Getting dataset on dataset segment"
    )
    parser.add_argument(
        "--subset_parquet",
        dest="subset_parquet",
        default='/home/ec2-user/SageMaker/collection_4k_classes_5k_each.snappy.parquet',
        help="parquet for subsetting from collection code",
        type=str,
    )
    parser.add_argument(
        "--target_path",
        dest="target_path",
        default='/home/ec2-user/SageMaker/collected',
        help="target folder path",
        type=str,
    )
    parser.add_argument(
        "--size",
        dest="size",
        help="shard size",
        default=10,
        type=int,
    )

    parser.add_argument(
        "--num_cpus",
        dest="num_cpus",
        help="number of cpus to use for multi processing",
        default=32,
        type=int,
    )
    parser.add_argument("--og_parts",
        dest="og_parts",
        help="number of parts in original laion400m_data",
        default=100,
        type=int,
    )

    return parser.parse_args()



if __name__ == "__main__":
    args = parse_args()
    laion_root = LAION_LOCATION
    subset_parquet_file = args.subset_parquet


    # # load the parquet
    # for part in range(N_PARTS):
    #     part_name = f'{part:05d}'
    #     part_parquet = pd.read_parquet(os.path.join(laion_root, f'{part_name}.parquet')) 

    os.makedirs('./temp', exist_ok=True)

    if os.path.exists(f'./temp/part2sample_{args.og_parts}.pt') and os.path.exists(f'./temp/sample2part_{args.og_parts}.pt'):
        print('loading cache files part2sample and sample2part')
        part2sample = torch.load(f'./temp/part2sample_{args.og_parts}.pt')
        sample2part = torch.load(f'./temp/sample2part_{args.og_parts}.pt')
    
    else:
        # creating the cache files
        print('creating cache files for part2sample and sample2part')
        print('using {} cpus'.format(args.num_cpus))

        def read_urls_from_parquet(part):
            # Read a parquet file
            # file_path = f'/notebooks/data/laion400m-data/{str(part).zfill(5)}.parquet'
            file = fs.open(os.path.join(laion_root, f'{int(part):05d}.parquet'),'rb')
            df = pd.read_parquet(file)
            # df = pd.read_parquet(file_path)
            # Return the list of URLs
            return part, df['url'].tolist()

        part2sample = {}
        sample2part = {}
        results = []

        num_cpus=args.num_cpus
        og_parts=args.og_parts


        results = tqdm_process_map(read_urls_from_parquet, range(og_parts), max_workers=num_cpus)

        for i in results:
            part2sample[i[0]] = i[1]

        for part, samples in part2sample.items():
            for sample in samples:
                sample2part[sample] = part

        torch.save(part2sample, f'./temp/part2sample_{args.og_parts}.pt')
        torch.save(sample2part, f'./temp/sample2part_{args.og_parts}.pt')

    
    # asd

    print('reading subset parquet file', subset_parquet_file)
    subset_parquet = pd.read_parquet(subset_parquet_file)

    # just for testing; remove later
    subset_parquet = subset_parquet[subset_parquet['URL'].isin(sample2part.keys())]


    # add part column to the subset_parquet

    subset_parquet['part'] = subset_parquet['URL'].apply(lambda x: sample2part[x])

    # go through all the parts and select the files

    part_list = subset_parquet['part'].unique()

    




    selected_files = []
    selected_dfs = []
    save_part = 0

    # now divide the part_list into num_cpu parts and process them in parallel
    # each process saves the files in temp_proc_{part} folder and 
    # then we combine them all at the end

    def process_part(part):
        file = fs.open(os.path.join(laion_root, f'{str(part).zfill(5)}.parquet'), 'rb')
        current_og_part = pd.read_parquet(file)
        current_og_part_selected = current_og_part[current_og_part['url'].isin(subset_parquet['URL'])]
        current_og_part_selected = current_og_part_selected[current_og_part_selected['status']=='success']
        selected_fs = select_stuff_from_tar(current_og_part_selected, part)
        selected_df = current_og_part_selected
        return selected_fs, selected_df
    

    def single_worker(parts_list, worker_id):
        target_path = os.path.join(args.target_path, f'temp_proc_{worker_id}')
        os.makedirs(target_path, exist_ok=True)
        selected_files = []
        selected_dfs = []
        save_part = 0
        if worker_id == 0:
            # Only worker 0 will have a tqdm progress bar
            progress_bar = tqdm(total=len(parts_list), desc=f"Worker {worker_id} Progress")

        for part in parts_list:
            selected_fs, selected_df = process_part(part)
            selected_files.extend(selected_fs)
            selected_dfs.append(selected_df)

            if len(selected_files)/3 >= args.size:
                part_name = create_tar(selected_files, target_path, save_part)
                selected_df = pd.concat(selected_dfs)
                # selected_df.to_parquet(os.path.join(args.target_path, f'{save_part}.parquet'))
                selected_files = []
                selected_dfs = []
                # print(f'created {part_name}')
                # print('at worker', worker_id)
                save_part += 1
            if worker_id == 0:
                progress_bar.update(1)
        # saving the remaining files
        part_name = create_tar(selected_files, target_path, save_part)
        if worker_id == 0:
            progress_bar.close()

    
    parts_list = sorted(list(part_list))
    print('parts_list length', len(parts_list))
    num_cpus = args.num_cpus

    parts_per_cpu = len(parts_list)//num_cpus

    chunked_parts = [parts_list[i:i+parts_per_cpu] for i in range(0, len(parts_list), parts_per_cpu)]

    print('chunked_parts length', len(chunked_parts))
    print([len(i) for i in chunked_parts])
    assert sum([len(i) for i in chunked_parts]) == len(parts_list)
    # asd

    processes = []
    for i, parts in enumerate(chunked_parts):
        p = multiprocessing.Process(target=single_worker, args=(parts, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    # asd
    # now combine all the parts
    print('combining all the parts')
    save_part = 0
    for i in tqdm(range(num_cpus+1)):
        target_path = os.path.join(args.target_path, f'temp_proc_{i}')
        for file in os.listdir(target_path):
            save_name = f'{save_part:05d}.tar'
            subprocess.call(f"mv {os.path.join(target_path, file)} {os.path.join(args.target_path, save_name)}",shell=True)
            save_part += 1
        subprocess.call(f"rm -r {target_path}",shell=True)
    

    print('done')