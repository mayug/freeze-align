import torch, random
from tqdm import tqdm
import sys
import re
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed


# LAION_LOCATION = "/home/storage/datasets"
LAION_LOCATION = "/notebooks/data"

# RESULTS_LOCATION = "./results/"
# CLASS_NUM = 3763

def parse_args():
    """
    Parse the following arguments for a default parser
    """
    parser = argparse.ArgumentParser(
        description="Getting dataset on dataset segment"
    )
    parser.add_argument(
        "--parts",
        dest="parts",
        help="how much parts to do",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--max_amount",
        dest="max_amount",
        help="priority queue size",
        default=50000,
        type=int,
    )

    parser.add_argument(
        "--per_class",
        dest="per_class",
        help="samples per class",
        default=50000,
        type=int,
    )

    parser.add_argument(
        "--temp_select_type",
        dest="temp_select_type",
        help="temp select",
        default=None,
        type=str,
    )
    parser.add_argument(
        "--class_num",
        dest="class_num",
        help="number of classes",
        default=3763,
        type=int,
    )

    parser.add_argument(
        "--results_location",
        dest="results_location",
        help="root folder for results",
        default="./results/",
        type=str,
    )

    parser.add_argument(
        "--suffix",
        dest="suffix",
        help="suffix for the results file",
        default=None,
        type=str,
    )

    parser.add_argument(
    "--dataset",
    dest="dataset",
    help="sdataset to subselect from the scores files",
    default=None,
    type=str,
)

    return parser.parse_args()


if __name__ == "__main__":

    

    args = parse_args()

    parts = args.parts
    max_amount = args.max_amount
    per_class = args.per_class
    temp_select_type = args.temp_select_type
    results_location = args.results_location
    suffix = args.suffix
    class_num = args.class_num

    subset = max_amount-1 # using temp select only for class order and subset for subsetting from current base sample ids and scores

    if temp_select_type is None:
        print("Temp select is not provided, using max_amount-1")
        temp_select = max_amount-1
    elif temp_select_type=='mean':
        print('Using mean temp select')
        temp_select = max_amount-1
    else:
        print('using temp select ', int(temp_select_type))
        temp_select = int(temp_select_type)
        

    if suffix is not None:
        suffix = '_'+suffix
    else:
        suffix = ''

    if args.dataset is not None:
        dataset2indices = torch.load('./class_embeddings/imgprotos/dataset2indices.pt')

    print("Loading the results")
    print(results_location)
    print(f'./{parts - 1}/scores_so_far_{max_amount}{suffix}.pt')
    current_best_scores = torch.load(os.path.join(results_location, f'./{parts - 1}/scores_so_far_{max_amount}{suffix}.pt'), map_location = "cpu").numpy()
    current_best_sample_id = torch.load(os.path.join(results_location, f'./{parts - 1}/samples_so_far_{max_amount}{suffix}.pt'), map_location = "cpu").numpy()

    if args.dataset is not None:
        print(f"Subselecting from the scores and sample ids for {args.dataset}")
        indices = dataset2indices[args.dataset]
        print('before ', [len(indices), current_best_scores.shape, current_best_sample_id.shape])
        current_best_scores = current_best_scores[:, indices[0]:indices[1]+1]
        current_best_sample_id = current_best_sample_id[:, indices[0]:indices[1]+1]
        print('after', [current_best_scores.shape, current_best_sample_id.shape])
        class_num = indices[1] - indices[0] + 1
        print('class num', class_num)
        # asd

    print("Asssigning")
    # assignment = [[] for i in range(CLASS_NUM)]
    assignment = {}
    # scores = [[] for i in range(CLASS_NUM)]
    scores = {}

    if temp_select_type == 'mean':
        class_order = np.argsort(current_best_scores.mean(axis=0))
    else:
        class_order = np.argsort(current_best_scores[temp_select])
    visited = set()
    for class_id in tqdm(class_order):
        # class_samples = [sample_id for sample_id in current_best_sample_id[:temp_select, class_id] if sample_id not in visited][:per_class]
        # class_scores = [current_best_scores[:temp_select, class_id][j] for j, sample_id in enumerate(current_best_sample_id[:temp_select, class_id]) if sample_id not in visited][:per_class]
        class_samples = [sample_id for sample_id in current_best_sample_id[:subset, class_id] if sample_id not in visited][:per_class]
        class_scores = [current_best_scores[:subset, class_id][j] for j, sample_id in enumerate(current_best_sample_id[:subset, class_id]) if sample_id not in visited][:per_class]
        for sample_id in class_samples:
            visited.add(sample_id)
        assignment[class_id] = class_samples
        scores[class_id] = class_scores



    length = [min(len(np.unique(class_samples)), per_class) for class_samples in assignment.values()]
    plt.plot(sorted(length))
    plt.ylabel('number of samples')
    plt.show()
    plt.savefig("class_distribution.png")

    
    print("Sample to class")

    sample_to_class={}
    # sample_to_score={}
    for k,v in tqdm(assignment.items()):
        for i, sample_id in enumerate(v):
            sample_to_class[sample_id] = k, scores[k][i]
            # sample_to_score[sample_id] = 


    print("Sample to part")
    sample_to_part = {}
    part_to_sample = [[] for i in range(parts)]

    for part in tqdm(range(parts)):   
        for k, cl in assignment.items():
            for sample_id in cl:
                sample_to_part[sample_id] = part
                part_to_sample[part].append(sample_id)


    print("Collecting table")
    table = []




    for part in tqdm(range(parts)):
        # sample_df = all_df[part]
        filename = f'{LAION_LOCATION}/laion400m-meta/part-{part:05d}-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet'
        # df = pd.read_parquet(filename)
        df = pq.read_table(filename).to_pandas()
        sample_df = df.dropna(subset=['TEXT'])


        part_samples = part_to_sample[part]
        picked_df = sample_df[sample_df["SAMPLE_ID"].isin(part_samples)].values.tolist()
        for i in range(len(picked_df)):
            row = picked_df[i].copy()
            sample_id = row[0]
            class_id, score = sample_to_class[sample_id]
            row.append(class_id)
            row.append(score)
            table.append(row)


    df = pd.DataFrame(table, columns = list(df.head().columns) + ["IMAGENET_CLASS", "SCORE"])
    df.to_parquet(f'./parquets/collection_{args.class_num}_classes_{per_class}_samples_{temp_select_type}_tempselect_{args.dataset}_{suffix}.snappy.parquet')
