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

LAION_LOCATION = "/shared/laion_data/"
CLASS_NUM = 1000

def parse_args():
    """
    Parse the following arguments for a default parser
    """
    parser = argparse.ArgumentParser(
        description="Getting dataset on dataset segment"
    )
    parser.add_argument(
        "--p",
        dest="parts",
        help="how much parts to do",
        default=5,
        type=int,
    )
    parser.add_argument(
        "--max",
        dest="max_amount",
        help="priority queue size",
        default=50000,
        type=int,
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    parts = args.parts
    max_amount = args.max_amount

    print("Collecting all parquet files")
    all_df = {}
    for part in tqdm(range(parts)):
            
        filename = f'{LAION_LOCATION}/laion400m-meta/part-{part:05d}-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet'
        df = pd.read_parquet(filename)
        df = df.dropna(subset=['TEXT'])
        df_size = df.shape[0]
    
        all_df[part] = df

    
    print("Loading the results")
    current_best_scores = torch.load(f'results/{parts - 1}/scores_so_far_{max_amount}.pt', map_location = "cpu").numpy()
    current_best_sample_id = torch.load(f'results/{parts - 1}/samples_so_far_{max_amount}.pt', map_location = "cpu").numpy()

    print("Asssigning")
    assignment = [[] for i in range(CLASS_NUM)]
    
    class_order = np.argsort(current_best_scores[10000])
    visited = set()
    for class_id in tqdm(class_order):
        class_samples = [sample_id for sample_id in current_best_sample_id[:10000, class_id] if sample_id not in visited][:1000]
        for sample_id in class_samples:
            visited.add(sample_id)
        assignment[class_id] = class_samples

    print("Building parquet file")
    sample_to_class = {}
    for class_id in tqdm(range(CLASS_NUM)):
        for row_id in range(10000):
            sample_id = current_best_sample_id[row_id, class_id]
            score = current_best_scores[row_id, class_id]
    
            if sample_id not in assignment[class_id]:
                continue
                
            sample_to_class[sample_id] = class_id, score


    sample_to_part = {}
    part_to_sample = [[] for i in range(parts)]
    
    for part in tqdm(range(parts)):   
        for cl in assignment:
            for sample_id in cl:
                sample_to_part[sample_id] = part
                part_to_sample[part].append(sample_id)
    
    
    table = []
    for part in tqdm(range(parts)):
        sample_df = all_df[part]
        part_samples = part_to_sample[part]
        picked_df = sample_df[sample_df["SAMPLE_ID"].isin(part_samples)].values.tolist()
        for i in range(len(picked_df)):
            row = picked_df[i].copy()
            sample_id = row[0]
            class_id, score = sample_to_class[sample_id]
            row.append(class_id)
            row.append(score)
            table.append(row)

    df = pd.DataFrame(table, columns = list(all_df[0].head().columns) + ["IMAGENET_CLASS", "SCORE"])
    df.to_parquet(f'collection_{parts}parts.snappy.parquet')

    