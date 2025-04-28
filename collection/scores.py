import pandas as pd
import numpy as np
import argparse

import torch, random
from tqdm import tqdm
import json
import os

LAION_LOCATION = '/notebooks/data'

def parse_args():
    """
    Parse the following arguments for a default parser
    """
    parser = argparse.ArgumentParser(
        description="Running experiments"
    )
    parser.add_argument(
        "--gpu",
        dest="gpu",
        help="gpu",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--b",
        dest="batch_num",
        help="batch_num",
        default=10000,
        type=int,
    )
    parser.add_argument(
        "--p",
        dest="part",
        help="part",
        default=0,
        type=int,
    )
    return parser.parse_args()

def run(part, batch_num):

    for batch_ind in tqdm(list(range(done_batches * batch_num, df_size, batch_num))):
    # for batch_ind in tqdm(list(range(0, df_size, batch_num))):
        batch_embeds = torch.load(f'{LAION_LOCATION}/laion400m_embeds/embeds_{part}/batch_{batch_ind}_{batch_num}.pt', map_location = device)
        scores = torch.matmul(batch_embeds, imagenet_classes.T).cpu()
        torch.save(scores, f'{LAION_LOCATION}/laion400m_embeds/embeds_{part}/scores_{batch_ind}_{batch_num}.pt')
        
if __name__ == "__main__":
    args = parse_args()

    gpu = f'cuda:{args.gpu}'
    device = torch.device(gpu)
    torch.cuda.empty_cache()

    part = args.part
    batch_num = args.batch_num

    filename = f'{LAION_LOCATION}/laion400m-meta/part-{part:05d}-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet'
    df = pd.read_parquet(filename)
    df = df.dropna(subset=['TEXT'])
    df_size = df.shape[0]

    done_batches = len([file for file in os.listdir(f'{LAION_LOCATION}/laion400m_embeds/embeds_{part}') if "scores" in file and f'_{batch_num}.pt' in file])
    print("Done batches", done_batches)
    imagenet_classes = torch.load(f'class_embeddings/label_to_clip_class.pt', map_location='cuda:0').to(device)  
    
    result = run(part, batch_num)
    