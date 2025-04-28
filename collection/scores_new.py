import pandas as pd
import numpy as np
import argparse

import torch, random
from tqdm import tqdm
import json
import os
from glob import glob

LAION_LOCATION = "/notebooks/data/"

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
        default=4096,
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

def float_to_int(x):
    max_b = 0.4
    min_b = -0.3
    if x.max() > max_b or x.min() < min_b:
        print(x.min(),x.max())
        print("went wrong")
        x[x.max() > max_b]= max_b
        x[x.min() < min_b] = min_b 
    x = (x-min_b)/(max_b-min_b)
    x= 2*x-1
    return (x * 128).to(torch.int8)


def run(part, batch_num, imagenet_classes):
    
    for batch_ind in tqdm(list(range(done_batches * batch_num, df_size, batch_num))):
        # try:
        batch_embeds = torch.load(f'{LAION_LOCATION}/laion400m_embeds/embeds_{part}/batch_{batch_ind}_{batch_num}.pt', map_location = device).half()
        batch_embeds = batch_embeds/torch.norm(batch_embeds, dim=1, keepdim=True)
        imagenet_classes = imagenet_classes/torch.norm(imagenet_classes, dim=1, keepdim=True)
        scores = torch.matmul(batch_embeds, imagenet_classes.T).cpu()
        # print(scores.min(),scores.max())
        scores = float_to_int(scores)
        # print(scores.min(),scores.max())
        torch.save(scores, f'{LAION_LOCATION}/laion400m_scores/embeds_{part}/scores_{batch_ind}_{batch_num}.pt')
        # except:
        #     pass

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

    os.makedirs(f'{LAION_LOCATION}/laion400m_scores/embeds_{part}/', exist_ok=True)
    
    done_batches = len([file for file in os.listdir(f'{LAION_LOCATION}/laion400m_scores/embeds_{part}') if "scores" in file and f'_{batch_num}.pt' in file])
    print("Done batches", done_batches)
    imagenet_classes = torch.load(f'class_embeddings/imgprotos/combined_14datasets_imgprotos_128_shot.pt', map_location=device)
    imagenet_classes = torch.tensor(np.stack(list(imagenet_classes.values()))).to(device).half()
    print(imagenet_classes.shape)
    done_batches=0
    result = run(part, batch_num, imagenet_classes)
    