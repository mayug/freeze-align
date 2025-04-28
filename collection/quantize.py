import argparse
import pandas as pd
import os
from tqdm import tqdm
import torch
from tqdm.contrib.concurrent import process_map

LAION_LOCATION = '/notebooks/data'

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
        dest="part",
        help="which part to do",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--batch_num",
        dest="batch_num",
        help="batch_size",
        default=50000,
        type=int,
    )
    parser.add_argument(
        "--proc",
        dest="processes",
        help="multithreading processes",
        default=10,
        type=int,
    )
    return parser.parse_args()

def float_to_int(x):
    return (x * 128).to(torch.int8)

def int_to_float(x):
    return (x / 128)

def round_tensor(score_f):
    scores = torch.load(f"{LAION_LOCATION}/laion400m_embeds/embeds_{part}/{score_f}", map_location = "cpu")
    torch.save(float_to_int(scores), f"{LAION_LOCATION}/laion400m_embeds/embeds_{part}_quant/int8_{score_f}")
        

if __name__ == "__main__":
    args = parse_args()

    part = args.part
    batch_num = args.batch_num
    processes = args.processes

    if not os.path.exists(f"{LAION_LOCATION}/laion400m_embeds/embeds_{part}_quant/"):
        os.makedirs(f"{LAION_LOCATION}/laion400m_embeds/embeds_{part}_quant/")

    done_paths = [score_file for score_file in os.listdir(f"{LAION_LOCATION}/laion400m_embeds/embeds_{part}_quant/") if "scores" in score_file and "int8" in score_file and f"_{batch_num}.pt" in score_file]
    scores_paths = [score_file for score_file in os.listdir(f"{LAION_LOCATION}/laion400m_embeds/embeds_{part}/") if "scores" in score_file and "int8" not in score_file and f"_{batch_num}.pt" in score_file and f"int8_{score_file}" not in done_paths]
    print(f"Done: {len(done_paths)}")
    print(f"To do: {len(scores_paths)}")
    process_map(round_tensor, scores_paths, max_workers = processes)
    