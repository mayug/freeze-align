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
        "--sort_batch_num",
        dest="sort_batch_num",
        help="batch_size for sorting step",
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

def aggregate(batch_num, sort_batch_num, part):
    df = pd.read_parquet(f'{LAION_LOCATION}/laion400m-meta/part-{part:05d}-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet')
    df_size = len(df)
    for batch_ind in tqdm(range(0, df_size, sort_batch_num)):
        scores = []
        for batch_ind_2 in range(batch_ind, min(batch_ind + sort_batch_num, df_size), batch_num):
            scores_i = torch.load(f'{LAION_LOCATION}/laion400m_scores/embeds_{part}/scores_{batch_ind_2}_{batch_num}.pt', map_location = 'cpu')
            scores.append(scores_i)
        scores = torch.cat(scores, dim = 0)
        torch.save(scores, f'{LAION_LOCATION}/laion400m_scores/embeds_{part}_quant_agg/scores_{batch_ind}_{sort_batch_num}.pt')

if __name__ == "__main__":
    args = parse_args()

    part = args.part
    batch_num = args.batch_num
    sort_batch_num = args.sort_batch_num
    processes = args.processes

    if not os.path.exists(f"{LAION_LOCATION}/laion400m_scores/embeds_{part}_quant_agg/"):
        os.makedirs(f"{LAION_LOCATION}/laion400m_scores/embeds_{part}_quant_agg/")

    # done_paths = [score_file for score_file in os.listdir(f"{LAION_LOCATION}/laion400m_embeds/embeds_{part}_quant_agg/") if "scores" in score_file and "int8" in score_file and f"_{batch_num}.pt" in score_file]
    # scores_paths = [score_file for score_file in os.listdir(f"{LAION_LOCATION}/laion400m_embeds/embeds_{part}_quant/") if "scores" in score_file and "int8" not in score_file and f"_{batch_num}.pt" in score_file and f"int8_{score_file}" not in done_paths]
    # print(f"Done: {len(done_paths)}")
    # print(f"To do: {len(scores_paths)}")
    # process_map(round_tensor, scores_paths, max_workers = processes)

    aggregate(batch_num, sort_batch_num, part)
    