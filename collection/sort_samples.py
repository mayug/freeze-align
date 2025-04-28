import argparse
import pandas as pd
import os
import multiprocessing
from multiprocessing import Queue as multi_queue
import time
from tqdm import tqdm
import heapq
import torch
import pickle

LAION_LOCATION = '/notebooks/data/'
CLASS_NUM = 2754

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
        "--max",
        dest="max_amount",
        help="priority queue size",
        default=50000,
        type=int,
    )
    parser.add_argument(
        "--b",
        dest="batch_num",
        help="batch_size",
        default=4096,
        type=int,
    )
    parser.add_argument(
        "--sort_b",
        dest="sort_batch_num",
        help="batch_size for sorting; this has to be high for efficiency",
        default=65536,
        type=int,
    )
    parser.add_argument(
        "--gpu",
        dest="gpu",
        help="gpu",
        default=0,
        type=int,
    )
    return parser.parse_args()

def float_to_int(x):
    return (x * 128).to(torch.int8)

def int_to_float(x):
    return (x / 128)
        

if __name__ == "__main__":
    args = parse_args()

    gpu = f'cuda:{args.gpu}'
    device = torch.device(gpu)
    torch.cuda.empty_cache()

    part = args.part
    batch_num = args.batch_num
    max_amount = args.max_amount
    sort_batch_num = args.sort_batch_num

    print("Started loading the parquet file")
    filename = f'{LAION_LOCATION}/laion400m-meta/part-{part:05d}-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet'
    df = pd.read_parquet(filename)
    df = df.dropna(subset=['TEXT'])
    df_size = df.shape[0]
    print("Loaded the parque file")

    os.makedirs("results", exist_ok = True)

    if str(part) not in os.listdir("results/"):
        os.mkdir(f"results/{part}")

    if os.path.isfile(f"results/{part}/current_batch_{batch_num}_{max_amount}.pt"):
        starting_ind = torch.load(f"results/{part}/current_batch_{batch_num}_{max_amount}.pt")
    else:
        starting_ind = 0

    if os.path.isfile(f'results/{part}/scores_so_far_{max_amount}.pt'):
        current_best_scores = torch.load(f'results/{part}/scores_so_far_{max_amount}.pt', map_location = device)
        current_best_sample_id = torch.load(f'results/{part}/samples_so_far_{max_amount}.pt', map_location = device)
    elif os.path.isfile(f'results/{part-1}/scores_so_far_{max_amount}.pt'):
        current_best_scores = torch.load(f'results/{part-1}/scores_so_far_{max_amount}.pt', map_location = device)
        current_best_sample_id = torch.load(f'results/{part-1}/samples_so_far_{max_amount}.pt', map_location = device)
    else:
        current_best_scores = torch.tensor([[-2 for i in range(CLASS_NUM)]]).to(device)
        current_best_sample_id = torch.tensor([[-2 for i in range(CLASS_NUM)]]).to(device)


    print("Running batches")
    print(f"BAtches left in this part: {len(list(range(starting_ind, df_size, sort_batch_num)))}")



    for batch_ind in tqdm(list(range(starting_ind, df_size, sort_batch_num))):
        
        batch = df.iloc[batch_ind: min(batch_ind + sort_batch_num, df_size)]
        # print([batch_ind, sort_batch_num, batch.shape])
        # asd
        sample_ids = torch.tensor(batch["SAMPLE_ID"].tolist(), dtype = torch.float64).to(device)
        
        if sort_batch_num== batch_num:
            scores = torch.load(f'{LAION_LOCATION}/laion400m_scores/embeds_{part}/scores_{batch_ind}_{batch_num}.pt', map_location = device)
        else:
            # aggregate till sort_batch_num
            scores = []
            for batch_ind_2 in range(batch_ind, min(batch_ind + sort_batch_num, df_size), batch_num):
                try:
                    scores_i = torch.load(f'{LAION_LOCATION}/laion400m_scores/embeds_{part}/scores_{batch_ind_2}_{batch_num}.pt', map_location = device)
                    scores.append(scores_i)
                except:
                    scores.append(-2*torch.ones((batch_num,CLASS_NUM)).to(device))
            scores = torch.cat(scores, dim = 0)
            # print("Aggregated scores")
            # print([scores.shape, scores.min(), scores.max()])
            # asd

    

        print(scores.min(),scores.max(), scores.sum())
        scores = int_to_float(scores)
        print(scores.min(),scores.max(), scores.sum())
        print("####################")
        remove_nans = torch.isnan(sample_ids) == False
        sample_ids = sample_ids[remove_nans]
        scores = scores[remove_nans]
        
        sorted_scores, indices = torch.sort(-torch.cat([current_best_scores, scores]), dim = 0)
        sorted_scores = -sorted_scores[:max_amount]
        # print(sorted_scores.shape)
        # print(torch.where(sorted_scores > 0.9, 1.0,0.0).sum()
        indices = indices[:max_amount]
        
        prev_sample_ids = (indices < current_best_scores.shape[0]) * torch.gather(current_best_sample_id, 0, (indices < current_best_scores.shape[0]) * indices)
        current_sample_ids = (indices >= current_best_scores.shape[0]) * sample_ids[(indices >= current_best_scores.shape[0]) * (indices - current_best_scores.shape[0])]
        
        current_best_sample_id = prev_sample_ids + current_sample_ids
        current_best_scores = sorted_scores
        print(sorted_scores.min(), sorted_scores.max())

    
    
    
    torch.save(current_best_scores.cpu(), f'results/{part}/scores_so_far_{max_amount}.pt')
    torch.save(current_best_sample_id.cpu(), f'results/{part}/samples_so_far_{max_amount}.pt')
    torch.save(batch_ind + batch_num, f"results/{part}/current_batch_{sort_batch_num}_{max_amount}.pt")
