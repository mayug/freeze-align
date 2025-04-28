from transformers import CLIPProcessor, CLIPModel
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
import argparse

import torch, random
from tqdm import tqdm
import json
import os

LAION_LOCATION = '/notebooks/data/'

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
        "--m",
        dest="model",
        help="model",
        default="clip",
        type=str,
    )
    parser.add_argument(
        "--p",
        dest="part",
        help="part",
        default=0,
        type=int,
    )
    return parser.parse_args()

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def run_file(file_num, batch_num):
    filename = f'{LAION_LOCATION}/laion400m-meta/part-{file_num:05d}-5b54c5d5-bbcf-484d-a2ce-0d6f73df1a36-c000.snappy.parquet'
    df = pd.read_parquet(filename)
    df = df.dropna(subset=['TEXT'])
    df_size = df.shape[0]

    result = []

    for batch_ind in tqdm(list(range(done_batches * batch_num, df_size, batch_num))):
    # for batch_ind in tqdm(list(range(0, 202000, batch_num))):

        batch = df.iloc[batch_ind: min(batch_ind + batch_num, df_size)]
        batch_embeds = run_batch(batch)
        torch.save(batch_embeds, f'{LAION_LOCATION}/laion400m_embeds/embeds_{part}/batch_{batch_ind}_{batch_num}.pt')

def run_batch(batch):
    batch_texts = batch["TEXT"].tolist()
    sample_ids = batch["SAMPLE_ID"].tolist()

    if model_name == "allroberta":
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt').to(device)
        with torch.no_grad():
            model_output = model(**encoded_input)
        text_representation = mean_pooling(model_output, encoded_input['attention_mask'])
        text_representation = text_representation / torch.norm(text_representation, dim = 1, keepdim = True)
    elif model_name == "clip":
        inputs = processor(text=batch_texts, return_tensors="pt", padding=True, truncation = True)
        inputs = inputs.to(device)
        with torch.no_grad():
            outputs = model.get_text_features(**inputs)
        text_representation = outputs
        text_representation = text_representation / torch.norm(text_representation, dim = 1, keepdim = True)
    return text_representation
    
if __name__ == "__main__":
    args = parse_args()

    gpu = f'cuda:{args.gpu}'
    device = torch.device(gpu)
    torch.cuda.empty_cache()

    part = args.part
    model_name = args.model
    batch_num = args.batch_num

    if not os.path.isdir(f'{LAION_LOCATION}/laion400m_embeds'):
        os.mkdir(f'{LAION_LOCATION}/laion400m_embeds')

    if not os.path.isdir(f'{LAION_LOCATION}/laion400m_embeds/embeds_{part}'):
        os.mkdir(f'{LAION_LOCATION}/laion400m_embeds/embeds_{part}') 

    done_batches = len([file for file in os.listdir(f'{LAION_LOCATION}/laion400m_embeds/embeds_{part}') if "batch" in file])

    if model_name == "allroberta":
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-roberta-large-v1')
        model = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1').to(device)
    elif model_name == "clip":
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        
    result = run_file(part, batch_num)
