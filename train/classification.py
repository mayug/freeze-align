import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
import hydra
from omegaconf import OmegaConf
from pydoc import locate

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from imagenetv2_pytorch import ImageNetV2Dataset
from torchvision import transforms
import imagenet_classes
from PIL import Image

from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from models import build

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer
from torchvision.datasets import ImageNet


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [
        float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy())
        / len(target)
        for k in topk
    ]


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Evaluation:"

    print("Computing features for evaluation...")
    start_time = time.time()

    texts = [f"photo of {_}" for _ in imagenet_classes.names]
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in tqdm(range(0, num_text, text_bs)):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=30,
            return_tensors="pt",
        ).to(device)
        text_output = model.text_encoder(
            text_input.input_ids, attention_mask=text_input.attention_mask, mode="text"
        )
        text_feat = text_output.last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds, dim=0)
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)

    image_feats = []
    image_embeds = []
    labels = []
    for image, label in tqdm(data_loader):
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats.append(image_feat)
        image_embeds.append(image_embed)
        labels.append(label)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)
    labels = torch.cat(labels, dim=0)

    sims_matrix = image_embeds @ text_embeds.t()
    # sims_save_path = Path(config['save_sims']) / 'sim_matrix.npy'
    # sims_save_path.parent.mkdir(parents=True, exist_ok=True)
    # np.save(sims_save_path, sims_matrix.cpu().numpy())
    score_matrix_i2t = torch.full((len(data_loader.dataset), len(texts)), -100.0).to(
        device
    )

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
        score_matrix_i2t[start + i, topk_idx] = topk_sim

    # sims_matrix = sims_matrix.t()
    # score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)

    # step = sims_matrix.size(0)//num_tasks + 1
    # start = rank*step
    # end = min(sims_matrix.size(0),start+step)

    # for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
    #     topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
    #     score_matrix_t2i[start+i,topk_idx] = topk_sim

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        # torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu(), labels.cpu()

# @torch.no_grad()
# def evaluation_token_average(model, data_loader, tokenizer, device,  classnames, config):
#     # this evaluation is for hte token average model
#     # test
#     model.eval()

#     metric_logger = utils.MetricLogger(delimiter="  ")
#     header = "Evaluation:"

#     print("Computing features for evaluation inside evaluation token average ...")
#     start_time = time.time()

#     texts = [f"photo of {_}" for _ in classnames]
#     num_text = len(texts)
#     text_bs = 256
#     text_inputs = []
#     for i in tqdm(range(0, num_text, text_bs)):
#         text = texts[i : min(num_text, i + text_bs)]
#         text_input = tokenizer(
#             text,
#             padding="max_length",
#             truncation=True,
#             max_length=30,
#             return_tensors="pt",
#         ).to(device)

#         text_inputs.append(text_input)
#     # combine text_inputs into one, keys are attention_mask, input_ids, token_type_ids
#     # print(text_input)
#     # print(type(text_input))
#     # asd
#     text_inputs_dict={}
#     for i in text_inputs:
#         for k,v in i.items():
#             if k not in text_inputs_dict:
#                 text_inputs_dict[k]=[]
#             text_inputs_dict[k].append(v)
#     for k,v in text_inputs_dict.items():
#         text_inputs_dict[k]=torch.cat(v,dim=0)
#     for k,v in text_inputs_dict.items():
#         print(k,v.shape)
    
#     image_feats = []
#     text_feats = []
#     # image_embeds = []
#     labels = []
#     for image, label in tqdm(data_loader):
#         image = image.to(device)
#         image_feat, text_feat = model.eval_forward(image, text_inputs_dict)
#         image_feats.append(image_feat)
#         text_feats.append(text_feat)
#         labels.append(label)

#     # print(len(image_feats), len(text_feats))
#     # print(image_feats[0].shape, text_feats[0].shape)
#     # asd

#     image_feats = torch.cat(image_feats, dim=0)
#     text_feats = torch.stack(text_feats, dim=0)
#     # print(text_feats[0].mean(), text_feats[1].mean(), text_feats[2].mean(), text_feats[0].shape)
#     # print(image_feats.shape, text_feats.shape)
#     # asd


#     labels = torch.cat(labels, dim=0)

#     sims_matrix = image_feats @ text_feats[0].t()
#     # sims_save_path = Path(config['save_sims']) / 'sim_matrix.npy'
#     # sims_save_path.parent.mkdir(parents=True, exist_ok=True)
#     # np.save(sims_save_path, sims_matrix.cpu().numpy())
#     score_matrix_i2t = torch.full((len(data_loader.dataset), len(texts)), -100.0).to(
#         device
#     )

#     num_tasks = utils.get_world_size()
#     rank = utils.get_rank()
#     step = sims_matrix.size(0) // num_tasks + 1
#     start = rank * step
#     end = min(sims_matrix.size(0), start + step)

#     for i, sims in enumerate(
#         metric_logger.log_every(sims_matrix[start:end], 50, header)
#     ):
#         topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
#         score_matrix_i2t[start + i, topk_idx] = topk_sim

#     # sims_matrix = sims_matrix.t()
#     # score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)

#     # step = sims_matrix.size(0)//num_tasks + 1
#     # start = rank*step
#     # end = min(sims_matrix.size(0),start+step)

#     # for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
#     #     topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
#     #     score_matrix_t2i[start+i,topk_idx] = topk_sim

#     if args.distributed:
#         dist.barrier()
#         torch.distributed.all_reduce(
#             score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
#         )
#         # torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

#     total_time = time.time() - start_time
#     total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#     print("Evaluation time {}".format(total_time_str))

#     return score_matrix_i2t.cpu(), labels.cpu()


@torch.no_grad()
def evaluation_token_average_fast(model, 
                                  data_loader, 
                                  tokenizer, device, 
                                  config, classnames, args=None):
    # this evaluation is for hte token average model
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Evaluation:"

    print("Computing features for evaluation inside evaluation token average ...")
    start_time = time.time()

    texts = [f"photo of {_}" for _ in classnames]

    num_text = len(texts)
    text_bs = 256
    text_inputs = []
    for i in tqdm(range(0, num_text, text_bs)):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = tokenizer(
            text,
            padding="longest",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        text_inputs.append(text_input)
    # combine text_inputs into one, keys are attention_mask, input_ids, token_type_ids
    # print(text_input)
    # print(type(text_input))
    # asd
    text_inputs_dict={}
    for i in text_inputs:
        for k,v in i.items():
            if k not in text_inputs_dict:
                text_inputs_dict[k]=[]
            text_inputs_dict[k].append(v)
    for k,v in text_inputs_dict.items():
        text_inputs_dict[k]=torch.cat(v,dim=0)


    
    image_feats = []
    text_feats = []
    # image_embeds = []
    labels = []
    for image, label in tqdm(data_loader):
        image = image.to(device)
        image_feat = model.eval_image_forward(image)
        # image_feat, _ = model.eval_forward(image, text_inputs_dict)

        image_feats.append(image_feat)
        
        labels.append(label)


        # if len(labels)==20:
        #     break

    text_feats = model.eval_text_forward(text_inputs_dict)
    # print(len(image_feats), len(text_feats))
    # print(image_feats[0].shape, text_feats[0].shape)
    # asd

    image_feats = torch.cat(image_feats, dim=0)
    # text_feats = torch.stack(text_feats, dim=0)
    # print(text_feats[0].mean(), text_feats[1].mean(), text_feats[2].mean(), text_feats[0].shape)
    # print('image_feats, text_feats ', image_feats.shape, text_feats.shape)
    # asd


    labels = torch.cat(labels, dim=0)

    sims_matrix = image_feats @ text_feats.t()
    # sims_save_path = Path(config['save_sims']) / 'sim_matrix.npy'
    # sims_save_path.parent.mkdir(parents=True, exist_ok=True)
    # np.save(sims_save_path, sims_matrix.cpu().numpy())

    # print('sims_matrix ', sims_matrix.shape)
    score_matrix_i2t = torch.full((len(data_loader.dataset), len(texts)), -100.0).to(
        device
    )

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 25000, header)
    ):
        # print([sims.shape, config["k_test"]])
        topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
        score_matrix_i2t[start + i, topk_idx] = topk_sim

    # sims_matrix = sims_matrix.t()
    # score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)

    # step = sims_matrix.size(0)//num_tasks + 1
    # start = rank*step
    # end = min(sims_matrix.size(0),start+step)

    # for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
    #     topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
    #     score_matrix_t2i[start+i,topk_idx] = topk_sim

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        # torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluation time {}".format(total_time_str))



    return score_matrix_i2t.cpu(), labels.cpu()



@torch.no_grad()
def evaluation_pacl(model, data_loader, 
                                  tokenizer, device, 
                                  config, classnames, args=None):
    # this evaluation is for hte token average model
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Evaluation:"

    print("Computing features for evaluation inside evaluation pacl ...")
    start_time = time.time()

    texts = [f"photo of {_}" for _ in classnames]

    num_text = len(texts)
    text_bs = 256
    text_inputs = []
    for i in tqdm(range(0, num_text, text_bs)):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=30,
            return_tensors="pt",
        ).to(device)

        text_inputs.append(text_input)
    # combine text_inputs into one, keys are attention_mask, input_ids, token_type_ids
    # print(text_input)
    # print(type(text_input))
    # asd
    text_inputs_dict={}
    for i in text_inputs:
        for k,v in i.items():
            if k not in text_inputs_dict:
                text_inputs_dict[k]=[]
            text_inputs_dict[k].append(v)
    for k,v in text_inputs_dict.items():
        text_inputs_dict[k]=torch.cat(v,dim=0)
    for k,v in text_inputs_dict.items():
        print(k,v.shape)

    # asd
    
    sims = []
    labels = []
    for image, label in tqdm(data_loader):
        image = image.to(device)
        sims_curr = model.eval_forward_sims(image, text_inputs_dict)

        sims.append(sims_curr)
        
        labels.append(label)


        # if len(labels)==5:
        #     break
    sims_matrix = torch.cat(sims, dim=0)
    labels = torch.cat(labels, dim=0)
    # print('sims_matrix ', sims_matrix.shape)


    score_matrix_i2t = torch.full((len(data_loader.dataset), len(texts)), -100.0).to(
        device
    )
    # score_matrix_i2t = torch.full((len(labels), len(texts)), -100.0).to(
    #     device
    # )
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 25000, header)
    ):
        # print([sims.shape, config["k_test"]])
        topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
        score_matrix_i2t[start + i, topk_idx] = topk_sim

    # sims_matrix = sims_matrix.t()
    # score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)

    # step = sims_matrix.size(0)//num_tasks + 1
    # start = rank*step
    # end = min(sims_matrix.size(0),start+step)

    # for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
    #     topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
    #     score_matrix_t2i[start+i,topk_idx] = topk_sim

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        # torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu(), labels.cpu()


@torch.no_grad()
def evaluation_token_average_fast_vdt(model, 
                                  data_loader, 
                                  tokenizer, device, 
                                  config, vdt_dict=None, args=None):
    # this evaluation is for hte token average model
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Evaluation:"

    print("Computing features for evaluation inside evaluation token average fast vdt")
    start_time = time.time()

    if vdt_dict is  None:
        texts = [f"photo of {_}" for _ in imagenet_classes.names]
    else: 
        texts = list(vdt_dict.values())
    num_text = len(texts)
    text_bs = 256
    text_inputs = []
    max_length=77
    text_feats = []
    for t in texts:

        if len(t)!=19:
            # print('t ', t)
            # repeat the first item to make it 19
            t = t + [t[0]]*(19-len(t))
            # asd

            
        if 'siglip' in config.get('text_encoder'):
            # padding="max_length", for siglip because it was trained that way
            text_input = tokenizer(
                t,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)
        else:
            text_input = tokenizer(
                t,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)



        

        text_feat = model.eval_text_forward(text_input)
        # print('text_feat ', text_feat.shape)
        # asd
        text_feats.append(text_feat)

    text_feats = torch.stack(text_feats, dim=0)

    # print('text_feats ', text_feats.shape)
    text_feats = text_feats.mean(dim=1)
    # print('text_feats ', text_feats.shape)
    # asd

    
    image_feats = []
    # image_embeds = []
    labels = []
    for image, label in tqdm(data_loader):
        image = image.to(device)
        image_feat = model.eval_image_forward(image)
        image_feats.append(image_feat)
        
        labels.append(label)


        # if len(labels)==5:
        #     break



    # print(len(image_feats), len(text_feats))
    # print(image_feats[0].shape, text_feats[0].shape)
    # asd

    image_feats = torch.cat(image_feats, dim=0)
    # text_feats = torch.stack(text_feats, dim=0)
    # print(text_feats[0].mean(), text_feats[1].mean(), text_feats[2].mean(), text_feats[0].shape)
    print('image_feats, text_feats ', image_feats.shape, text_feats.shape)
    # asd


    labels = torch.cat(labels, dim=0)

    sims_matrix = image_feats @ text_feats.t()
    # sims_save_path = Path(config['save_sims']) / 'sim_matrix.npy'
    # sims_save_path.parent.mkdir(parents=True, exist_ok=True)
    # np.save(sims_save_path, sims_matrix.cpu().numpy())

    # print('sims_matrix ', sims_matrix.shape)
    score_matrix_i2t = torch.full((len(labels), len(texts)), -100.0).to(
        device
    )

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        # print([sims.shape, config["k_test"]])
        topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
        score_matrix_i2t[start + i, topk_idx] = topk_sim

    # sims_matrix = sims_matrix.t()
    # score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)

    # step = sims_matrix.size(0)//num_tasks + 1
    # start = rank*step
    # end = min(sims_matrix.size(0),start+step)

    # for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
    #     topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
    #     score_matrix_t2i[start+i,topk_idx] = topk_sim

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        # torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu(), labels.cpu()


@torch.no_grad()
def evaluation_pacl_vdt(model, 
                        data_loader, 
                        tokenizer, device, 
                        config, vdt_dict, args=None):
    # this evaluation is for hte token average model
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Evaluation:"

    print("Computing features for evaluation inside evaluation pacl ...")
    start_time = time.time()

    texts = list(vdt_dict.values())

    num_text = len(texts)
    text_bs = 256
    text_inputs = []

    max_length=77
    for t in texts:

        if len(t)!=19:
            # print('t ', t)
            # repeat the first item to make it 19
            t = t + [t[0]]*(19-len(t))
            # asd
        text_input = tokenizer(
            t,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        text_inputs.append(text_input)

    print('text_inputs ', len(text_inputs))
    print('text_inputs ', text_inputs[0]['input_ids'].shape)
    # asd

    # combine text_inputs into one, keys are attention_mask, input_ids, token_type_ids
    # print(text_input)
    # print(type(text_input))
    # asd


    # text_inputs_dict={}
    # for i in text_inputs:
    #     for k,v in i.items():
    #         if k not in text_inputs_dict:
    #             text_inputs_dict[k]=[]
    #         text_inputs_dict[k].append(v)
    # for k,v in text_inputs_dict.items():
    #     text_inputs_dict[k]=torch.cat(v,dim=0)
    # for k,v in text_inputs_dict.items():
    #     print(k,v.shape)

    text = text_inputs
    print(len(text))
    # get pooled and projected text embeddings
    print('calculating text features for vdt')
    text_feats = []
    for t in tqdm(text):
        _, text_embeds, text_pooled_output = model._forward_(None, t, return_dict=False, return_pooled_output=True)
        _, text_feat, _, text_embeds = model._get_features_(None, text_embeds, t, 
                                                            text_pooled_output=text_pooled_output)
        text_feats.append(text_feat)

    text_feats = torch.stack(text_feats)
    text_feat = text_feats.mean(dim=1)
    print('text_feat', [text_feat.shape, text_feat.min(), text_feat.max()])

    
    sims = []
    labels = []
    for image, label in tqdm(data_loader):
        image = image.to(device)
        sims_curr = model.eval_forward_sims_vdt(image, text_feat)

        sims.append(sims_curr)
        
        labels.append(label)


        # if len(labels)==5:
        #     break
    sims_matrix = torch.cat(sims, dim=0)
    labels = torch.cat(labels, dim=0)
    # print('sims_matrix ', sims_matrix.shape)


    score_matrix_i2t = torch.full((len(data_loader.dataset), len(texts)), -100.0).to(
        device
    )
    # score_matrix_i2t = torch.full((len(labels), len(texts)), -100.0).to(
    #     device
    # )
    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 25000, header)
    ):
        # print([sims.shape, config["k_test"]])
        topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
        score_matrix_i2t[start + i, topk_idx] = topk_sim

    # sims_matrix = sims_matrix.t()
    # score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)

    # step = sims_matrix.size(0)//num_tasks + 1
    # start = rank*step
    # end = min(sims_matrix.size(0),start+step)

    # for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
    #     topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
    #     score_matrix_t2i[start+i,topk_idx] = topk_sim

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        # torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu(), labels.cpu()

@torch.no_grad()
def evaluation_cross(model, data_loader, tokenizer, device, config):
    # this evaluation is for hte token average model
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Evaluation:"

    print("Computing features for evaluation inside evaluation cross ...")
    start_time = time.time()

    texts = [f"photo of {_}" for _ in imagenet_classes.names]
    num_text = len(texts)
    text_bs = 256
    text_inputs = []
    for i in tqdm(range(0, num_text, text_bs)):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=30,
            return_tensors="pt",
        ).to(device)

        text_inputs.append(text_input)
    # combine text_inputs into one, keys are attention_mask, input_ids, token_type_ids
    # print(text_input)
    # print(type(text_input))
    # asd
        
    text_inputs_dict={}
    for i in text_inputs:
        for k,v in i.items():
            if k not in text_inputs_dict:
                text_inputs_dict[k]=[]
            text_inputs_dict[k].append(v)
    for k,v in text_inputs_dict.items():
        text_inputs_dict[k]=torch.cat(v,dim=0)
    for k,v in text_inputs_dict.items():
        print(k,v.shape)

    
    
    image_feats = []
    text_feats = []
    labels = []
    

    # expand by 1000 inside the eval_forward, fold into batchsize and do the forward
    sims = []
    bs = data_loader.batch_size
    for image, label in tqdm(data_loader):
        # image = image.unsqueeze(0)
        
        image = image.to(device)

        image_feat, text_feat = model.eval_forward(image, text_inputs_dict)

        # print(image_feat.shape, text_feat.shape)
        # asd
        text_feat = text_feat.permute(1, 0, 2) # now image dimension is first and text is second
        # image_feat = image_feat.unsqueeze(1).repeat(1, text_feat.shape[1], 1) # repeat image features for each text feature

        # print('image_feat ', image_feat.shape)
        sims_temp = [F.cosine_similarity(image_feat[i,:].unsqueeze(0), text_feat[i]) for i in range(bs)]

        # sim_i2t = torch.bmm(image_feat, text_feat.permute(0, 2, 1)) 
        sim_i2t = torch.stack(sims_temp, dim=0)


        # print('sim_i2t ', sim_i2t.shape)
        # print('sim_t2i ', sim_t2i.shape)
        
        # sim_i2t = sim_i2t.max(dim=-1).values

        # print('sim_i2t ', sim_i2t.shape)
        # asd

        sims.append(sim_i2t)
        labels.append(label)
        # print(sims[0].shape)
        # print(sims)
        # asd

        # if len(sims)==100:
        #     break
        # image_feats.append(image_feat)
        # text_feats.append(text_feat)


    # print(len(image_feats), len(text_feats))
    # print(image_feats[0].shape, text_feats[0].shape)
    # print('sims len ', len(sims))
    # asd

    # image_feats = torch.cat(image_feats, dim=0)
    # text_feats = torch.stack(text_feats, dim=0)
    # print(text_feats[0].mean(), text_feats[1].mean(), text_feats[2].mean(), text_feats[0].shape)
    # print(image_feats.shape, text_feats.shape)
    # asd


    labels = torch.cat(labels, dim=0)

    sims_matrix = torch.cat(sims, dim=0)
    print('sims_matrix', sims_matrix.shape)
    print(len(labels))
    # asd
    # sims_matrix = image_feats @ text_feats[0].t()
    
    # sims_save_path = Path(config['save_sims']) / 'sim_matrix.npy'
    # sims_save_path.parent.mkdir(parents=True, exist_ok=True)
    # np.save(sims_save_path, sims_matrix.cpu().numpy())
    score_matrix_i2t = torch.full((len(data_loader.dataset), len(texts)), -100.0).to(
        device
    )

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
        score_matrix_i2t[start + i, topk_idx] = topk_sim

    # sims_matrix = sims_matrix.t()
    # score_matrix_t2i = torch.full((len(texts),len(data_loader.dataset.image)),-100.0).to(device)

    # step = sims_matrix.size(0)//num_tasks + 1
    # start = rank*step
    # end = min(sims_matrix.size(0),start+step)

    # for i,sims in enumerate(metric_logger.log_every(sims_matrix[start:end], 50, header)):
    #     topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
    #     score_matrix_t2i[start+i,topk_idx] = topk_sim

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        # torch.distributed.all_reduce(score_matrix_t2i, op=torch.distributed.ReduceOp.SUM)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu(), labels.cpu()

@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, txt2img, img2txt):

    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        rank = 1e20
        for i in img2txt[index]:
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank

    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])

    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]
        ranks[index] = np.where(inds == txt2img[index])[0][0]

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {
        "txt_r1": tr1,
        "txt_r5": tr5,
        "txt_r10": tr10,
        "txt_r_mean": tr_mean,
        "img_r1": ir1,
        "img_r5": ir5,
        "img_r10": ir10,
        "img_r_mean": ir_mean,
        "r_mean": r_mean,
    }
    return eval_result


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (config["image_res"], config["image_res"]), interpolation=Image.BICUBIC
            ),
            transforms.ToTensor(),
            normalize,
        ]
    )

    if args.use_vdt_augmentation:
        vdt_dict = torch.load('./imagenet_vdt+wordnet.pt')
    else:
        vdt_dict = None


    #### Dataset ####
    print("Creating retrieval dataset")
    # train_dataset, val_dataset, test_dataset = create_dataset('re', config)

    if args.dataset == "imagenetv2":
        print("Using ImageNetV2 dataset")
        test_dataset = ImageNetV2Dataset(
            location=config['imagenetv2_path'], transform=test_transform
        )
    elif args.dataset == "imagenet":
        print("Using ImageNet dataset")
        test_dataset = ImageNet(
            root=config['imagenet_path'], split="val", transform=test_transform
        )

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = [None, None, None]
    else:
        samplers = [None, None, None]

    _, _, test_loader = create_loader(
        [test_dataset, test_dataset, test_dataset],
        samplers,
        batch_size=[config["batch_size_train"]] + [config["batch_size_test"]] * 2,
        num_workers=[4, 4, 4],
        is_trains=[True, False, False],
        collate_fns=[None, None, None],
    )

    if config.version == 1:
        tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    elif config.version > 1:
        tokenizer = build.tokenizer(config)

    #### Model ####
    print("Creating model")
    model_class = locate(config.model_config.import_path)
    model = model_class(
        config=config, text_encoder=args.text_encoder, tokenizer=tokenizer
    )

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint["model"]

        # reshape positional embedding to accomodate for image resolution change
        # pos_embed_reshaped = interpolate_pos_embed(
        #     state_dict["visual_encoder.pos_embed"], model.visual_encoder
        # )
        # state_dict["visual_encoder.pos_embed"] = pos_embed_reshaped
        required_keys = model.state_dict().keys()
        state_dict = {k: v for k, v in state_dict.items() if k in required_keys}
        # for key in list(state_dict.keys()):
        #     if 'bert' in key:
        #         encoder_key = key.replace('bert.','')
        #         state_dict[encoder_key] = state_dict[key]
        #         del state_dict[key]
        msg = model.load_state_dict(state_dict, strict=True)

        print("load checkpoint from %s" % args.checkpoint)
        print(msg)

    model = model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    arg_opt = utils.AttrDict(config["optimizer"])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config["schedular"])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    max_epoch = config["schedular"]["epochs"]
    warmup_steps = config["schedular"]["warmup_epochs"]

    start_time = time.time()
    for epoch in range(0, max_epoch):

        # score_test_i2t, labels = evaluation(
        #     model_without_ddp, test_loader, tokenizer, device, config
        # )

        # score_test_i2t, labels = evaluation_cross(
        #     model_without_ddp, test_loader, tokenizer, device, config
        # )

        # score_test_i2t, labels = evaluation_token_average(
        #     model_without_ddp, test_loader, tokenizer, device, config
        # )


        if args.use_vdt_augmentation:
                
            score_test_i2t, labels = evaluation_token_average_fast_vdt(
                model_without_ddp, test_loader, tokenizer, device, config,
                vdt_dict=vdt_dict, args=args
            
            )


                
            # score_test_i2t, labels = evaluation_pacl_vdt(
            #     model_without_ddp, test_loader, tokenizer, device, config,
            #     vdt_dict=vdt_dict, args=args
            
            # )

        else:
            score_test_i2t, labels = evaluation_pacl(
                model_without_ddp, test_loader, tokenizer, device, config,
                args=args
            
            )

            score_test_i2t, labels = evaluation_token_average_fast(
                model_without_ddp, test_loader, tokenizer, device, config, args=args
            
            )


        if utils.is_main_process():
            ranks = (1, 5, 10)
            test_result = accuracy(score_test_i2t, labels, topk=ranks)
            print({f"Rank@{r}": v * 100 for r, v in zip(ranks, test_result)})

            # test_result = itm_eval(score_test_i2t, score_test_t2i, test_loader.dataset.txt2img, test_loader.dataset.img2txt)
            # print(test_result)

            # if args.evaluate:
            #     log_stats = {
            #                  **{f'test_{k}': v for k, v in test_result.items()},
            #                  'epoch': epoch,
            #                 }
            #     with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            #         f.write(json.dumps(log_stats) + "\n")
            # else:
            #     log_stats = {
            #                  **{f'test_{k}': v for k, v in test_result.items()},
            #                  'epoch': epoch,
            #                 }
            #     with open(os.path.join(args.output_dir, "log.txt"),"a") as f:
            #         f.write(json.dumps(log_stats) + "\n")

        if args.evaluate:
            break

        lr_scheduler.step(epoch + warmup_steps + 1)
        dist.barrier()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Time {}".format(total_time_str))


def get_config(args):
        with hydra.initialize(config_path="./configs-v2"):
            config = hydra.compose(config_name=args.config)
        return config


def merge_model_config(config, config_model):
    config.text_encoder = config_model.text_encoder
    config.vision_encoder = config_model.vision_encoder

    config.local_vision_projection = config_model.local_vision_projection

    config.cls_vision_projection = config_model.cls_vision_projection
    config.vis_pooling = config_model.vis_pooling
    config.text_pooling = config_model.text_pooling
    config.text_projection = config_model.text_projection
    config.local_text_projection = config_model.local_text_projection

    config.image_res = config_model.image_res


    config.model_config = config_model.model_config

    return config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Retrieval_AdjustableCLIP_Flickr_cross")
    parser.add_argument("--output_dir", default="./clf_output/")
    parser.add_argument("--checkpoint", default="./storage/lilt_cache/proj_cross_testing1/checkpoint_01.pth")
    parser.add_argument("--text_encoder", default="bert-base-uncased")
    parser.add_argument("--evaluate", default=True, action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--distributed", default=True, type=bool)
    parser.add_argument("--overrides", nargs="+", default=[])
    parser.add_argument("--dataset", default="imagenetv2", type=str)
    parser.add_argument("--use_vdt_augmentation", default=False, type=bool)

    parser.add_argument('--use_checkpoint_config', action='store_true', help='use model config file from checkpoint')

    args = parser.parse_args()

    with hydra.initialize(config_path="./configs-v2"):
        config = hydra.compose(config_name=args.config, overrides=args.overrides)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # load the model config from checkpoint
    if args.use_checkpoint_config:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        config_model = checkpoint["config"]
        config = merge_model_config(config, config_model)

    print(config)

    yaml.dump(
        OmegaConf.to_object(config),
        open(os.path.join(args.output_dir, "config.yaml"), "w"),
    )

    print("Running with config:\n{}".format(config))

    main(args, config)
