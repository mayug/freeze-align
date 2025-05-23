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

from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from models import build

import utils
from dataset import create_dataset, create_sampler, create_loader
from scheduler import create_scheduler
from optim import create_optimizer


@torch.no_grad()
def evaluation(model, data_loader, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Evaluation:"

    print("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
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
    for image, img_id in tqdm(data_loader):
        image = image.to(device)
        image_feat = model.visual_encoder(image)
        image_embed = model.vision_proj(image_feat[:, 0, :])
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats.append(image_feat)
        image_embeds.append(image_embed)

    image_feats = torch.cat(image_feats, dim=0)
    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = image_embeds @ text_embeds.t()
    # sims_save_path = Path(config['save_sims']) / 'sim_matrix.npy'
    # sims_save_path.parent.mkdir(parents=True, exist_ok=True)
    # np.save(sims_save_path, sims_matrix.cpu().numpy())
    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(device)

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

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
        score_matrix_t2i[start + i, topk_idx] = topk_sim

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()

@torch.no_grad()
def evaluation_token_average(model, data_loader, tokenizer, device, config):
    # test
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Evaluation:"

    print("Computing features for evaluation...")
    start_time = time.time()

    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 32
    text_embeds = []
    max_length= 77
    print('number of text ', num_text)
    for i in tqdm(range(0, num_text, text_bs)):
        text = texts[i : min(num_text, i + text_bs)]
        # text_input = tokenizer(
        #     text,
        #     padding="max_length",
        #     truncation=True,
        #     max_length=max_length,
        #     return_tensors="pt",
        # ).to(device)

        if 'siglip' in config.get('text_encoder'):
            # padding="max_length", for siglip because it was trained that way
            text_input = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).to(device)
        else:
            text_input = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)

        # print(text_input.input_ids.shape)
        # print(text_input.attention_mask.shape)
        # # asd
        text_embed = model.eval_text_forward(text_input)
        # print(text_embed.shape)
        # asd
        text_embeds.append(text_embed.detach())
    text_embeds = torch.cat(text_embeds, dim=0)
    # asd
    image_embeds = []
    for image, img_id in tqdm(data_loader):
        image = image.to(device)
        image_embed = model.eval_image_forward(image)
        image_embeds.append(image_embed.detach())

    image_embeds = torch.cat(image_embeds, dim=0)

    sims_matrix = image_embeds @ text_embeds.t()
    # sims_save_path = Path(config['save_sims']) / 'sim_matrix.npy'
    # sims_save_path.parent.mkdir(parents=True, exist_ok=True)
    # np.save(sims_save_path, sims_matrix.cpu().numpy())
    score_matrix_i2t = torch.full(
        (len(data_loader.dataset.image), len(texts)), -100.0
    ).to(device)

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

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full(
        (len(texts), len(data_loader.dataset.image)), -100.0
    ).to(device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(
        metric_logger.log_every(sims_matrix[start:end], 50, header)
    ):
        topk_sim, topk_idx = sims.topk(k=config["k_test"], dim=0)
        score_matrix_t2i[start + i, topk_idx] = topk_sim

    if args.distributed:
        dist.barrier()
        torch.distributed.all_reduce(
            score_matrix_i2t, op=torch.distributed.ReduceOp.SUM
        )
        torch.distributed.all_reduce(
            score_matrix_t2i, op=torch.distributed.ReduceOp.SUM
        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Evaluation time {}".format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy()


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

    #### Dataset ####
    print("Creating retrieval dataset")
    train_dataset, val_dataset, test_dataset = create_dataset("re", config)

    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        samplers = create_sampler([train_dataset], [True], num_tasks, global_rank) + [
            None,
            None,
        ]
    else:
        samplers = [None, None, None]

    train_loader, val_loader, test_loader = create_loader(
        [train_dataset, val_dataset, test_dataset],
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

        score_test_i2t, score_test_t2i = evaluation_token_average(
            model_without_ddp, test_loader, tokenizer, device, config
        )

        if utils.is_main_process():

            test_result = itm_eval(
                score_test_i2t,
                score_test_t2i,
                test_loader.dataset.txt2img,
                test_loader.dataset.img2txt,
            )
            print(test_result)

            if args.evaluate:
                log_stats = {
                    **{f"test_{k}": v for k, v in test_result.items()},
                    "epoch": epoch,
                }
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                log_stats = {
                    **{f"test_{k}": v for k, v in test_result.items()},
                    "epoch": epoch,
                }
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        if args.evaluate:
            break

        lr_scheduler.step(epoch + warmup_steps + 1)
        dist.barrier()
        torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Time {}".format(total_time_str))


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
    parser.add_argument("--config", default="./configs/Retrieval_flickr.yaml")
    parser.add_argument("--output_dir", default="output/Retrieval_flickr")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--text_encoder", default="bert-base-uncased")
    parser.add_argument("--evaluate", action="store_true")
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
    # asd

    yaml.dump(
        OmegaConf.to_object(config),
        open(os.path.join(args.output_dir, "config.yaml"), "w"),
    )

    main(args, config)
