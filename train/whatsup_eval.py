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

from whatsup_vlms.dataset_zoo import get_dataset
from whatsup_vlms.misc import seed_all, _default_collate, save_scores
from whatsup_vlms.model_zoo.clip_models import CLIPWrapper

from torchvision import transforms
from PIL import Image



class LILTWrapper(CLIPWrapper):
    def __init__(self, model, tokenizer,  device):
        
        self.tokenizer = tokenizer
        super().__init__(model, device)

    @torch.no_grad()
    def get_text_embeddings(self, texts, text_batch_size=256, normalize=False):
        num_text = len(texts)
        text_embeds = []
        print('number of text ', num_text)
        for i in tqdm(range(0, num_text, text_batch_size)):
            text = texts[i : min(num_text, i + text_batch_size)]
            text_input = self.tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)
            # print(text_input.input_ids.shape)
            # print(text_input.attention_mask.shape)
            # # asd
            text_embed = self.model.eval_text_forward(text_input)
            # print(text_embed.shape)
            # asd
            text_embeds.append(text_embed.detach())
        text_embeds = torch.cat(text_embeds, dim=0)
        return text_embeds

    def tokenize(self, text):
        return self.tokenizer(text, padding="max_length", truncation=True, max_length=30, return_tensors="pt")

    @torch.no_grad()
    def get_image_embeddings(self, images, image_batch_size=256, normalize=False):
        image_embeds = []
        for image in tqdm(images):
            image = image["image"].to(self.device)
            image_embed = self.model.eval_image_forward(image)
            image_embeds.append(image_embed.detach())

        image_embeds = torch.cat(image_embeds, dim=0)
        return image_embeds
    
    @torch.no_grad()
    def get_retrieval_scores_batched(self, joint_loader):
        """Computes the scores for each image_option / caption_option pair in the joint loader.

        Args:
            joint_loader (DataLoader): batches have "image_options" and "caption_options" fields.
            "image_options" is a list of images, and "caption_options" is a list of captions.

        Returns:
            all_scores: A numpy array containing the scores of the shape NxKxL,
            where N is the number of test cases, K is the number of image options per the test case,
            and L is the number of caption options per the test case.
        """
        scores = []
        tqdm_loader = tqdm(joint_loader)
        tqdm_loader.set_description("Computing retrieval scores")
        for batch in tqdm_loader:
            image_options = []
            for i_option in batch["image_options"]:
                image_embeddings = self.model.eval_image_forward(i_option.to(self.device)).cpu().numpy() # B x D
                image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True) # B x D
                image_options.append(np.expand_dims(image_embeddings, axis=1))
            
            caption_options = []
            # print(len(batch["caption_options"]))

            for c_option in batch["caption_options"]:

                caption_tokenized = [[self.tokenize(c) for c in c_option]]

                # print([len(caption_tokenized), len(caption_tokenized[0])])
                # for c in caption_tokenized[0]:
                #     # print(c)
                #     print(c['input_ids'].shape)
                #     print(c['attention_mask'].shape)
                text_inputs_dict={}
                for i in caption_tokenized[0]:
                    for k,v in i.items():
                        if k not in text_inputs_dict:
                            text_inputs_dict[k]=[]
                        text_inputs_dict[k].append(v)
                for k,v in text_inputs_dict.items():
                    text_inputs_dict[k]=torch.cat(v,dim=0).to(self.device)

                # for k,v in text_inputs_dict.items():
                #     print(k,v.shape)
                # asd

                caption_embeddings = self.model.eval_text_forward(text_inputs_dict).cpu().numpy() # B x D
                caption_embeddings = caption_embeddings / np.linalg.norm(caption_embeddings, axis=1, keepdims=True) # B x D

                # print('caption_embeddings', caption_embeddings.shape)

                caption_options.append(np.expand_dims(caption_embeddings, axis=1))
                
            image_options = np.concatenate(image_options, axis=1) # B x K x D
            caption_options = np.concatenate(caption_options, axis=1) # B x L x D
            batch_scores = np.einsum("nkd,nld->nkl", image_options, caption_options) # B x K x L
            scores.append(batch_scores)
        
        all_scores = np.concatenate(scores, axis=0) # N x K x L
        return all_scores
        




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
    print('number of text ', num_text)
    for i in tqdm(range(0, num_text, text_bs)):
        text = texts[i : min(num_text, i + text_bs)]
        text_input = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=30,
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


def get_preprocess(config):

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )

    if 'dinov2' in config.vision_encoder:
        normalize = transforms.Normalize( (0.485,
    0.456,
    0.406), (  0.229,
    0.224,
    0.225))
    test_transform = transforms.Compose(
    [
        transforms.Resize(
            (config["image_res"], config["image_res"]), interpolation=Image.BICUBIC
        ),
        transforms.ToTensor(),
        normalize,
    ]
    )

    return test_transform

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
    print("Creating whatsup dataset")
    test_transform = get_preprocess(config)
    test_dataset  = get_dataset(args.dataset, image_preprocess=test_transform)
    
    collate_fn = _default_collate if test_transform is None else None

    joint_loader =  DataLoader(test_dataset,
                                batch_size=64, 
                                shuffle=False, num_workers=8, collate_fn=collate_fn)
     
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


    # Liltwrapper for whatsup

    model = LILTWrapper(model_without_ddp, tokenizer, device)

    scores = model.get_retrieval_scores_batched(joint_loader)
    result_records = test_dataset.evaluate_scores(scores)

    print(result_records)
    # asd


    #### Evaluation ####
        

    # arg_opt = utils.AttrDict(config["optimizer"])
    # optimizer = create_optimizer(arg_opt, model)
    # arg_sche = utils.AttrDict(config["schedular"])
    # lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    # max_epoch = config["schedular"]["epochs"]
    # warmup_steps = config["schedular"]["warmup_epochs"]

    # start_time = time.time()
    # for epoch in range(0, max_epoch):

    #     score_test_i2t, score_test_t2i = evaluation_token_average(
    #         model_without_ddp, joint_loader, tokenizer, device, config
    #     )

    #     if utils.is_main_process():

    #         test_result = itm_eval(
    #             score_test_i2t,
    #             score_test_t2i,
    #             test_loader.dataset.txt2img,
    #             test_loader.dataset.img2txt,
    #         )
    #         print(test_result)

    #         if args.evaluate:
    #             log_stats = {
    #                 **{f"test_{k}": v for k, v in test_result.items()},
    #                 "epoch": epoch,
    #             }
    #             with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
    #                 f.write(json.dumps(log_stats) + "\n")
    #         else:
    #             log_stats = {
    #                 **{f"test_{k}": v for k, v in test_result.items()},
    #                 "epoch": epoch,
    #             }
    #             with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
    #                 f.write(json.dumps(log_stats) + "\n")

    #     if args.evaluate:
    #         break

    #     lr_scheduler.step(epoch + warmup_steps + 1)
    #     dist.barrier()
    #     torch.cuda.empty_cache()

    # total_time = time.time() - start_time
    # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    # print("Time {}".format(total_time_str))


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
    parser.add_argument("--dataset", default="Controlled_Images_A", type=str)

    parser.add_argument("--overrides", nargs="+", default=[])
    args = parser.parse_args()

    with hydra.initialize(config_path="./configs-v2"):
        config = hydra.compose(config_name=args.config, overrides=args.overrides)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(
        OmegaConf.to_object(config),
        open(os.path.join(args.output_dir, "config.yaml"), "w"),
    )

    main(args, config)
