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
from dataset.cub_dataset import Cub2011
from scheduler import create_scheduler
from optim import create_optimizer
from torchvision.datasets import ImageNet, ImageFolder
import torchvision
from classification import accuracy, itm_eval, evaluation_pacl, evaluation_pacl_vdt

from einops import rearrange, reduce, repeat
from pydoc import locate


def get_dataset(args, test_transform):
    if args.dataset == "imagenetv2":
        print("Using ImageNetV2 dataset")
        test_dataset = ImageNetV2Dataset(
            location="/notebooks/data/imagenetv2/ImageNetV2/imagenetv2-matched-frequency-format-val/", 
            transform=test_transform
        )
        prompts = imagenet_classes.names

    elif args.dataset == "imagenet":
        print("Using ImageNet dataset")
        test_dataset = ImageNet(
            root="/FEAT/data/imagenet/imagenet_new/", split="val", transform=test_transform
        )
        prompts = imagenet_classes.names

    elif args.dataset == "caltech-101":
        print("Using Caltech101 dataset")
        test_dataset = torchvision.datasets.Caltech101("/FEAT/data/caltech-101/", transform = test_transform, download = True)
        prompts = test_dataset.categories

    elif args.dataset == 'oxford_pets':
        print("Using OxfordIIITPet dataset")
        test_dataset = torchvision.datasets.OxfordIIITPet("/FEAT/data/oxford_pets/", 
                                                          transform = test_transform, download = True, split = "test")
        prompts = test_dataset.classes

    elif args.dataset == 'stanford_cars':
        print("Using StanfordCars dataset")
        test_dataset = torchvision.datasets.StanfordCars("/FEAT/data/", 
                                         transform = test_transform, 
                                         split = "test", download=False)
        prompts = test_dataset.classes

    elif args.dataset == 'oxford_flowers':
        print("Using OxfordFlowers dataset")
        test_dataset = torchvision.datasets.Flowers102("/FEAT/data/", 
                                                       transform = test_transform, 
                                                       download = True, split = "test")
        
        prompts = torch.load('/notebooks/LilT/vdt_dicts/labels/flowers_labels_ordered.pt')

    elif args.dataset == 'food-101':
        print("Using Food101 dataset")
        test_dataset = torchvision.datasets.Food101("/FEAT/data/", transform = test_transform, download = True, split = "test")
        prompts = [f.replace("_"," ") for f in test_dataset.classes]

    elif args.dataset == 'fgvc_aircraft':
        print("Using FGVCAircraft dataset")
        test_dataset = torchvision.datasets.FGVCAircraft("/FEAT/data/", transform = test_transform, download = True, split = "test")
        prompts = test_dataset.classes

    elif args.dataset == 'sun397':
        print("Using SUN397 dataset")
        test_dataset = torchvision.datasets.SUN397('/FEAT/data/', test_transform, download=False)
        prompts = test_dataset.classes
    elif args.dataset == 'dtd':
        print("Using DescribableTextures dataset")
        test_dataset = torchvision.datasets.DTD("/FEAT/data/", transform = test_transform, download = False, split = 'test')
        prompts = test_dataset.classes

    elif args.dataset == 'cub':
        test_dataset = Cub2011(root = "/FEAT/data/CUB/", train = False, transform = test_transform, download=False)
        prompts = torch.load('/FEAT/data/CUB/classnames_ordered.pt')
    elif args.dataset == 'eurosat':
        test_dataset = torchvision.datasets.EuroSAT('/FEAT/data/eurosat/alternate/', 
                                       test_transform,
                                       download=False)
        prompts = test_dataset.classes
    elif args.dataset == 'ucf101':
        test_dataset = ImageFolder('/FEAT/data/ucf101//UCF-101-midframes',
                                test_transform)
        prompts = test_dataset.classes
    # to setup cub, eurosat, ucf-101, 
    return test_dataset, prompts

def get_vdt_dict(dataset):
    if 'imagenet' in dataset:
        vdt_dict = torch.load('./vdt_dicts/transformed/imagenet_vdt+wordnet.pt')
    else:
        vdt_dict = torch.load(f'./vdt_dicts/transformed/{dataset}.pt')
    return vdt_dict

def main(args, config):

    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    


    def channel_check(x):
        if x.shape[0] == 3:
            return x
        #print("got here")
        return repeat(x, "c h w -> (c rep) h w", rep = 3)

    normalize = transforms.Normalize(
        (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
    )
    test_transform = transforms.Compose(
        [
            transforms.Resize(
                (config["image_res"], config["image_res"]), interpolation=Image.BICUBIC
            ),
            transforms.ToTensor(),
            channel_check,
            normalize,
        ]
    )

    if args.use_vdt_augmentation:
        vdt_dict = get_vdt_dict(args.dataset)
        eval_function = locate(config.classification_config.import_path_vdt)
    else:
        eval_function = locate(config.classification_config.import_path)
        vdt_dict = None


    #### Dataset ####
    print("Creating retrieval dataset")
    # train_dataset, val_dataset, test_dataset = create_dataset('re', config)


    test_dataset, prompts = get_dataset(args, test_transform)   

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


        if args.use_vdt_augmentation:

            score_test_i2t, labels = eval_function(
                model_without_ddp, test_loader, tokenizer, device, config,
                vdt_dict=vdt_dict, args=args)
                        
            # score_test_i2t, labels = evaluation_token_average_fast_vdt(
            #     model_without_ddp, test_loader, tokenizer, device, config,
            #     vdt_dict=vdt_dict, args=args
            
            # )

            
        else:
            score_test_i2t, labels = eval_function(
                model_without_ddp, test_loader, tokenizer, device, config, classnames=prompts,
                args=args
            
            )
            # score_test_i2t, labels = evaluation_token_average_fast(
            #     model_without_ddp, test_loader, tokenizer, device, config, class_names=prompts, args=args
            
            # )

        print('Dataset:', args.dataset)

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
    args = parser.parse_args()

    with hydra.initialize(config_path="./configs-v2"):
        config = hydra.compose(config_name=args.config, overrides=args.overrides)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(
        OmegaConf.to_object(config),
        open(os.path.join(args.output_dir, "config.yaml"), "w"),
    )

    print("Running with config:\n{}".format(config))

    main(args, config)
