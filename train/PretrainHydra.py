"""
 * Copyright (c) 2021, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path
from pydoc import locate

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import wandb
import hydra
from omegaconf import OmegaConf
from torch.cuda.amp import autocast

from models.vit import interpolate_pos_embed
from models import build
from models.tokenization_bert import BertTokenizer

import utils
from dataset import create_dataset, create_sampler, create_loader, create_wds_loader
from scheduler import create_scheduler
from optim import create_optimizer
from dataset.utils import collate_safe
from torchvision.datasets import ImageNet
from classification import evaluation_token_average_fast, accuracy
from torchvision import transforms
from PIL import Image
from pydoc import locate



def train(
    model,
    data_loader,
    optimizer,
    tokenizer,
    epoch,
    warmup_steps,
    device,
    scheduler,
    config,
    image_tokenizer,
    wandb_logger=None,
    gradient_accumulation_steps=1,
):

    global_loss_weight = config.global_loss_weight
    local_loss_weight = config.local_loss_weight

    # train
    scaler = torch.cuda.amp.GradScaler()
    model.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter(
        "lr", utils.SmoothedValue(window_size=50, fmt="{value:.6f}")
    )

    meters_added = False

    header = "Train Epoch: [{}]".format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    if args.distributed and isinstance(data_loader, torch.utils.data.DataLoader):
        data_loader.sampler.set_epoch(epoch)

    for i, (image, text) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):

        text = [t if isinstance(t, str) else "This is a photo" for t in text] 
        

        image = image.to(device, non_blocking=True)

        if 'siglip'in config.text_encoder.lower():
            # only padding=max_length works for siglip because of the way the model is trained
            text_input = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        ).to(device)
        
        else:
            text_input = tokenizer(
                text, padding="longest", truncation=True, max_length=77, return_tensors="pt"
            ).to(device)



        with autocast(enabled=config.fp16):
            model_output = model(
                image=image, text=text_input, return_dict=True
            )

            if not meters_added:
                for loss_name, _ in model_output["losses"].items():
                    metric_logger.add_meter(
                        loss_name,
                        utils.SmoothedValue(window_size=50, fmt="{value:.4f}"),
                    )
                meters_added = True

            loss = global_loss_weight * model_output["losses"]["loss_ita"] + local_loss_weight * model_output["losses"]["loss_fg"]
            

        scaler.scale(loss/gradient_accumulation_steps).backward()
        grad_norm = utils.calculate_gradient_norm(model)
        grad_norm_ln_img, grad_norm_ln_txt = utils.calculate_gradient_norms_for_ln(
            model.module
        )
        (
            grad_norm_nonln_img,
            grad_norm_nonln_txt,
        ) = utils.calculate_gradient_norms_for_non_ln(model.module)

        if (i + 1) % gradient_accumulation_steps == 0:

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        if i % print_freq == 0:
            if utils.is_main_process() and wandb_logger:
                wandb_logger.log(
                    data={
                        **{
                            loss_name: loss_value.item()
                            for loss_name, loss_value in model_output["losses"].items()
                        },
                        "grad_norm/all": grad_norm,
                        "grad_norm/ln/vit": grad_norm_ln_img,
                        "grad_norm/ln/bert": grad_norm_ln_txt,
                        "grad_norm/non_ln/vit": grad_norm_nonln_img,
                        "grad_norm/non_ln/bert": grad_norm_nonln_txt,
                        "lr": optimizer.param_groups[0]["lr"],
                    }
                )
                # log gradients of all parameters with gradient
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        wandb_logger.log(
                            data={
                                f"grad/{name}": param.grad.norm().item(),
                                f"param/{name}": param.norm().item(),
                            }
                        )

        for loss_name, loss_value in model_output["losses"].items():
            # Turn it into a dictionary first, because the meter
            # asks for the loss updates to be specified as keyword args.
            metric_logger.update(**{loss_name: loss_value.item()})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)
        # break
        

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())
    return {
        k: "{:.3f}".format(meter.global_avg)
        for k, meter in metric_logger.meters.items()
    }


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config["schedular"]["epochs"]
    warmup_steps = config["schedular"]["warmup_epochs"]

    eval_function = locate(config.classification_config.import_path)

    #### Dataset ####
    print("Creating dataset")

    if config.trainset not in ["cc3m", "laion800k", "laion5m"]:

        datasets = [create_dataset("pretrain", config)]
        if config.limit_num_samples:
            datasets[0].truncate_to(config.limit_num_samples)

        if args.distributed:
            num_tasks = utils.get_world_size()
            global_rank = utils.get_rank()
            samplers = create_sampler(datasets, [True], num_tasks, global_rank)

        else:
            samplers = [None]


        data_loader = create_loader(
            datasets,
            samplers,
            batch_size=[config["batch_size"]],
            num_workers=[16],
            is_trains=[True],
            collate_fns=[collate_safe],
        )[0]
    else:
    
            data_loader = create_wds_loader(config)


    # test loader
    samplers = [None, None, None]

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

    test_dataset = ImageNet(root=config["imagenet_path"],
                            split="val", transform=test_transform
    )
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
        config=config,
        tokenizer=tokenizer,
    )

    disable_wandb = config.get("disable_wandb", False)  # Enable by default.
    if utils.is_main_process() and not disable_wandb:
        print("Is main process, creating W&B logger.")
        print(os.path.basename(args.output_dir))
        # print(args.output_dir.split('/')[-2])
        # asd
        wandb_logger = wandb.init(
            project="vision-language-alignment",
            entity="stoic",
            name=args.output_dir.split('/')[-2],
            config=OmegaConf.to_container(config),
        )
        wandb_logger.watch(model, log_graph=False)
    else:
        wandb_logger = None

    model = model.to(device)

    arg_opt = utils.AttrDict(config["optimizer"])
    optimizer = create_optimizer(arg_opt, model)
    arg_sche = utils.AttrDict(config["schedular"])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = checkpoint["model"]
        if args.resume:
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            start_epoch = checkpoint["epoch"] + 1
        else:
            pos_embed_reshaped = interpolate_pos_embed(
                state_dict["visual_encoder.pos_embed"], model.visual_encoder
            )
            # m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],model.visual_encoder_m)
            state_dict["visual_encoder.pos_embed"] = pos_embed_reshaped
            # state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
        model.load_state_dict(state_dict, strict=config.load_strict)
        print("load checkpoint from %s" % args.checkpoint)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    image_tokenizer = None

    print("Start training")
    start_time = time.time()

    # start_epoch = 0 # remove when resuming training Important   
    for epoch in range(start_epoch, max_epoch):

        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        train_stats = train(
            model,
            data_loader,
            optimizer,
            tokenizer,
            epoch,
            warmup_steps,
            device,
            lr_scheduler,
            config,
            image_tokenizer,
            wandb_logger=wandb_logger,
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        )
        if utils.is_main_process():
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
            save_obj = {
                "model": model_without_ddp.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "config": config,
                "epoch": epoch,
            }
            if config.get("save_last_only", False):
                if epoch == max_epoch - 1:
                    torch.save(
                        save_obj,
                        os.path.join(args.output_dir, "checkpoint_%02d.pth" % epoch),
                    )
            else:
                torch.save(
                    save_obj,
                    os.path.join(args.output_dir, "checkpoint_%02d.pth" % epoch),
                )

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        dist.barrier()


        if epoch%2==0 or epoch==max_epoch-1:
            # run imagenet 
            kwargs = {'args':args}
            if 'vdt' in config.classification_config.import_path:
                vdt_dict = torch.load('./imagenet_vdt+wordnet.pt')
                kwargs['vdt_dict'] = vdt_dict
            score_test_i2t, labels = eval_function(
                model_without_ddp, test_loader, tokenizer, device, config, **kwargs
            )
            if utils.is_main_process():
                ranks = (1, 5, 10)
                test_result = accuracy(score_test_i2t, labels, topk=ranks)
                print({f"Rank@{r}": v * 100 for r, v in zip(ranks, test_result)})
                with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps({"Rank@{r}": v * 100 for r, v in zip(ranks, test_result)}) + "\n")



    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dinov2-arl")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--resume", default=False, type=bool)
    parser.add_argument("--output_dir", default="./storage/lilt_cache/dinov2-arl-testing")
    parser.add_argument("--text_encoder", default="bert-base-uncased")
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
    args = parser.parse_args()

    with hydra.initialize(config_path="./configs-v2"):
        config = hydra.compose(config_name=args.config, overrides=args.overrides)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(
        OmegaConf.to_object(config),
        open(os.path.join(args.output_dir, "config.yaml"), "w"),
    )

    print('Running with config: ', args.config)
    print(config)

    main(args, config)
