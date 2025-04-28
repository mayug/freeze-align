from torchvision.datasets import VOCSegmentation
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from pydoc import locate
from torchvision import transforms


from models.vit import interpolate_pos_embed
from models.tokenization_bert import BertTokenizer
from models import build
from notebook_utils import return_config

from PIL import Image
import argparse
import hydra
import cv2
from tqdm import tqdm




def get_model(config, args):
    if config.version == 1:
        tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    elif config.version > 1:
        tokenizer = build.tokenizer(config)


    device = torch.device(args.device)

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

    model.eval()

    return model, tokenizer, device


class_dict = {
    'background': 0,
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20,
    # 'ignore':255
}

class_dict_inv = {v:k for k,v in class_dict.items()}

img_mean = (0.48145466, 0.4578275, 0.40821073)
img_std = (0.26862954, 0.26130258, 0.27577711)

def get_dataset(config):

    dataset = VOCSegmentation('/notebooks/data/pascal_voc/', year='2012', image_set='val', download=True)


    normalize = transforms.Normalize(
        img_mean, img_std
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

    test_transform_wo_norm = transforms.Compose(
        [
            transforms.Resize(
                (config["image_res"], config["image_res"]), interpolation=Image.NEAREST
            ),
            # transforms.ToTensor(),
        ]
    )

    return dataset, test_transform, test_transform_wo_norm

def get_image(img):
    # undo normalization
    img = img *img_std + img_mean

    # clip to [0, 1]
    img = img.clip(0, 1)
    return img

def resize_target(target, size):
    target = cv2.resize(target, size, interpolation=cv2.INTER_LINEAR)
    return target

import numpy as np

def calculate_iou(mask1, mask2):
    # Ensure the masks are boolean arrays
    mask1_bool = mask1.astype(bool)
    mask2_bool = mask2.astype(bool)
    
    # Calculate intersection and union
    intersection = np.logical_and(mask1_bool, mask2_bool)
    union = np.logical_or(mask1_bool, mask2_bool)
    
    # Compute IoU
    iou = np.sum(intersection) / np.sum(union)
    
    return iou


def model_forward(img, cls_name, model, tokenizer, device):
    model.eval()
    with torch.no_grad():
        img  = img.to('cuda')

        if 'siglip' in config.text_encoder:
            text_input = tokenizer(
            cls_name,
            padding="max_length",
            return_tensors="pt",
        ).to(device)
        else:
            text_input = tokenizer(
            cls_name,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        ).to(device)

        image_embeds, text_embeds, text_pooled_output = model._forward_(img.unsqueeze(0), text_input, return_pooled_output=model.return_pooled_output)
        image_feat, text_feat, image_embeds, text_embeds = model._get_features_(image_embeds, text_embeds, text_input, text_pooled_output=text_pooled_output)

        # apply global projection too
 
        image_embeds = model.vision_proj(image_embeds)

        text_output = F.normalize(text_feat, p=2, dim=-1)

        spatial_tokens = image_embeds[:, 1:]

        vision_mean = image_feat
        vision_mean = F.normalize(vision_mean, p=2, dim=-1)


        spatial_sims = spatial_tokens @ text_output.T



        return spatial_sims
    



def get_iou(model, tokenizer, device, dataset, test_transform, test_transform_wo_norm):
    # use all ground truth classes for evaluation


    threshold = 0.4




    instance_ious = []


    cls_ious = {k:[] for k in list(class_dict_inv.values())}

    for i in tqdm(range(len(dataset))):
        # i=100
        img = dataset[i][0]
        
        target = dataset[i][1]



        target = test_transform_wo_norm(target)
        target = np.array(target)
        target[target==255]=0

        cls_name = list(class_dict_inv.values())


        img = test_transform(img)

        spatial_sims = model_forward(img, cls_name , model, tokenizer, device)
        spatial_sims = spatial_sims.squeeze().cpu().numpy()

        ious = []
        for i in range(1, len(cls_name)):
            spatial_sims_current = spatial_sims[:,i]

            spatial_sims_current = spatial_sims_current.reshape(18,18) 
            spatial_sims_resized = resize_target(spatial_sims_current, (256,256))

            # spatial_sims_resized = F.sigmoid(torch.Tensor(spatial_sims_resized)).numpy()
            spatial_sims_resized = (spatial_sims_resized-spatial_sims_resized.min()) / (spatial_sims_resized.max()-spatial_sims_resized.min())

            # print([spatial_sims_resized.shape, spatial_sims_resized.min(), spatial_sims_resized.max()])

            spatial_sims_resized = spatial_sims_resized > threshold



            # calculate the iou only for non background classes
            iou = calculate_iou(resize_target(target, (256,256))==i, spatial_sims_resized)

            
            
            # add to cls_ious only if the class is present in the target
            if i in np.unique(target):
                cls_ious[class_dict_inv[i]].append(iou)
                ious.append(iou)


        
            # asd
        instance_ious.append(np.mean(ious))

    temp = []
    for k, v in cls_ious.items():
        print([k, np.mean(v)])
        if np.mean(v) != np.nan:
            temp.append(np.mean(v))

    

    return np.mean(instance_ious), np.mean(temp[1:])

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="Retrieval_AdjustableCLIP_Flickr_cross")
    parser.add_argument("--checkpoint", default="./storage/lilt_cache/proj_cross_testing1/checkpoint_01.pth")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=42, type=int)


    parser.add_argument("--overrides", nargs="+", default=[])
    parser.add_argument("--dataset", default="pascalvoc", type=str)
    parser.add_argument("--use_vdt_augmentation", default=False, type=bool)
    args = parser.parse_args([])



    config = return_config(args)

    model, tokenizer, device = get_model(config, args)
    dataset, test_transform, test_transform_wo_norm = get_dataset(config)

    print(get_iou(model, tokenizer, device, dataset, test_transform, test_transform_wo_norm))

# python zero_shot_segmentation.py --config Retrieval_AdjustableCLIP_Flickr_cross --checkpoint ./storage/lilt_cache/proj_cross_testing1/checkpoint_01.pth --device cuda --seed 42
