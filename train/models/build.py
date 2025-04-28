from omegaconf import DictConfig, OmegaConf
from models.med import BertModel, BertConfig
from models.tokenization_bert import BertTokenizer
from torch import nn
from functools import partial
from models.vit import VisionTransformer, interpolate_pos_embed
from models.sentence_transformer_train import SentenceTransformer
from models.deit_v2 import vit_models, Layer_scale_init_Block
from models.med import BertLayer
import torch
from transformers import AutoImageProcessor, AutoModel
from transformers import AutoTokenizer, AutoConfig
from models.dinov2 import Dinov2Model
from models.roberta import RobertaModel
from models.clip import CLIPTextModel, CLIPVisionModel
from open_clip import create_model_from_pretrained, get_tokenizer 
import timm
# from sentence_transformer_train import SentenceTransformer_train

# from transformers import ViTModel

_BERT_CONFIG_MAP = {
    "large": "princeton-nlp/unsup-simcse-bert-large-uncased",
    "base": "princeton-nlp/unsup-simcse-bert-base-uncased",
    "base_mlm": "bert-base-uncased",
    "small": "prajjwal1/bert-small",
    "tiny": "prajjwal1/bert-tiny",
    "base_multilingual": "bert-base-multilingual-cased",
}


def tokenizer(config: DictConfig):
    print('config.text_encoder', config.text_encoder)
    if 'sentence-transformers' in config.text_encoder:
        tokenizer = AutoTokenizer.from_pretrained(config.text_encoder)
        return tokenizer
    
    if 'clip' in config.text_encoder:
        tokenizer = AutoTokenizer.from_pretrained(config.text_encoder)
        return tokenizer
    elif 'siglip' in config.text_encoder.lower():

        tokenizer = AutoTokenizer.from_pretrained(config.text_encoder)
        return tokenizer

    try:
        return BertTokenizer.from_pretrained(_BERT_CONFIG_MAP[config.text_encoder])
    except KeyError:
        raise ValueError(f"Unknown text encoder: {config.text_encoder}")

def Identity(x):
    return x

def text_encoder(config: DictConfig, text_encoder: str, adapter_append: bool):
    # if 'all-roberta-large-v1' in text_encoder:
    #     # raise ValueError(f"Unknown text encoder: {text_encoder}")

    #     language_model = AutoModel.from_pretrained('sentence-transformers/all-roberta-large-v1')
    #     return language_model

    if config.pretrained_text:
        if 'sentence-transformers' in text_encoder:
            try:
                bert_config = AutoConfig.from_pretrained(text_encoder)
                # Some options (adapter related) are specified in the configuration file and
                # are put into effect by cascading those options into the model configuration,
                # which is a `BertConfig` object.defined by the `transformers` library.
                if adapter_append:
                    bert_config.num_hidden_layers += 1
                if config.conventional_adapter.insert in ("language_only", True):
                    bert_config.reduction_factor = (
                        config.conventional_adapter.reduction_factor
                    )
                    bert_config.insert_adapter = True
                else:
                    bert_config.insert_adapter = False
            except KeyError as exc:
                raise ValueError(f"Unknown text encoder: {text_encoder}") from exc
            else:
                # print(RobertaModel.from_pretrained(
                #     config.text_encoder,
                #     config=bert_config,
                #     add_pooling_layer=False,
                # ))
                # asd

                if 'roberta' in text_encoder:

                    return RobertaModel.from_pretrained(
                        config.text_encoder,
                        config=bert_config,
                        add_pooling_layer=False,
                    )
                else:
                    print(f'loading sentence-transformers model {text_encoder} using AutoModel')
                    a = AutoModel.from_pretrained(text_encoder)
                    return a
        elif 'clip' in text_encoder:
            try:
                bert_config = AutoConfig.from_pretrained(text_encoder)
                bert_config = bert_config.text_config
                # Some options (adapter related) are specified in the configuration file and
                # are put into effect by cascading those options into the model configuration,
                # which is a `BertConfig` object.defined by the `transformers` library.
                if adapter_append:
                    bert_config.num_hidden_layers += 1
                if config.conventional_adapter.insert in ("language_only", True):
                    bert_config.reduction_factor = (
                        config.conventional_adapter.reduction_factor
                    )
                    bert_config.insert_adapter = True
                else:
                    bert_config.insert_adapter = False
            except KeyError as exc:
                raise ValueError(f"Unknown text encoder: {text_encoder}") from exc
            else:

                # print(CLIPTextModel.from_pretrained(
                #     text_encoder,
                #     config=bert_config,
                # ))

                # asd
                a = CLIPTextModel.from_pretrained(
                    text_encoder,
                    config=bert_config,
                )

                return a
        elif 'siglip' in text_encoder.lower():
            
            # huggingface siglip
            a = AutoModel.from_pretrained(text_encoder).text_model

            return a
        else:
            try:
                bert_config = BertConfig.from_pretrained(
                    _BERT_CONFIG_MAP[config.text_encoder]
                )
                # Some options (adapter related) are specified in the configuration file and
                # are put into effect by cascading those options into the model configuration,
                # which is a `BertConfig` object.defined by the `transformers` library.
                if adapter_append:
                    bert_config.num_hidden_layers += 1
                if config.conventional_adapter.insert in ("language_only", True):
                    bert_config.reduction_factor = (
                        config.conventional_adapter.reduction_factor
                    )
                    bert_config.insert_adapter = True
                else:
                    bert_config.insert_adapter = False
            except KeyError as exc:
                raise ValueError(f"Unknown text encoder: {text_encoder}") from exc
            else:
                return BertModel.from_pretrained(
                    _BERT_CONFIG_MAP[config.text_encoder],
                    config=bert_config,
                    add_pooling_layer=False,
                )

    else:
        try:
            bert_config = BertConfig.from_pretrained(_BERT_CONFIG_MAP[text_encoder])
            # We set all adapter related args to null, because there's no point
            # (right now) in using adapters with a randomly initialized model.
            bert_config.insert_adapter = False
            bert_config.reduction_factor = None  # Doesn't matter.
        except KeyError as exc:
            raise ValueError(f"Unknown text encoder: {text_encoder}") from exc
        else:
            return BertModel(bert_config, add_pooling_layer=False)


def vision_encoder(config: DictConfig, vision_encoder: str, adapter_append: bool):
    # All ViT models except the large have the same depth.
    depth = 12 if vision_encoder in ("tiny", "small", "base", "base_dino") else 24
    if adapter_append:
        depth += 1

    pretrained = config.pretrained_vision
    adapter_config = OmegaConf.create(config.conventional_adapter)
    # The `hidden_size` attribute of the config will be filled by
    # the specific model constructor method (e.g. deit_small_patch16_224).
    # This is because the ViT models don't have a config object that gets
    # passed to all the blocks.
    should_insert_adapter = config.conventional_adapter.insert in ("vision_only", True)
    conventional_adapter_kwargs = {
        "insert_adapter": should_insert_adapter,
        "adapter_config": adapter_config,
    }
    if vision_encoder == "base":
        model = deit_base_patch16_224(
            pretrained=pretrained,
            image_res=config.image_res,
            mask_token=False,
            depth=depth,
            conventional_adapter_kwargs=conventional_adapter_kwargs,
        )
    elif vision_encoder == "small":
        model = deit_small_patch16_224(
            pretrained=pretrained,
            image_res=config.image_res,
            mask_token=False,
            depth=depth,
            conventional_adapter_kwargs=conventional_adapter_kwargs,
        )
    elif vision_encoder == "tiny":
        model = deit_tiny_patch16_224(
            pretrained=pretrained,
            image_res=config.image_res,
            mask_token=False,
            depth=depth,
            conventional_adapter_kwargs=conventional_adapter_kwargs,
        )
    elif vision_encoder == "large":
        model = deit_large_patch16_LS(
            pretrained=pretrained,
            img_size=config.image_res,
            depth=depth,
            conventional_adapter_kwargs=conventional_adapter_kwargs,
        )
    elif vision_encoder == "base_dino":
        model = dino_base_patch16(
            pretrained=pretrained, image_res=config.image_res, depth=depth
        )
    elif 'dinov2' in vision_encoder:
        processor = AutoImageProcessor.from_pretrained(vision_encoder)
        dino_config = AutoConfig.from_pretrained(vision_encoder)
        if config.conventional_adapter.insert in ("vision_only", True):
            dino_config.insert_adapter = True
            dino_config.reduction_factor = config.conventional_adapter.reduction_factor
        else:
            dino_config.insert_adapter = False
        model = Dinov2Model.from_pretrained(vision_encoder, config=dino_config)
        model = [processor, model]
    elif 'clip' in vision_encoder:

        processor = AutoImageProcessor.from_pretrained(vision_encoder)
        dino_config = AutoConfig.from_pretrained(vision_encoder)
        dino_config.vision_config.insert_adapter = False
        model = CLIPVisionModel.from_pretrained(vision_encoder, config=dino_config.vision_config)

        model = [processor, model]
    elif 'timm' in vision_encoder:
        model = timm.create_model(vision_encoder, pretrained=pretrained)
        data_config = timm.data.resolve_model_data_config(model)
        transforms_obj = timm.data.create_transform(**data_config, is_training=False)
        model = [transforms_obj, model]
    
    elif 'convnext' in vision_encoder:
        model = AutoModel.from_pretrained(vision_encoder)
        processor = AutoImageProcessor.from_pretrained(vision_encoder)

        model = [processor, model]
    elif 'dino' in vision_encoder:
        model = AutoModel.from_pretrained(vision_encoder)
        processor = AutoImageProcessor.from_pretrained(vision_encoder)

        model = [processor, model]
    else:
        raise ValueError(f"Unknown vision encoder: {vision_encoder}")

    if config.conventional_adapter.insert:
        if 'dinov2' in vision_encoder:
            model[1].adapter_config = OmegaConf.create(config.conventional_adapter)
            model[1].adapter_config.hidden_size = model[1].embed_dim
            model[1].insert_adapter = True
        else:
            model.adapter_config = OmegaConf.create(config.conventional_adapter)
            model.adapter_config.hidden_size = model.embed_dim
            model.insert_adapter = True

    return model


def adapter_kwargs_for_vit(kwargs, embed_dim):
    adapter_kwargs = kwargs.get("conventional_adapter_kwargs")
    if adapter_kwargs is not None:
        adapter_kwargs["adapter_config"].hidden_size = embed_dim
    return adapter_kwargs


def deit_tiny_patch16_224(
    pretrained=False, image_res=224, mask_token=False, depth: int = 12, **kwargs
):
    embed_dim = 192
    adapter_kwargs = adapter_kwargs_for_vit(kwargs, embed_dim)
    model = VisionTransformer(
        img_size=image_res,
        mask_token=mask_token,
        patch_size=16,
        embed_dim=192,
        depth=depth,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **adapter_kwargs,
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu",
            check_hash=True,
        )
        state_dict = checkpoint["model"]
        pos_embed_reshaped = interpolate_pos_embed(state_dict["pos_embed"], model)
        state_dict["pos_embed"] = pos_embed_reshaped
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model


def deit_small_patch16_224(
    pretrained=False, image_res=224, mask_token=False, depth: int = 12, **kwargs
):
    embed_dim = 384
    adapter_kwargs = adapter_kwargs_for_vit(kwargs, embed_dim)
    model = VisionTransformer(
        img_size=image_res,
        mask_token=mask_token,
        patch_size=16,
        embed_dim=384,
        depth=depth,
        num_heads=6,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **adapter_kwargs,
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu",
            check_hash=True,
        )
        state_dict = checkpoint["model"]
        pos_embed_reshaped = interpolate_pos_embed(state_dict["pos_embed"], model)
        state_dict["pos_embed"] = pos_embed_reshaped
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model


def deit_base_patch16_224(
    pretrained=False, image_res=224, mask_token=False, depth: int = 12, **kwargs
):
    embed_dim = 768
    adapter_kwargs = adapter_kwargs_for_vit(kwargs, embed_dim)
    model = VisionTransformer(
        img_size=image_res,
        mask_token=mask_token,
        patch_size=16,
        embed_dim=768,
        depth=depth,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **adapter_kwargs,
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu",
            check_hash=True,
        )
        state_dict = checkpoint["model"]
        pos_embed_reshaped = interpolate_pos_embed(state_dict["pos_embed"], model)
        state_dict["pos_embed"] = pos_embed_reshaped
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model


def deit_large_patch16_LS(
    pretrained=False, img_size=224, pretrained_21k=False, depth: int = 24, **kwargs
):
    embed_dim = 1024
    adapter_kwargs = adapter_kwargs_for_vit(kwargs, embed_dim)
    model = vit_models(
        img_size=img_size,
        patch_size=16,
        embed_dim=1024,
        depth=depth,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        block_layers=Layer_scale_init_Block,
        **adapter_kwargs,
    )
    if pretrained:
        _pretrained_img_size = 224
        name = (
            "https://dl.fbaipublicfiles.com/deit/deit_3_large_"
            + str(_pretrained_img_size)
            + "_"
        )
        if pretrained_21k:
            name += "21k.pth"
        else:
            name += "1k.pth"

        checkpoint = torch.hub.load_state_dict_from_url(
            url=name, map_location="cpu", check_hash=True
        )
        state_dict = checkpoint["model"]
        pos_embed_reshaped = interpolate_pos_embed(state_dict["pos_embed"], model)
        state_dict["pos_embed"] = pos_embed_reshaped
        model.load_state_dict(state_dict, strict=False)
    return model


def dino_base_patch16(
    image_res=224, pretrained=False, img_size=224, depth: int = 12, **kwargs
):
    model = VisionTransformer(
        img_size=image_res,
        mask_token=False,
        patch_size=16,
        embed_dim=768,
        depth=depth,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/dino/dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth",
            map_location="cpu",
            check_hash=True,
        )
        state_dict = checkpoint
        pos_embed_reshaped = interpolate_pos_embed(state_dict["pos_embed"], model)
        state_dict["pos_embed"] = pos_embed_reshaped
        msg = model.load_state_dict(state_dict, strict=False)
        print(msg)
    return model
