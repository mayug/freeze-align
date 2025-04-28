import utils
from models import build

import torch
import torch.nn.functional as F
from torch import nn

import random
from models.multihead_attn import MultiHeadAttention
from models.sparc_loss import compute_fg_loss



class ProjectionHead(nn.Module):
    def __init__(self,
        embedding_dim,
        projection_dim=512,
        dropout=0.1):

        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x
    

class Patch_Projection(torch.nn.Module):
    def __init__(self, embedding_dim, projection_dim):
        super(Patch_Projection, self).__init__()
        
        self.linear_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
        )
        self.non_linear_projection = nn.Sequential(
            nn.Linear(embedding_dim, projection_dim),
            nn.GELU(),
            nn.Linear(projection_dim, projection_dim),
        )
    def forward(self, x):
        return self.linear_projection(x) + self.non_linear_projection(x)
    

class CLIP(nn.Module):
    def __init__(
        self, tokenizer=None, config=None,
        **kwargs,
    ):
        super().__init__()

        print('This is the CLIP model from models/clip_adjustable_combine_vis_cls.py')
        # asd

        self.fg_thresh = config.fg_thresh

        self.vis_pooling = config.vis_pooling
        self.text_pooling = config.text_pooling

        self.return_pooled_output = True if config.text_pooling == 'cls' else False

        print('vis_pooling ', self.vis_pooling)
        print('text_pooling ', self.text_pooling)
        print('return_pooled_output ', self.return_pooled_output)

        self.tokenizer = tokenizer
        embed_dim = config["embed_dim"]

        self.visual_processor, self.visual_encoder = build.vision_encoder(
            config, config.vision_encoder, config.adapter_append
        )
        vision_width = self.visual_encoder.config.hidden_size
        self.text_encoder = build.text_encoder(
            config, config.text_encoder, config.adapter_append
        )

        try:
            text_width = self.text_encoder.config.hidden_size
        except:
            text_width = 768

        self.vision_proj = nn.Identity()
        self.text_proj = nn.Identity()


        if config.local_vision_projection == 'linear':
            print('Using linear projection for local vision')
            self.local_vision_proj = nn.Linear(vision_width, embed_dim)
            self.local_combine_alpha=1.0
        elif config.local_vision_projection == 'mlp':
            print('Using mlp projection for local vision')
            self.local_vision_proj = ProjectionHead(embedding_dim=vision_width, 
                                                    projection_dim=embed_dim)
            self.local_combine_alpha=1.0
        elif config.local_vision_projection == 'patch':
            print('Using patch projection for local vision')
            self.local_vision_proj = nn.Sequential(
            nn.LayerNorm(vision_width),
            nn.Dropout(0.1),
            Patch_Projection(embedding_dim=vision_width, 
                            projection_dim=embed_dim),
        )
            self.local_combine_alpha=1.0
        else:
            print('Using identity projection for local vision')
            self.local_vision_proj = nn.Identity()
            self.local_combine_alpha=0.0
        
        if config.cls_vision_projection == 'linear':
            print('Using linear projection for cls vision')
            self.cls_vision_proj = nn.Linear(vision_width, embed_dim)
            self.combine_alpha=1.0

        elif config.cls_vision_projection == 'mlp':
            print('Using mlp projection for cls vision')
            self.cls_vision_proj = ProjectionHead(embedding_dim=vision_width, 
                                                    projection_dim=embed_dim)
            self.combine_alpha=1.0

        elif config.cls_vision_projection == 'patch':
            print('Using patch projection for cls vision')
            self.cls_vision_proj = nn.Sequential(
            nn.LayerNorm(vision_width),
            nn.Dropout(0.1),
            Patch_Projection(embedding_dim=vision_width, 
                            projection_dim=embed_dim),
        )
            self.combine_alpha=1.0

        else:
            print('Using identity projection for cls vision; not adding cls embedding')
            self.cls_vision_proj = nn.Identity()
            self.combine_alpha=0.0

        self.local_text_proj = nn.Identity()

        if config.text_projection == 'linear':
            print('Using linear projection for text')
            self.text_proj = nn.Linear(text_width, embed_dim)
        elif config.text_projection == 'mlp':
            print('Using mlp projection for text')
            self.text_proj = ProjectionHead(embedding_dim=text_width, 
                                                    projection_dim=embed_dim)
        elif config.text_projection == 'patch':
            print('Using patch projection for text')
            self.text_proj = nn.Sequential(
            nn.LayerNorm(text_width),
            nn.Dropout(0.1),
            Patch_Projection(embedding_dim=text_width, 
                            projection_dim=embed_dim),
            )
        else:
            print('Using identity projection for local text')
            self.text_proj = nn.Identity()

        if config.local_text_projection == 'linear':
            print('Using linear projection for local text')
            self.local_text_proj = nn.Linear(text_width, embed_dim)
        elif config.local_text_projection == 'mlp':
            print('Using mlp projection for local text')
            self.local_text_proj = ProjectionHead(embedding_dim=text_width, 
                                                    projection_dim=embed_dim)
        elif config.local_text_projection == 'patch':
            print('Using patch projection for local text')
            self.local_text_proj = nn.Sequential(
            nn.LayerNorm(text_width),
            nn.Dropout(0.1),
            Patch_Projection(embedding_dim=text_width, 
                            projection_dim=embed_dim),
            )
        else:
            print('Using identity projection for local text')
            self.local_text_proj = nn.Identity()
                


        self.temp = nn.Parameter(torch.ones([]) * config["temp"])

        if config.freeze_vision_encoder:
            utils.freeze_model(self.visual_encoder)

        if config.freeze_text_encoder:
            utils.freeze_model(self.text_encoder)

        if config.freeze_proj:
            utils.freeze_model(self.vision_proj)
            utils.freeze_model(self.text_proj)

        if config.unlock_layernorm:
            if config.unlock_layernorm in ("vision_only", True):
                for name, param in self.visual_encoder.named_parameters():
                    if "norm" in name.lower():
                        param.requires_grad = True
            if config.unlock_layernorm in ("language_only", True):
                for name, param in self.text_encoder.named_parameters():
                    if "LayerNorm" in name:
                        param.requires_grad = True

        if config.unlock_dense:
            for name, param in self.visual_encoder.named_parameters():
                if "mlp" in name.lower():
                    param.requires_grad = True
            for name, param in self.text_encoder.named_parameters():
                if "dense" in name:
                    param.requires_grad = True

        if config.unlock_attn:
            for name, param in self.visual_encoder.named_parameters():
                if "attn" in name.lower():
                    param.requires_grad = True
            for name, param in self.text_encoder.named_parameters():
                if "attention" in name:
                    param.requires_grad = True

        if config.unlock_random:
            bert_choices = (
                "query",
                "key",
                "value",
                "attention.output.dense",
                "intermediate.dense",
            )
            for block in self.text_encoder.encoder.layer:
                parameter_to_unlock = random.choice(bert_choices)
                for name, param in block.named_parameters():
                    if parameter_to_unlock in name.lower():
                        param.requires_grad = True

            vit_choices = (
                "proj",
                "fc1",
                "fc2",
            )
            for block in self.visual_encoder.blocks:
                parameter_to_unlock = random.choice(vit_choices)
                for name, param in block.named_parameters():
                    if parameter_to_unlock in name.lower():
                        param.requires_grad = True

        if config.add_adapter:
            last_lm_layer = self.text_encoder.encoder.layer[-1]
            for param in last_lm_layer.parameters():
                param.requires_grad = True

            last_vit_layer = self.visual_encoder.blocks[-1]
            for param in last_vit_layer.parameters():
                param.requires_grad = True

            for param in self.visual_encoder.norm.parameters():
                param.requires_grad = True

        # if config.add_cross_adapter:
        #     print('Adding cross adapter')
        # # if True:
        #     self.cross_attn =  MultiHeadAttention(n_head=8, d_model=768, d_k=96, d_v=96)
        #     for name, param in self.cross_attn.named_parameters():
        #         param.requires_grad = True

        if config.conventional_adapter.insert:
            if config.conventional_adapter.insert in ("vision_only", True):
                for name, param in self.visual_encoder.named_parameters():
                    if "adapter" in name:
                        param.requires_grad = True

            if config.conventional_adapter.insert in ("language_only", True):
                if 'clip' in config.text_encoder:
                    for name, param in self.text_encoder.text_model.named_parameters():
                        if "adapter" in name:
                            param.requires_grad = True
                    
                else:
                    for name, param in self.text_encoder.encoder.named_parameters():
                        if "adapter" in name:
                            param.requires_grad = True

        if config.bitfit: # for all-roberta-large-v1, no need to use the bias in the robertapooler as it is not used in sentence_transformers
            if config.bitfit in ("vision_only", True):
                for name, param in self.visual_encoder.named_parameters():
                    if "bias" in name:
                        param.requires_grad = True
            if config.bitfit in ("language_only", True):
                for name, param in self.text_encoder.named_parameters():
                    if "bias" in name and "pooler" not in name:
                        param.requires_grad = True

        if config.always_freeze:
            for idx_always_locked in config.always_freeze.visual_encoder:
                for block_idx, block in enumerate(self.visual_encoder.blocks):
                    if idx_always_locked == block_idx:
                        for name, param in block.named_parameters():
                            param.requires_grad = False

            for idx_always_locked in config.always_freeze.text_encoder:
                for block_idx, block in enumerate(self.text_encoder.encoder.layer):
                    if idx_always_locked == block_idx:
                        for name, param in block.named_parameters():
                            param.requires_grad = False

        trainable_params = sum(
            param.numel() for param in self.parameters() if param.requires_grad
        )
        total_params = sum(param.numel() for param in self.parameters())

        print("Trainable parameters:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)

        print(
            "percentage_trainable={}".format(
                round(trainable_params / total_params * 100, 2)
            )
        )
        print("num trainable={}".format(trainable_params))
        print("total params={}".format(total_params))

        self.config = config
        # # print norms of all adapters
        # print('all adapter weight norms before training')

        # for name, param in self.named_parameters():
        #     if 'adapter' in name:
        #         print(name, param.norm().item())
        


        # asd


    def _forward_(self, image, text, return_dict=False, return_pooled_output=False):
        # this function returns the image and text final token embeddings
        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        if image is not None:
            image_embeds = self.visual_encoder(image).last_hidden_state
        else:
            image_embeds = None


        if text is not None:
            # print(text['input_ids'].shape)
            # print(text['attention_mask'].shape)

            text_output = self.text_encoder(
                **text, output_hidden_states=True
            )
            text_embeds = text_output.last_hidden_state
            text_pooled_output = text_output.pooler_output
        else:
            text_embeds = None
            
        if return_pooled_output is False:
            text_pooled_output = None

        return image_embeds, text_embeds, text_pooled_output
    

    #### here we use only local and then do pooling, removing the unncessary normalizations too
    def _get_features_(self, image_embeds, text_embeds, text, text_pooled_output=None):
        # projections and averaging happen inside this function

        # this function pools the token embeddings and returns the image and text features
        # text pooling uses attention mask to get the mean of the tokens


        text_feat = None

        if text_pooled_output is not None:
            text_feat = self.text_proj(text_pooled_output)
            text_feat = F.normalize(text_feat, dim=-1)

        if text_feat is None:
            if text_embeds is not  None:
                # expand text.attention mask and apply to text_embeds to get the mean


                # print('text_embeds ', text_embeds.shape)
                # asd
                text_embeds = self.local_text_proj(text_embeds)

                text_feat = text_embeds * text['attention_mask'].unsqueeze(-1)
                # print('text_embeds ', text_embeds.shape)
                text_feat = text_feat.sum(dim=1) / text['attention_mask'].sum(dim=1).unsqueeze(-1)

                text_feat =self.text_proj(text_feat)
                text_feat = F.normalize(text_feat, dim=-1)

            else:
                text_feat = None

        if image_embeds is not None:
            image_pooled_output = image_embeds[:,0,:]
            image_embeds = self.local_vision_proj(image_embeds)

            # print(image_embeds.shape)
            # asd

            if self.vis_pooling == 'mean':
                image_feat = image_embeds[:,1:,:].mean(dim=1)
            elif self.vis_pooling == 'max':
                image_feat = image_embeds[:,1:,:].max(dim=1).values

            image_feat_cls = self.cls_vision_proj(image_pooled_output)
            image_feat = self.local_combine_alpha*image_feat + self.combine_alpha*image_feat_cls

            image_feat = F.normalize(self.vision_proj(image_feat), dim=-1)

        else:
            image_feat = None


        


        
        return image_feat, text_feat, image_embeds, text_embeds

    def eval_forward(self, image, text, return_dict=False):
        image_embeds, text_embeds, text_pooled_output = self._forward_(image, text, return_dict=False, return_pooled_output=self.return_pooled_output)
        image_feat, text_feat, image_embeds, text_embeds = self._get_features_(image_embeds, text_embeds, text, text_pooled_output=text_pooled_output)
        return image_feat, text_feat
    
    def eval_text_forward(self, text, return_dict=False):
        _, text_embeds, text_pooled_output = self._forward_(None, text, return_dict=False, return_pooled_output=self.return_pooled_output)
        _, text_feat, _, text_embeds = self._get_features_(None, text_embeds, text, text_pooled_output=text_pooled_output)
        return text_feat
    
    def eval_image_forward(self, image, return_dict=False):
        image_embeds, _, _ = self._forward_(image, None, return_dict=False)
        image_feat, _, image_embeds, _ = self._get_features_(image_embeds, None, None)
        return image_feat


    def forward(self, image, text, return_dict=False):
        # print(text)
        # asd

        image_embeds, text_embeds, text_pooled_output = self._forward_(image, text, return_dict=False, return_pooled_output=self.return_pooled_output)
        # print('image_embeds ', [image_embeds.shape, image_embeds.min(), image_embeds.max()])
        # print('text_embeds ', [text_embeds.shape, text_embeds.min(), text_embeds.max()])
        # asd
        image_feat, text_feat, image_embeds_proj, text_embeds_proj = self._get_features_(image_embeds, text_embeds, text, text_pooled_output=text_pooled_output)
        
  
        sim_i2t = image_feat @ text_feat.T / self.temp
        sim_t2i = text_feat @ image_feat.T / self.temp

        # print('sim_i2t ', [sim_i2t.shape, sim_i2t.min(), sim_i2t.max()])
        # print('sim_t2i ', [sim_t2i.shape, sim_t2i.min(), sim_t2i.max()])



        with torch.no_grad():
            sim_targets = torch.zeros(sim_i2t.size()).to(image.device)
            sim_targets.fill_diagonal_(1)

        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2

        if self.config.local_loss_weight > 0:

            threshold = self.fg_thresh
            # loss_fg = get_fg_loss(image_embeds_proj, text_embeds_proj, text['attention_mask'], self.temp)
            # print(image_embeds_proj.shape)
            # print(text_embeds_proj.shape)
            # asd

            eos_token_id = self.tokenizer.eos_token_id
            # print(text['input_ids'])
            # print()
            # print(eos_token_id)
            text_pooled_id = (text['input_ids']==eos_token_id).int().argmax(dim=-1)
            # print('text attention mask before ', text['attention_mask'])

            # print('text_pooled_id', text_pooled_id)
            # make the pooled token and first token in attention mask as False
            new_attention_mask = text['attention_mask'].clone()
            new_attention_mask[torch.arange(text['attention_mask'].shape[0], device=text['attention_mask'].device),text_pooled_id] = 0
            new_attention_mask[:,0] = 0
            text['attention_mask'] = new_attention_mask


            # fg loss doesn't include the pooled image token and the first and last text token
            loss_fg = get_fg_loss_new(image_embeds_proj[:,1:,:], text_embeds_proj, text['attention_mask'], 
                                    threshold, self.temp)
        else:
            loss_fg = torch.tensor(0.0).to(image.device)


        if return_dict:
            return {
                "losses": {
                    "loss_ita": loss_ita,
                    "loss_fg": loss_fg
                }
            }
        return loss_ita

    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(
                model_pair[0].parameters(), model_pair[1].parameters()
            ):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient



@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


def get_fg_loss(image_embeds, text_embeds, 
                attention_mask, temp):

    threshold = 1/image_embeds.shape[1] # threshold set to 1/num_image_tokens according to sparc paper

    # print('image_embeds.shape ', image_embeds.shape)
    # print('text_embeds.shape ', text_embeds.shape)
    # print('attention_mask.shape ', [attention_mask.shape, attention_mask.sum(dim=1)])

    # calculate the similarity matrix between all image and text tokens
    # image_embeds is bs*image_tokens*embed_dim and text_embeds is bs*text_tokens*embed_dim
    # we want to get a similarity matrix of size bs*text_tokens*image_tokens

    sims = torch.matmul(text_embeds, image_embeds.transpose(1, 2)) / temp

    # print('sims.shape ', [sims.shape, sims.min(), sims.max()])

    # print('first token embeddings')
    # print(text_embeds[0,0,:])
    # print(text_embeds[0,1,:])
    # print(text_embeds[0,2,:])

    # print('last token embeddings')
    # print(text_embeds[0,-1,:])
    # print(text_embeds[0,-2,:])
    # print(text_embeds[0,-3,:])

    loss = []

    for image_embeds_single, text_embeds_single, sims_single, attn_mask_single in zip(image_embeds, text_embeds, sims, attention_mask):
        # print('text_embeds_single.shape ', text_embeds_single.shape)
        # print('sims_single.shape ', sims_single.shape)
        # print('attn_mask_single.shape ', attn_mask_single.shape)


        sum_ = attn_mask_single.sum().item()
        sims_single = sims_single[:sum_, :]
        text_embeds_single = text_embeds_single[:sum_]

        # print('sims_single.shape ', [sims_single.shape, sims_single.min(), sims_single.max()])
        # print('text_embeds_single.shape ', [text_embeds_single.shape, text_embeds_single.min(), text_embeds_single.max()])

        # asd


        loss.append(fg_loss_single(sims_single, text_embeds_single, image_embeds_single, threshold, temp))

    loss = torch.stack(loss).mean()

    return loss 


    

def fg_loss_single(sims, text_embeds, image_embeds, threshold, temp):
    threshold = 0.3
    # print('thresh ', threshold)
    
    # minmax across image tokens
    min_ = sims.min(dim=-1)
    max_ = sims.max(dim=-1)

    sims = (sims-min_.values.unsqueeze(-1))/(max_.values-min_.values).unsqueeze(-1)

    # print('sims.shape ', [sims.shape, sims.min(), sims.max(), sims.mean()])

    # thresholding
    thresh_mask = (sims > threshold).detach().float()
    # print(thresh_mask.sum(axis=1))
    # asd
    sims = sims * thresh_mask

    # print('sims.shape ', [sims.shape, sims.min(), sims.max(), sims.mean()])

    # alignment weights
    al_wts = sims/sims.mean(dim=-1).unsqueeze(-1)

    # print('al_wts.shape ', [al_wts.shape, al_wts.min(), al_wts.max(), al_wts.mean()])
    # print('image_embeds ', image_embeds.shape)
    # asd

    # get language grouped image tokens

    image_embeds_grouped = al_wts@image_embeds

    # print('text_embeds_transformed.shape ', [image_embeds_grouped.shape, image_embeds_grouped.min(), image_embeds_grouped.max()])
    # asd


    # normalize both
    image_embeds_grouped = F.normalize(image_embeds_grouped, dim=-1)
    text_embeds = F.normalize(text_embeds, dim=-1)

    sim_i2t = image_embeds_grouped @ text_embeds.T/ temp
    sim_t2i = text_embeds @ image_embeds_grouped.T/ temp

    # print('sim_i2t.shape ', [sim_i2t.shape, sim_i2t.min(), sim_i2t.max()])
    # print('sim_t2i.shape ', [sim_t2i.shape, sim_t2i.min(), sim_t2i.max()])
    # asd


    with torch.no_grad():
        sim_targets = torch.zeros(sim_i2t.size()).to(text_embeds.device)
        sim_targets.fill_diagonal_(1)

    loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_targets, dim=1).mean()
    loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_targets, dim=1).mean()

    loss_ita = (loss_i2t + loss_t2i) / 2

    return loss_ita





def get_fg_loss_new(image_embeds, text_embeds, 
                attention_mask, threshold, temp):

    if threshold is None:
        threshold = 1/image_embeds.shape[1] # threshold set to 1/num_image_tokens according to sparc paper

    loss_vl_local, loss_lv_local = compute_fg_loss(
        text_embeds, image_embeds, attention_mask, 
        similarity_threshold=threshold, inverse_temperature=1/temp)
    
    return 0.5 * (loss_vl_local + loss_lv_local)