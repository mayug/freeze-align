from sentence_transformers import SentenceTransformer, models
from sentence_transformers.util import batch_to_device

import torch, random
from tqdm import tqdm
import sys
import re
import torch.nn as nn

from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification, ConvNextImageProcessor, ConvNextForImageClassification
import timm
from sentence_transformers import SentenceTransformer, models

from open_clip.loss import ClipLoss, SigLipLoss
import torch.nn.functional as F 
from typing import List, Dict, Tuple, Iterable, Type, Union, Callable, Optional, Literal, TYPE_CHECKING
import numpy as np
from torch import nn, Tensor, device
from numpy import ndarray
from tqdm.autonotebook import trange

# from .util import 


class SentenceTransformer_train(SentenceTransformer):
    
        # def __init__(self, model_name_or_path: str, device: str = None):
        #     super().__init__(model_name_or_path, device)
        #     self.train()
        #     self.model = SentenceTransformer(model_name_or_path).to('cuda:0')
    
        # def forward(self, features):
        #     """Returns token_embeddings, cls_token"""
        #     return super().forward(features)
        
        def encode(self, sentences: Union[str, List[str]],
               batch_size: int = 32,
               show_progress_bar: bool = None,
               output_value: str = 'sentence_embedding',
               convert_to_numpy: bool = True,
               convert_to_tensor: bool = False,
               device: str = None,
               normalize_embeddings: bool = False) -> Union[List[Tensor], ndarray, Tensor]:
            """
            Computes sentence embeddings.

            :param sentences: the sentences to embed.
            :param batch_size: the batch size used for the computation.
            :param show_progress_bar: Whether to output a progress bar when encode sentences.
            :param output_value: The type of embeddings to return: "sentence_embedding" to get sentence embeddings,
                "token_embeddings" to get wordpiece token embeddings, and `None`, to get all output values. Defaults
                to "sentence_embedding".
            :param convert_to_numpy: Whether the output should be a list of numpy vectors. If False, it is a list of PyTorch tensors.
            :param convert_to_tensor: Whether the output should be one large tensor. Overwrites `convert_to_numpy`.
            :param device: Which `torch.device` to use for the computation.
            :param normalize_embeddings: Whether to normalize returned vectors to have length 1. In that case,
                the faster dot-product (util.dot_score) instead of cosine similarity can be used.

            :return: By default, a list of tensors is returned. If convert_to_tensor, a stacked tensor is returned.
                If convert_to_numpy, a numpy matrix is returned.
            # """
            # self.eval()
            self.train()
            # print('inside custom encode')

            if convert_to_tensor:
                convert_to_numpy = False

            if output_value != 'sentence_embedding':
                convert_to_tensor = False
                convert_to_numpy = False

            input_was_string = False
            if isinstance(sentences, str) or not hasattr(sentences, '__len__'): #Cast an individual sentence to a list with length 1
                sentences = [sentences]
                input_was_string = True

            if device is None:
                device = self.device

            self.to(device)

            all_embeddings = []
            length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
            sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

            inner_ctr = 0

            for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
                sentences_batch = sentences_sorted[start_index:start_index+batch_size]
                features = self.tokenize(sentences_batch)
                features = batch_to_device(features, device)

                # print('inner_ctr ', inner_ctr)

                # with torch.no_grad(): # removing torch.no_grad to enable backprop
                out_features = self.forward(features)

                # check if out_features
                # print('out_features requires_grad ', out_features['sentence_embedding'].requires_grad)
                # asd

                if output_value == 'token_embeddings':
                    embeddings = []
                    for token_emb, attention in zip(out_features[output_value], out_features['attention_mask']):
                        last_mask_id = len(attention)-1
                        while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                            last_mask_id -= 1

                        embeddings.append(token_emb[0:last_mask_id+1])
                elif output_value is None:  #Return all outputs
                    embeddings = []
                    for sent_idx in range(len(out_features['sentence_embedding'])):
                        row =  {name: out_features[name][sent_idx] for name in out_features}
                        embeddings.append(row)
                else:   #Sentence embeddings
                    embeddings = out_features[output_value]
                    # embeddings = embeddings.detach() # removing detach to enable backprop
                    if normalize_embeddings:
                        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                    # fixes for #522 and #487 to avoid oom problems on gpu with large datasets
                    if convert_to_numpy:
                        embeddings = embeddings.cpu()

                    all_embeddings.extend(embeddings)
                    inner_ctr += 1

            all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

            if convert_to_tensor:
                if len(all_embeddings):
                    all_embeddings = torch.stack(all_embeddings)
                else:
                    all_embeddings = torch.Tensor()
            elif convert_to_numpy:
                all_embeddings = np.asarray([emb.numpy() for emb in all_embeddings])

            if input_was_string:
                all_embeddings = all_embeddings[0]

            # print('all_embeddings  requires_grad ', all_embeddings.requires_grad)
            # asd
            return all_embeddings
                
        def encode_new(self, sentences: Union[str, List[str]],
                    batch_size: int = 32,
                    show_progress_bar: bool = None,
                    output_value: str = 'sentence_embedding',
                    convert_to_numpy: bool = True,
                    convert_to_tensor: bool = False,
                    device: str = None,
                    normalize_embeddings: bool = False) -> Union[List[Tensor], ndarray, Tensor]:
            
                features = self.tokenize(sentences)
                features = batch_to_device(features, device)

                out_features = self.forward(features)

                # print('out_features', out_features)
                return out_features['sentence_embedding']
                # asd