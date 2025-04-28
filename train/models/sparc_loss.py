import torch
import torch.nn.functional as F


def pairwise_contrastive_loss(a, b, inverse_temperature=1.0):
    labels = torch.eye(a.shape[0], device=a.device)
    
    # Compute the dot product similarity scaled by inverse_temperature
    logits_ab = torch.matmul(a, b.T) * inverse_temperature
    
    # Apply softmax cross-entropy loss
    loss = F.cross_entropy(logits_ab, labels, reduction='mean')
    
    return loss

def masked_pairwise_contrastive_loss(a, b, mask, inverse_temperature=1.0, INF=1e9):
    batch_size, seq_len, _ = a.shape
    
    # Creating a mask for logits where 1.0 - mask indicates positions to ignore
    mask_logits = (1-mask).unsqueeze(1).repeat(1, seq_len, 1)

    # print('mask_logits')
    # print(mask_logits)
    # print(mask_logits.shape)

    mask_logits = mask_logits.view(-1, seq_len)
    # print(mask_logits.shape)
    
    # Creating labels for matching pairs within each sequence
    labels = torch.eye(seq_len, device=a.device).repeat(batch_size, 1)
    # print('labels')
    # print(labels)
    # print(labels.shape)
    
    # Computing pairwise logits with temperature scaling
    logits = torch.einsum('bmd,bnd->bmn', a, b) * inverse_temperature

    # print(logits.shape)

    logits = logits.view(-1, seq_len) 

    # print(logits.shape)
    # print('logits after reshape ')
    # print(logits)
    
    # Subtracting the mask_logits scaled by INF to ignore certain pairs
    logits_masked = logits - mask_logits*INF

    # print(logits_masked.shape)
    # print(logits_masked)

    
    # Applying softmax cross-entropy loss
    loss = F.cross_entropy(logits_masked, labels, reduction='none').view(batch_size, -1)
    
    # Applying the mask to the loss and normalizing
    loss = torch.sum(loss * mask) / torch.sum(mask)
    
    return loss

def l2_normalize(tensor, axis=-1):
    return F.normalize(tensor, p=2, dim=axis)

def compute_similarity_and_align_weights(l_token_embed, v_patch_embed, similarity_threshold, language_mask):
    # similarity calculation
    similarity = torch.einsum('btd,bpd->btp', l_token_embed, v_patch_embed)
    
    # min-max normalization
    min_val = torch.min(similarity, dim=-1, keepdim=True).values
    max_val = torch.max(similarity, dim=-1, keepdim=True).values
    similarity = (similarity - min_val) / (max_val - min_val)
    
    # thresholding
    similarity = torch.where(similarity < similarity_threshold, torch.zeros_like(similarity), similarity)
    
    # alignment-weighting
    v_align_weights = similarity / torch.sum(similarity, dim=-1, keepdim=True)
    
    # compute weighted sum of v_patch_embed based on alignment weights
    l_grouped_v_patch_embed = torch.einsum('btp,bpd->btd', v_align_weights, v_patch_embed)
    
    # l2 normalization
    l_grouped_v_patch_embed = l2_normalize(l_grouped_v_patch_embed, axis=-1)
    l_token_embed = l2_normalize(l_token_embed, axis=-1)
    
    return l_grouped_v_patch_embed, l_token_embed

def compute_fg_loss(l_token_embed, v_patch_embed, language_mask, similarity_threshold,
                    inverse_temperature=1.0):


    l_grouped_v_patch_embed, l_token_embed_normalized = compute_similarity_and_align_weights(
        l_token_embed, v_patch_embed, similarity_threshold, language_mask)
    
    # Assuming masked_pairwise_contrastive_loss is implemented as previously described
    loss_vl_local = masked_pairwise_contrastive_loss(l_grouped_v_patch_embed, 
                                                     l_token_embed_normalized, 
                                                     language_mask, 
                                                     inverse_temperature)
    loss_lv_local = masked_pairwise_contrastive_loss(l_token_embed_normalized,
                                                      l_grouped_v_patch_embed, 
                                                      language_mask,
                                                      inverse_temperature)
    
    return loss_vl_local, loss_lv_local
