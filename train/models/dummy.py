# import sys
# import torch
# sys.path.append('/notebooks/segment-anything-2/')
# from sam2.modeling.sam2_base import SAM2Base
# model = torch.load("/notebooks/segment-anything-2/checkpoints/sam2_model.pth", weights_only=False)

# print(model)

# import sys
# sys.path.append('/notebooks/segment-anything-2/')
# from sam2.build_sam import build_sam2, build_sam2_hf

# model = build_sam2_hf('facebook/sam2-hiera-tiny')
# print(model)

def return_dummy():
    import sys
    sys.path.append('/notebooks/segment-anything-2/')
    from sam2.build_sam import build_sam2, build_sam2_hf

    model = build_sam2_hf('facebook/sam2-hiera-tiny')
    print(model)
    return model