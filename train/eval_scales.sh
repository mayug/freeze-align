
# 5m, 20m, 80m, 160m


# python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config dinov2-arl-wds-combined-laion5m --output_dir ./clf_output --checkpoint ./storage/lilt_cache/last_checkpoints/dinov2-arl-5m/checkpoint_14.pth --evaluate  --use_checkpoint_config >> logs/dinov2-arl-5m.log

# python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config dinov2-arl-wds-combined-laion5m --output_dir ./clf_output --checkpoint ./storage/lilt_cache/last_checkpoints/dinov2-arl-20m/checkpoint_14.pth --evaluate  --use_checkpoint_config >> logs/dinov2-arl-20m.log

# python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config dinov2-arl-wds-combined-laion5m --output_dir ./clf_output --checkpoint ./storage/lilt_cache/last_checkpoints/dinov2-arl-80m/checkpoint_14.pth --evaluate  --use_checkpoint_config >> logs/dinov2-arl-80m.log

# python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config dinov2-arl-wds-combined-laion5m --output_dir ./clf_output --checkpoint ./storage/lilt_cache/last_checkpoints/dinov2-arl-160m/checkpoint_14.pth --evaluate  --use_checkpoint_config >> logs/dinov2-arl-160m.log



# python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config dinov2-arl-wds-combined-laion5m --output_dir ./clf_output --checkpoint ./storage/lilt_cache/last_checkpoints/dinov2-arl-5m/checkpoint_14.pth --evaluate  --use_checkpoint_config >> logs/dinov2-arl-5m-coco.log

# python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config dinov2-arl-wds-combined-laion5m --output_dir ./clf_output --checkpoint ./storage/lilt_cache/last_checkpoints/dinov2-arl-20m/checkpoint_14.pth --evaluate  --use_checkpoint_config >> logs/dinov2-arl-20m-coco.log

# python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config dinov2-arl-wds-combined-laion5m --output_dir ./clf_output --checkpoint ./storage/lilt_cache/last_checkpoints/dinov2-arl-80m/checkpoint_14.pth --evaluate  --use_checkpoint_config >> logs/dinov2-arl-80m-coco.log

# python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config dinov2-arl-wds-combined-laion5m --output_dir ./clf_output --checkpoint ./storage/lilt_cache/last_checkpoints/dinov2-arl-160m/checkpoint_14.pth --evaluate  --use_checkpoint_config >> logs/dinov2-arl-160m-coco.log


python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config dinov2-arl-wds-combined-laion5m --output_dir ./clf_output --checkpoint ./storage/lilt_cache/last_checkpoints/dinov2-arl-5m/checkpoint_14.pth --evaluate --dataset imagenet --use_vdt_augmentation True >> logs/dinov2-arl-5m-imagenet.log

python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config dinov2-arl-wds-combined-laion5m --output_dir ./clf_output --checkpoint ./storage/lilt_cache/last_checkpoints/dinov2-arl-20m/checkpoint_14.pth --evaluate --dataset imagenet --use_vdt_augmentation True >> logs/dinov2-arl-20m-imagenet.log

python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config dinov2-arl-wds-combined-laion5m --output_dir ./clf_output --checkpoint ./storage/lilt_cache/last_checkpoints/dinov2-arl-80m/checkpoint_14.pth --evaluate --dataset imagenet --use_vdt_augmentation True >> logs/dinov2-arl-80m-imagenet.log

python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config dinov2-arl-wds-combined-laion5m --output_dir ./clf_output --checkpoint ./storage/lilt_cache/last_checkpoints/dinov2-arl-160m/checkpoint_14.pth --evaluate --dataset imagenet --use_vdt_augmentation True >> logs/dinov2-arl-160m-imagenet.log
