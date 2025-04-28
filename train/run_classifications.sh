#!/bin/bash

config="$1"
checkpoint="$2"

python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset imagenet --overrides k_test=20
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset imagenetv2 --overrides k_test=20
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset caltech-101 --overrides k_test=20
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset oxford_pets --overrides k_test=20
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset stanford_cars --overrides k_test=20
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset oxford_flowers --overrides k_test=20
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset food-101 --overrides k_test=20
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset fgvc_aircraft --overrides k_test=20
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset dtd --overrides k_test=20