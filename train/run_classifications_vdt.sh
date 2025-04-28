#!/bin/bash

config="$1"
checkpoint="$2"

python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset imagenet --use_vdt_augmentation True --overrides k_test=20
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset imagenetv2 --use_vdt_augmentation True --overrides k_test=20
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset caltech-101 --use_vdt_augmentation True --overrides k_test=20
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset oxford_pets --use_vdt_augmentation True --overrides k_test=20
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset stanford_cars --use_vdt_augmentation True --overrides k_test=20
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset oxford_flowers --use_vdt_augmentation True --overrides k_test=20
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset food-101 --use_vdt_augmentation True --overrides k_test=20
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset fgvc_aircraft --use_vdt_augmentation True --overrides k_test=20
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset sun397 --use_vdt_augmentation True --overrides k_test=20
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset dtd --use_vdt_augmentation True --overrides k_test=20
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset cub --use_vdt_augmentation True --overrides k_test=20
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset eurosat --use_vdt_augmentation True --overrides k_test=7
python -m torch.distributed.launch --nproc_per_node=1 --use_env classification_all.py --config "$config" --output_dir ./clf_output --checkpoint "$checkpoint" --evaluate --dataset ucf101 --use_vdt_augmentation True --overrides k_test=20

