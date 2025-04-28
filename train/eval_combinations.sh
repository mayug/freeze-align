
# all retrieval evals

python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-re-exp1/checkpoint_04.pth --evaluate  --use_checkpoint_config >> ./logs/combination_ablation2/coco-exp1-flickr.log

python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-re-exp2/checkpoint_04.pth --evaluate  --use_checkpoint_config >> ./logs/combination_ablation2/coco-exp2-flickr.log

python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-re-exp3/checkpoint_04.pth --evaluate  --use_checkpoint_config >> ./logs/combination_ablation2/coco-exp3-flickr.log

python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-re-exp4/checkpoint_04.pth --evaluate  --use_checkpoint_config >> ./logs/combination_ablation2/coco-exp4-flickr.log

python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-re-exp5/checkpoint_04.pth --evaluate  --use_checkpoint_config >> ./logs/combination_ablation2/coco-exp5-flickr.log

python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-re-exp6/checkpoint_04.pth --evaluate  --use_checkpoint_config >> ./logs/combination_ablation2/coco-exp6-flickr.log

python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-re-exp7/checkpoint_04.pth --evaluate  --use_checkpoint_config >> ./logs/combination_ablation2/coco-exp7-flickr.log

python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-re-exp8/checkpoint_04.pth --evaluate  --use_checkpoint_config >> ./logs/combination_ablation2/coco-exp8-flickr.log

python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-re-exp9/checkpoint_04.pth --evaluate  --use_checkpoint_config >> ./logs/combination_ablation2/coco-exp9-flickr.log

python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-re-exp10/checkpoint_04.pth --evaluate  --use_checkpoint_config >> ./logs/combination_ablation2/coco-exp10-flickr.log

python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-re-exp11/checkpoint_04.pth --evaluate  --use_checkpoint_config >> ./logs/combination_ablation2/coco-exp11-flickr.log

python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-re-exp12/checkpoint_04.pth --evaluate  --use_checkpoint_config >> ./logs/combination_ablation2/coco-exp12-flickr.log

python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-re-exp13/checkpoint_04.pth --evaluate  --use_checkpoint_config >> ./logs/combination_ablation2/coco-exp13-flickr.log

python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-re-exp14/checkpoint_04.pth --evaluate  --use_checkpoint_config >> ./logs/combination_ablation2/coco-exp14-flickr.log

python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-re-exp15/checkpoint_04.pth --evaluate  --use_checkpoint_config >> ./logs/combination_ablation2/coco-exp15-flickr.log


python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablations2_replacements/coco-re-exp3/checkpoint_04.pth --evaluate  --use_checkpoint_config >> ./logs/combination_ablations2_replacements/coco-exp3-flickr.log

python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablations2_replacements/coco-re-exp4/checkpoint_04.pth --evaluate  --use_checkpoint_config >> ./logs/combination_ablations2_replacements/coco-exp4-flickr.log


python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combinations_dinov2/coco-re-exp16/checkpoint_04.pth --evaluate  --use_checkpoint_config >> ./logs/combinations_dinov2/coco-exp16-flickr.log

python -m torch.distributed.launch --nproc_per_node=1 --use_env zero_shot_retrieval.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combinations_dinov2/coco-re-exp17/checkpoint_04.pth --evaluate  --use_checkpoint_config >> ./logs/combinations_dinov2/coco-exp17-flickr.log


# all classification evals

# python -m torch.distributed.launch --nproc_per_node=1 --use_env classification.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-exp1/checkpoint_04.pth --evaluate  --use_checkpoint_config --dataset imagenet --use_vdt_augmentation True >> ./logs/combination_ablation2/coco-exp1-clf.log

# python -m torch.distributed.launch --nproc_per_node=1 --use_env classification.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-exp2/checkpoint_04.pth --evaluate  --use_checkpoint_config --dataset imagenet --use_vdt_augmentation True >> ./logs/combination_ablation2/coco-exp2-clf.log

# python -m torch.distributed.launch --nproc_per_node=1 --use_env classification.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-exp3/checkpoint_04.pth --evaluate  --use_checkpoint_config --dataset imagenet --use_vdt_augmentation True >> ./logs/combination_ablation2/coco-exp3-clf.log

# python -m torch.distributed.launch --nproc_per_node=1 --use_env classification.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-exp4/checkpoint_04.pth --evaluate  --use_checkpoint_config --dataset imagenet --use_vdt_augmentation True >> ./logs/combination_ablation2/coco-exp4-clf.log

# python -m torch.distributed.launch --nproc_per_node=1 --use_env classification.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-exp5/checkpoint_04.pth --evaluate  --use_checkpoint_config --dataset imagenet --use_vdt_augmentation True >> ./logs/combination_ablation2/coco-exp5-clf.log

# python -m torch.distributed.launch --nproc_per_node=1 --use_env classification.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-exp6/checkpoint_04.pth --evaluate  --use_checkpoint_config --dataset imagenet --use_vdt_augmentation True >> ./logs/combination_ablation2/coco-exp6-clf.log

# python -m torch.distributed.launch --nproc_per_node=1 --use_env classification.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-exp7/checkpoint_04.pth --evaluate  --use_checkpoint_config --dataset imagenet --use_vdt_augmentation True >> ./logs/combination_ablation2/coco-exp7-clf.log

# python -m torch.distributed.launch --nproc_per_node=1 --use_env classification.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-exp8/checkpoint_04.pth --evaluate  --use_checkpoint_config --dataset imagenet --use_vdt_augmentation True >> ./logs/combination_ablation2/coco-exp8-clf.log

# python -m torch.distributed.launch --nproc_per_node=1 --use_env classification.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-exp9/checkpoint_04.pth --evaluate  --use_checkpoint_config --dataset imagenet --use_vdt_augmentation True >> ./logs/combination_ablation2/coco-exp9-clf.log

# python -m torch.distributed.launch --nproc_per_node=1 --use_env classification.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-exp10/checkpoint_04.pth --evaluate  --use_checkpoint_config --dataset imagenet --use_vdt_augmentation True >> ./logs/combination_ablation2/coco-exp10-clf.log

# python -m torch.distributed.launch --nproc_per_node=1 --use_env classification.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-exp11/checkpoint_04.pth --evaluate  --use_checkpoint_config --dataset imagenet --use_vdt_augmentation True >> ./logs/combination_ablation2/coco-exp11-clf.log

# python -m torch.distributed.launch --nproc_per_node=1 --use_env classification.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-exp12/checkpoint_04.pth --evaluate  --use_checkpoint_config --dataset imagenet --use_vdt_augmentation True >> ./logs/combination_ablation2/coco-exp12-clf.log

# python -m torch.distributed.launch --nproc_per_node=1 --use_env classification.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-exp13/checkpoint_04.pth --evaluate  --use_checkpoint_config --dataset imagenet --use_vdt_augmentation True >> ./logs/combination_ablation2/coco-exp13-clf.log

# python -m torch.distributed.launch --nproc_per_node=1 --use_env classification.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-exp14/checkpoint_04.pth --evaluate  --use_checkpoint_config --dataset imagenet --use_vdt_augmentation True >> ./logs/combination_ablation2/coco-exp14-clf.log

# python -m torch.distributed.launch --nproc_per_node=1 --use_env classification.py --config combinations-wds-coco-combined --output_dir ./clf_output --checkpoint ./storage/lilt_cache/combination_ablation2/coco-exp15/checkpoint_04.pth --evaluate  --use_checkpoint_config --dataset imagenet --use_vdt_augmentation True >> ./logs/combination_ablation2/coco-exp15-clf.log


