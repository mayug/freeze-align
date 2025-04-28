# freeze-align


# 🚀 Let's Align Modalities Efficiently!


Freeze-Align is a lightweight framework for **flexible multimodal alignment**.  
It connects **frozen vision and language encoders** using small trainable projectors, allowing efficient cross-modal learning **without the need for full-scale retraining**.  
This repository contains the code and datasets associated with our CVPR 2025 paper:

> **Harnessing Frozen Unimodal Encoders for Flexible Multimodal Alignment**
This framework is built on the premise that semantically similar vision and language embedding spaces can be aligned through simple projection transformations. For example, by aligning DINOv2 with the Sentence-Transformer model `all-roberta-large-v1`, we achieve a remarkable **76% zero-shot ImageNet accuracy**, surpassing comparable CLIP models while reducing alignment compute by **65x** and paired data requirements by **20x**.

We believe this approach holds immense potential for further advancements. We invite the open-source community to explore aligning newer, more powerful vision and language encoders to develop high-performing CLIP-like models with minimal effort. Notably, recent improvements in language models on the MTEB benchmark and advancements in SSL vision models present exciting opportunities for experimentation and innovation.

---

## 📚 Resources
- **[Paper](https://arxiv.org/abs/2409.19425)**
- **[Video]** (Coming soon)
- **[Slides]** (Coming soon)

---

## ✨ Features
- Optimal vision/language encoder pair discovery via **Centered Kernel Alignment (CKA)**
- Lightweight training using **frozen** unimodal backbones
- Curate high-quality datasets from **LAION** or other pools to enable efficient alignment
- Supports flexible vision encoders (huggingface transformers) and language encoders (sentence transformers)
- Drastically **reduced compute and data requirements**. For instance we outperform OpenAI , LAION CLIP models with 20x less paired data nd 65x less compute. 

---

## ⚡ Quickstart

```bash
# Clone the repository
git clone https://github.com/mayug/freeze-align.git
cd freeze-align

# Create environment
conda env create -f environment.yaml
conda activate freeze-align


---

## 🧹 Code Structure

### Alignment Module (`/alignment`)

Find the most semantically similar vision-language encoder pairs.

```bash
python get_semantic_sim.py
```

**Workflow:**
1. Download the COCO dataset.
2. Generate embeddings for selected models.
3. Compute linear CKA scores.
4. Save a plot (`cka_results.png`) and output the best encoder pair.

---

### 🏋️‍♂️ Training Module (`/train`)

Train lightweight projectors between frozen encoders.

**Setup:**
1. **Download datasets** using [img2dataset](https://github.com/rom1504/img2dataset):
    - CC3M
    - CC12M
    - SBU
    - LAION class-collected 6M ([Google Drive](https://drive.google.com/file/d/1h-fkZx5d0xmTNQLXwgBLBC9RiafITy-o/view?usp=sharing), Hugging Face link coming soon)
    - ImageNet validation set
2. **Configure dataset paths** in `dinov2-arl-wds-combined.yaml`.
3. **Set up environment:**

```bash
conda env create -f environment.yaml
conda activate freeze-align
bash extra_install.sh
```

**Training Command:**

```bash
python -m torch.distributed.launch --master_port=43770 --nproc_per_node=8 --use_env PretrainHydra.py --config dinov2-arl-wds-combined --output_dir ./storage/output/ --overrides +save_last_only=False fp16=True disable_wandb=False text_pooling=mean local_vision_projection=patch local_text_projection=patch text_projection=mlp
```

---

### 📦 Data Collection Module (`/collection`)

Curate a concept-rich dataset from LAION.

**Steps:**

1. **Download LAION Metadata** into `/laion400m-meta/`.
2. **Compute Embeddings:**

```bash
python getting_laion_embeds.py --gpu <GPU_ID> --b <BATCH_SIZE> --m <MODEL> --p <PART>
```

3. **Calculate Similarity Scores:**

```bash
python scores_new.py --gpu <GPU_ID> --b <BATCH_SIZE> --p <PART>
```

4. **Sort Top Samples:**

```bash
python sort_samples.py --p <PART> --max <MAX_SAMPLES> --b <BATCH_SIZE> --sort_b <SORT_BATCH_SIZE> --gpu <GPU_ID>
```

5. **Deduplicate and Collect Final Samples:**

```bash
python collect_fast.py --parts <NUM_PARTS> --max <MAX_SAMPLES>
```

> 🔥 **Tip:** Complete all parts for each step before proceeding to the next.

---

## 🛠️ Planned Improvements (TODO)
- [ ] Publish class-collected datasets to Hugging Face Datasets
- [ ] Add `push_to_hub` utility for uploading trained models
- [ ] Release Colab demos for alignment and training
- [ ] Add support for additional vision backbones (SAM, EVA-CLIP, etc.)

---

## 📜 Citation

If you use our work, please cite:

```bibtex
@inproceedings{maniparambil2025harnessing,
  title={Harnessing Frozen Unimodal Encoders for Flexible Multimodal Alignment},
  author={Maniparambil, Mayug and Akshulakov, Raiymbek and Djilali, Yasser Abdelaziz Dahou and Narayan, Sanath and Singh, Ankit and O'Connor, Noel E},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2025}
}
```

---

## 🤝 Acknowledgments
- Code adapted in part from the [LiLT](https://github.com/codezakh/LilT) project.
- Dataset downloads powered by [img2dataset](https://github.com/rom1504/img2dataset).

---



