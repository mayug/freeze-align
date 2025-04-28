import argparse
import os
import torch
from tqdm import tqdm
from torchvision.datasets import CocoCaptions
from torchvision.transforms import PILToTensor
from transformers import AutoImageProcessor, AutoModel
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader

# Constants
COCO_ROOT = "../coco_dataset/val2017"
NOCAPS_ROOT = "../datasets/openimages/validation"
COCO_ANN = "../coco_dataset/annotations/captions_val2017.json"
NOCAPS_ANN = "../datasets/openimages/nocaps_val_4500_captions.json"

# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(description="Generate and save embeddings")
    parser.add_argument("--vision_model", help="Vision transformer model name", type=str, required=True)
    parser.add_argument("--text_model", help="Sentence transformer model name", type=str, required=True)
    parser.add_argument("--dataset", help="Dataset to use (coco/nocaps)", default="coco", type=str)
    parser.add_argument("--gpu", help="GPU ID to use", default=0, type=int)
    return parser.parse_args()

# Dataset Loader
def get_dataset(dataset_name):
    if dataset_name == "coco":
        return CocoCaptions(root=COCO_ROOT, annFile=COCO_ANN)
    elif dataset_name == "nocaps":
        return CocoCaptions(root=NOCAPS_ROOT, annFile=NOCAPS_ANN, transform=PILToTensor())
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

# Model Loader
def load_vision_model(model_name, device):
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return model, processor

def load_text_model(model_name, device):
    return SentenceTransformer(model_name).to(device)

# Embedding Generation

def generate_vision_embeddings(model, processor, dataset, device, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                            collate_fn=lambda x: list(zip(*x)), num_workers=4)
    embeddings = []

    for images, _ in tqdm(dataloader, desc="Generating vision embeddings"):
        inputs = processor(images=list(images), return_tensors="pt").to(device)
        outputs = model(**inputs)
        batch_embeddings = outputs.last_hidden_state.mean(dim=1).detach().cpu()
        embeddings.append(batch_embeddings)

    return torch.cat(embeddings)

def generate_text_embeddings(model, dataset, device):
    embeddings = []
    for _, captions in tqdm(dataset, desc="Generating text embeddings"):
        embeddings_batch = model.encode(captions, convert_to_tensor=True, device=device)
        embeddings.append(embeddings_batch.mean(dim=0).cpu())
    return torch.stack(embeddings)

# Main Function
def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    # Load models
    vision_model, vision_processor = load_vision_model(args.vision_model, device)
    text_model = load_text_model(args.text_model, device)

    # Load dataset
    dataset = get_dataset(args.dataset)

    # Generate and save embeddings
    vision_embeddings = generate_vision_embeddings(vision_model, vision_processor, dataset, device)
    torch.save(vision_embeddings, f"data/{args.dataset}_{args.vision_model}_img.pt")

    text_embeddings = generate_text_embeddings(text_model, dataset, device)
    torch.save(text_embeddings, f"data/{args.dataset}_{args.text_model}_text.pt")

if __name__ == "__main__":
    main()
