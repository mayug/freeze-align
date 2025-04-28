import os
import torch
import matplotlib.pyplot as plt
from download_data import download_coco_dataset
from get_embeds import load_vision_model, load_text_model, get_dataset, generate_vision_embeddings, generate_text_embeddings
from utils import linear_CKA

def main():
    # Step 1: Download the COCO dataset
    data_dir = "../coco_dataset/"
    embed_dir = "../embeddings/"
    if not os.path.exists(embed_dir):
        os.makedirs(embed_dir)
    if not os.path.exists(data_dir):
        print("Downloading COCO dataset...")
        download_coco_dataset(data_dir)

    # Step 2: Generate embeddings for specified models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vision_models = ["facebook/dinov2-base"]

    text_models = ["all-mpnet-base-v2",
    "multi-qa-mpnet-base-dot-v1",
    "all-distilroberta-v1",
    "all-MiniLM-L12-v2",
    "multi-qa-distilbert-cos-v1",
    "all-MiniLM-L6-v2",
    "multi-qa-MiniLM-L6-cos-v1",
    "paraphrase-multilingual-mpnet-base-v2",
    "paraphrase-albert-small-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "paraphrase-MiniLM-L3-v2",
    "distiluse-base-multilingual-cased-v1",
    "distiluse-base-multilingual-cased-v2"]


    
    dataset_name = "coco"
    embeddings = {}

    print("Generating embeddings...")
    dataset = get_dataset(dataset_name)

    # Generate vision embeddings
    for vision_model_name in vision_models:
        print(f"Processing vision model: {vision_model_name}")
        # load from cache if exists
        if os.path.exists(f"{embed_dir}/{dataset_name}_{vision_model_name.replace('/', '_')}_img.pt"):
            print(f"Loading cached vision embeddings from {embed_dir}/{dataset_name}_{vision_model_name.replace('/', '_')}_img.pt")
            vision_embeddings = torch.load(f"{embed_dir}/{dataset_name}_{vision_model_name.replace('/', '_')}_img.pt")
            embeddings[f"{vision_model_name}_img"] = vision_embeddings
        else:
            vision_model, vision_processor = load_vision_model(vision_model_name, device)
            vision_embeddings = generate_vision_embeddings(vision_model, vision_processor, dataset, device)
            embeddings[f"{vision_model_name}_img"] = vision_embeddings
            print(f"Vision embeddings shape: {vision_embeddings.shape}")
            torch.save(vision_embeddings, f"{embed_dir}/{dataset_name}_{vision_model_name.replace('/', '_')}_img.pt")

    # Generate text embeddings
    for text_model_name in text_models:
        print(f"Processing text model: {text_model_name}")
        if os.path.exists(f"{embed_dir}/{dataset_name}_{text_model_name}_text.pt"):
            print(f"Loading cached text embeddings from {embed_dir}/{dataset_name}_{text_model_name}_text.pt")
            text_embeddings = torch.load(f"{embed_dir}/{dataset_name}_{text_model_name}_text.pt")
            embeddings[f"{text_model_name}_text"] = text_embeddings
        else:
            text_model = load_text_model(text_model_name, device)
            text_embeddings = generate_text_embeddings(text_model, dataset, device)
            embeddings[f"{text_model_name}_text"] = text_embeddings
            print(f"Text embeddings shape: {text_embeddings.shape}")
            torch.save(text_embeddings, f"{embed_dir}/{dataset_name}_{text_model_name}_text.pt")

    # Step 3: Calculate linear CKA for all pairs
    print("Calculating linear CKA...")
    cka_results = {}
    for vision_model_name in vision_models:
        for text_model_name in text_models:
            key = f"{vision_model_name}_img vs {text_model_name}_text"
            cka_value = linear_CKA(
                embeddings[f"{vision_model_name}_img"].to(device),
                embeddings[f"{text_model_name}_text"].to(device),
                device
            )
            cka_results[key] = cka_value.item()

    # Step 4: Plot the CKA values
    print("Plotting results...")
    plt.figure(figsize=(10, 6))
    plt.bar(cka_results.keys(), cka_results.values())
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Linear CKA")
    plt.title("Linear CKA between Vision and Text Models")
    plt.tight_layout()
    plt.savefig("cka_results.png")
    plt.show()

    # Step 5: Identify the highest CKA pair
    highest_cka_pair = max(cka_results, key=cka_results.get)
    print(f"Highest CKA Pair: {highest_cka_pair} with CKA = {cka_results[highest_cka_pair]:.4f}")

if __name__ == "__main__":
    main()