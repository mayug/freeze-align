import os
import requests
from zipfile import ZipFile

def download_coco_dataset(destination_folder):
    """
    Downloads the COCO 2017 dataset (train, val, and annotations) and extracts it.
    """
    base_url = "http://images.cocodataset.org"
    files = {
        # "train2017": f"{base_url}/zips/train2017.zip",
        "val2017": f"{base_url}/zips/val2017.zip",
        "annotations": f"{base_url}/annotations/annotations_trainval2017.zip"
    }

    print('Creating destination folder if it does not exist...')
    os.makedirs(destination_folder, exist_ok=True)

    for name, url in files.items():
        print(f"Downloading {name} from {url}...")
        zip_path = os.path.join(destination_folder, f"{name}.zip")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"Extracting {name}...")
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(destination_folder)

        os.remove(zip_path)
        print(f"{name} downloaded and extracted successfully.")

if __name__ == "__main__":
    destination = "../coco_dataset"
    download_coco_dataset(destination)