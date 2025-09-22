import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

TIN_DIR = "/users/adgs945/sharedscratch/tiny-imagenet-200"  
DATA_DIR = "/users/adgs945/Individual_project_code_/data"
CACHE_TRAIN = os.path.join(DATA_DIR, "tinyimagenet_train_cache.pt")
CACHE_VAL   = os.path.join(DATA_DIR, "tinyimagenet_val_cache.pt")


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975),
                         (0.2302, 0.2265, 0.2262)),
])

def build_cache(split):
    if split == 'train':
        split_dir = os.path.join(TIN_DIR, 'train')
        samples = []
        targets = []
        for class_folder in tqdm(os.listdir(split_dir), desc="Scanning train"):
            class_path = os.path.join(split_dir, class_folder, 'images')
            for img_file in os.listdir(class_path):
                img = Image.open(os.path.join(class_path, img_file)).convert('RGB')
                samples.append(transform(img))
                targets.append(class_folder)

    elif split == 'val':
        split_dir = os.path.join(TIN_DIR, 'val')
        val_ann_path = os.path.join(split_dir, 'val_annotations.txt')
        with open(val_ann_path, 'r') as f:
            lines = f.readlines()
        img_to_label = {line.split('\t')[0]: line.split('\t')[1] for line in lines}

        samples = []
        targets = []
        for img_file, label in tqdm(img_to_label.items(), desc="Scanning val"):
            img = Image.open(os.path.join(split_dir, 'images', img_file)).convert('RGB')
            samples.append(transform(img))
            targets.append(label)
    else:
        raise ValueError("Invalid split")

    # Map labels to ints
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(set(targets)))}
    targets = [class_to_idx[t] for t in targets]

    samples_tensor = torch.stack(samples)
    targets_tensor = torch.tensor(targets, dtype=torch.long)

    return {"samples": samples_tensor, "targets": targets_tensor}

if __name__ == "__main__":
    print("Building train cache...")
    torch.save(build_cache("train"), CACHE_TRAIN)
    print(f"Saved train cache to {CACHE_TRAIN}")

    print("Building val cache...")
    torch.save(build_cache("val"), CACHE_VAL)
    print(f"Saved val cache to {CACHE_VAL}")