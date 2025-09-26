import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


# Paths (adjust these if running in a different environment)
TIN_DIR = "/users/adgs945/sharedscratch/tiny-imagenet-200" # TinyImageNet root
DATA_DIR = "/users/adgs945/Individual_project_code_/data" # Local cache directory
CACHE_TRAIN = os.path.join(DATA_DIR, "tinyimagenet_train_cache.pt")
CACHE_VAL   = os.path.join(DATA_DIR, "tinyimagenet_val_cache.pt")


# Image preprocessing - convert to tensor and normalise using dataset statistics
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4802, 0.4481, 0.3975),
                         (0.2302, 0.2265, 0.2262)),
])

def build_cache(split):
    """
    Build TinyImageNet cache for faster data loading.

    Args:
        split (str): "train" or "val"

    Returns:
        dict: Dictionary containing:
            - "samples": Tensor of shape (N, 3, H, W) with preprocessed images
            - "targets": LongTensor of shape (N,) with integer labels
    """
    if split == 'train':
        split_dir = os.path.join(TIN_DIR, 'train')
        samples, targets = [], []

        # Loop through all class folders
        for class_folder in tqdm(os.listdir(split_dir), desc="Scanning train"):
            class_path = os.path.join(split_dir, class_folder, 'images')

            # Load and preprocess each image
            for img_file in os.listdir(class_path):
                img = Image.open(os.path.join(class_path, img_file)).convert('RGB')
                samples.append(transform(img))
                targets.append(class_folder)

    elif split == 'val':
        split_dir = os.path.join(TIN_DIR, 'val')
        val_ann_path = os.path.join(split_dir, 'val_annotations.txt')

        # Load validation labels
        with open(val_ann_path, 'r') as f:
            lines = f.readlines()
        img_to_label = {line.split('\t')[0]: line.split('\t')[1] for line in lines}

        samples, targets = [], []
        for img_file, label in tqdm(img_to_label.items(), desc="Scanning val"):
            img = Image.open(os.path.join(split_dir, 'images', img_file)).convert('RGB')
            samples.append(transform(img))
            targets.append(label)
    else:
        raise ValueError("Invalid split (must be 'train' or 'val')")

    # Map labels to ints
    class_to_idx = {cls: idx for idx, cls in enumerate(sorted(set(targets)))}
    targets = [class_to_idx[t] for t in targets]

    # Stack into tensors for efficient storage
    samples_tensor = torch.stack(samples)
    targets_tensor = torch.tensor(targets, dtype=torch.long)

    return {"samples": samples_tensor, "targets": targets_tensor}

if __name__ == "__main__":
    # Build and save training cache
    print("Building train cache...")
    torch.save(build_cache("train"), CACHE_TRAIN)
    print(f"Saved train cache to {CACHE_TRAIN}")

    # Build and save validation cache
    print("Building val cache...")
    torch.save(build_cache("val"), CACHE_VAL)
    print(f"Saved val cache to {CACHE_VAL}")