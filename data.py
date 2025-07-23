import os
import json
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm

def compute_and_cache_mean_std(dataset_path, cache_file='mean_std_cache.json', img_size=224, force_recompute=False):
    cache_key = os.path.abspath(dataset_path)

    # Load from cache if available
    if os.path.exists(cache_file) and not force_recompute:
        with open(cache_file, 'r') as f:
            cache = json.load(f)
        if cache_key in cache:
            print(f"✅ Loaded mean/std from cache: {cache_file}")
            mean, std = cache[cache_key]
            return mean, std

    # Step 1: Define preprocessing (without normalization)
    transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])

    # Step 2: Load dataset
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    # Step 3: Compute mean and std
    print("⏳ Computing mean and std...")
    mean = 0.0
    std = 0.0
    total_images = 0

    for images, _ in tqdm(loader):
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_images += batch_samples

    mean /= total_images
    std /= total_images

    mean = mean.tolist()
    std = std.tolist()
    print(f"✅ Computed Mean: {mean}")
    print(f"✅ Computed Std : {std}")

    # Step 4: Cache results
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            cache = json.load(f)
    else:
        cache = {}
    cache[cache_key] = [mean, std]
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)

    return mean, std


def get_dataset(root, img_size=224, force_recompute=False):
    """
    Returns preprocessed train/test datasets for a custom dataset at `root`.

    Expects:
        - root/train/CLASS_NAME/...
        - root/val/CLASS_NAME/...

    Returns:
        train_data, test_data, class_labels, dataset_mean, dataset_std
    """
    train_dir = os.path.join(root, 'train')
    val_dir = os.path.join(root, 'val')

    # Compute or load mean/std
    dataset_mean, dataset_std = compute_and_cache_mean_std(
        dataset_path=train_dir,
        img_size=img_size,
        force_recompute=force_recompute
    )
    normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)

    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])

    # Load datasets
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data = datasets.ImageFolder(val_dir, transform=test_transform)
    class_labels = train_data.classes

    return train_data, test_data, class_labels, dataset_mean, dataset_std
