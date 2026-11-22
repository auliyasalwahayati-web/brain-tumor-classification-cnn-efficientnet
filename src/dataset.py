from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.config import (
    TRAIN_DIR,
    TEST_DIR,
    CLASS_NAMES,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    VAL_SPLIT,
    SEED,
)

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class BrainTumorDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def collect_samples(root_dir: Path):
    samples = []
    class_to_idx = {class_name: idx for idx, class_name in enumerate(CLASS_NAMES)}

    for class_name in CLASS_NAMES:
        class_dir = root_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Folder kelas tidak ditemukan: {class_dir}")

        label = class_to_idx[class_name]
        for file_path in sorted(class_dir.iterdir()):
            if file_path.is_file() and file_path.suffix.lower() in EXTENSIONS:
                samples.append((file_path, label))

    return samples


def get_transforms(img_size: int = IMG_SIZE):
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, eval_transform


def create_dataloaders(
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    val_split: float = VAL_SPLIT,
    seed: int = SEED,
):
    train_all_samples = collect_samples(TRAIN_DIR)
    test_samples = collect_samples(TEST_DIR)

    labels = [label for _, label in train_all_samples]
    train_samples, val_samples = train_test_split(
        train_all_samples,
        test_size=val_split,
        random_state=seed,
        stratify=labels,
    )

    train_transform, eval_transform = get_transforms(img_size=img_size)

    train_dataset = BrainTumorDataset(train_samples, transform=train_transform)
    val_dataset = BrainTumorDataset(val_samples, transform=eval_transform)
    test_dataset = BrainTumorDataset(test_samples, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader, test_loader