from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import IMAGE_SIZE, BATCH_SIZE, NUM_WORKERS


def get_test_transforms() -> transforms.Compose:

    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225],
        ),
    ])


def create_test_loader(test_dir: str) -> Tuple[DataLoader, datasets.ImageFolder]:

    transform = get_test_transforms()

    dataset = datasets.ImageFolder(
        root=test_dir,
        transform=transform,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,         # Testte shuffle kapalı
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return loader, dataset
