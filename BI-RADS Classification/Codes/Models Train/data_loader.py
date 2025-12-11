import os
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from config import DATA_DIR, IMG_SIZE, BATCH_SIZE, MEAN, STD


def get_data_transforms():
    data_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    return data_transform


def load_datasets():
    data_transform = get_data_transforms()
    
    try:
        train_dataset = ImageFolder(
            root=os.path.join(DATA_DIR, 'train'),
            transform=data_transform
        )
        test_dataset = ImageFolder(
            root=os.path.join(DATA_DIR, 'test'),
            transform=data_transform
        )
        
        print(f"Eğitim Setindeki Sınıf İsimleri: {train_dataset.classes}")
        print(f"Toplam Eğitim Görüntüsü: {len(train_dataset)}")
        print(f"Toplam Test Görüntüsü: {len(test_dataset)}")
        
        return train_dataset, test_dataset, train_dataset.classes
        
    except FileNotFoundError:
        print(f"\n[HATA] Veri setleri '{DATA_DIR}/train' ve '{DATA_DIR}/test' klasörlerinde bulunamadı.")
        print("Lütfen DATA_DIR değişkenini ve klasör yapısını kontrol edin.")
        exit()


def get_data_loaders(train_dataset, test_dataset):
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    return train_loader, test_loader
