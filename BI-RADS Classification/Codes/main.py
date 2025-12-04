from config import N_CLASSES, COUNTER
from data_loader import load_datasets, get_data_loaders
from models import get_model, list_available_models
from trainer import train_model
from utils import set_seed, get_device, print_metrics_summary


def main():
    set_seed()

    device = get_device()
    
    print("\n" + "="*60)
    print("Veri Setleri Yükleniyor...")
    print("="*60)
    train_dataset, test_dataset, class_names = load_datasets()
    train_loader, test_loader = get_data_loaders(train_dataset, test_dataset)

    print("\n" + "="*60)
    list_available_models()
    print("="*60)
    

    MODEL_NAME = 'efficientnet_b2' 
    
    print(f"\nSeçilen Model: {MODEL_NAME.upper()}")
    print("="*60)
    
    all_metrics = []
    
    for run in range(COUNTER):
        print(f"\n{'#'*60}")
        print(f"# Eğitim Turu: {run + 1}/{COUNTER}")
        print(f"{'#'*60}")
        
        model = get_model(MODEL_NAME, num_classes=N_CLASSES).to(device)
        
        metrics = train_model(model, train_loader, test_loader, device)
        all_metrics.append(metrics)
    
    if COUNTER > 1:
        print_metrics_summary(all_metrics)


if __name__ == "__main__":
    main()
