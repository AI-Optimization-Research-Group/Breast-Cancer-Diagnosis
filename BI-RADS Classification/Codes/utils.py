import torch
import numpy as np
import random
from config import SEED

def set_seed(seed=SEED):
    """
    Random seed ayarlar (tekrarlanabilirlik için)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed ayarlandı: {seed}")


def get_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")
    return device


def print_metrics_summary(metrics_list):

    if len(metrics_list) == 0:
        print("Henüz metrik bulunmuyor.")
        return
    
    avg_acc = np.mean([m['accuracy'] for m in metrics_list])
    avg_prec = np.mean([m['precision'] for m in metrics_list])
    avg_rec = np.mean([m['recall'] for m in metrics_list])
    avg_f1 = np.mean([m['f1_score'] for m in metrics_list])
    
    print("\n" + "="*60)
    print("Tüm Eğitimlerin Ortalama Test Metrikleri")
    print("="*60)
    print(f"Ortalama Accuracy:  {avg_acc:.4f}")
    print(f"Ortalama Precision: {avg_prec:.4f} (Weighted)")
    print(f"Ortalama Recall:    {avg_rec:.4f} (Weighted)")
    print(f"Ortalama F1 Score:  {avg_f1:.4f} (Weighted)")
    print("="*60 + "\n")
