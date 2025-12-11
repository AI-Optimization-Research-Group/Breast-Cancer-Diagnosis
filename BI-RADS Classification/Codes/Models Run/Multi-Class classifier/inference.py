import argparse
from pathlib import Path
from typing import List

import torch

from config import DEVICE, CLASS_NAMES, SUPPORTED_MODELS, WEIGHT_PATHS
from data_loader import create_test_loader
from utils import load_model, compute_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BIRADS 1/2/4/5 test klasörü üzerinde model doğrulama scripti"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=SUPPORTED_MODELS,
        help="Kullanılacak model ismi",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=False,
        help="Model ağırlık dosyasının yolu (.pth). "
             "Verilmezse config.WEIGHT_PATHS içinden okunur.",
    )
    parser.add_argument(
        "--test-dir",
        type=str,
        required=True,
        help="Test klasörü yolu (BIRADS1/2/4/5 alt klasörlerini içermeli)",
    )
    return parser.parse_args()


def run_inference(
    model_name: str,
    weights_path: str,
    test_dir: str,
) -> None:
  
    test_loader, test_dataset = create_test_loader(test_dir=test_dir)

    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Weights: {weights_path}")
    model = load_model(model_name=model_name, weights_path=weights_path)

    all_preds: List[int] = []
    all_targets: List[int] = []


    samples = test_dataset.samples 

    print("\n[INFO] Tahminler (örnek bazında):")
    print("index\ttrue\tpred\tpath")

    model.eval()
    with torch.no_grad():
        global_index = 0
        for batch_idx, (inputs, targets) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)

            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(targets.cpu().tolist())

            batch_size = inputs.size(0)
            for i in range(batch_size):
                path, true_idx = samples[global_index]
                pred_idx = preds[i].item()
                print(
                    f"{global_index:05d}\t"
                    f"{CLASS_NAMES[true_idx]}\t"
                    f"{CLASS_NAMES[pred_idx]}\t"
                    f"{path}"
                )
                global_index += 1

    all_preds_tensor = torch.tensor(all_preds)
    all_targets_tensor = torch.tensor(all_targets)
    acc = compute_accuracy(all_preds_tensor, all_targets_tensor)
    print(f"\n[RESULT] Toplam örnek: {len(all_targets)}")
    print(f"[RESULT] Accuracy: {acc * 100:.2f} %")

    try:
        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(all_targets, all_preds)
        print("\n[RESULT] Confusion Matrix:")
        print(cm)

        print("\n[RESULT] Classification Report:")
        print(
            classification_report(
                all_targets,
                all_preds,
                target_names=CLASS_NAMES,
                digits=4,
            )
        )
    except ImportError:
        print("\n[INFO] sklearn yüklü değil, confusion matrix ve classification report yazdırılmadı.")


def main() -> None:
    args = parse_args()

    model_name = args.model
    test_dir = args.test_dir

    if args.weights is not None:
        weights_path = args.weights
    else:
        if model_name not in WEIGHT_PATHS:
            raise ValueError(
                f"Ağırlık dosyası yolu belirtilmedi ve config.WEIGHT_PATHS içinde {model_name} için kayıt yok."
            )
        weights_path = WEIGHT_PATHS[model_name]

    # Basit kontrol
    if not Path(test_dir).is_dir():
        raise FileNotFoundError(f"Test klasörü bulunamadı: {test_dir}")
    if not Path(weights_path).is_file():
        raise FileNotFoundError(f"Ağırlık dosyası bulunamadı: {weights_path}")

    run_inference(
        model_name=model_name,
        weights_path=weights_path,
        test_dir=test_dir,
    )


if __name__ == "__main__":
    main()
