from typing import Tuple, List
import torch
import torch.nn as nn

from config import DEVICE, NUM_CLASSES, CLASS_NAMES
from models import MODEL_FACTORY


def load_model(model_name: str, weights_path: str) -> nn.Module:
 
    if model_name not in MODEL_FACTORY:
        raise ValueError(f"Unsupported model name: {model_name}")

    model_class = MODEL_FACTORY[model_name]
    model = model_class(num_classes=NUM_CLASSES)

    state = torch.load(weights_path, map_location=DEVICE)

    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    elif isinstance(state, dict):
        model.load_state_dict(state)
    else:
        model = state

    model.to(DEVICE)
    model.eval()
    return model


def compute_accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def preds_to_class_names(preds: List[int]) -> List[str]:
    return [CLASS_NAMES[i] for i in preds]
