from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from config import DEVICE, CLASS_ORDER, NUM_CLASSES
from models import build_model
from decision import decide_from_binary_models


def load_binary_models(
    arch_name: str,
    weight_paths: Dict[str, str],
    device: str = DEVICE,
) -> Dict[str, nn.Module]:
    models_dict: Dict[str, nn.Module] = {}

    for cls_name, ckpt_path in weight_paths.items():
        model = build_model(arch_name, NUM_CLASSES)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        models_dict[cls_name] = model
        print(f"{cls_name} ({arch_name}) model loaded from: {ckpt_path}")

    return models_dict


def collective_predict_batch(
    models_dict: Dict[str, nn.Module],
    x_batch: torch.Tensor,
    class_order: List[str] | None = None,
) -> Tuple[List[str], List[str], List[Dict[str, Dict[str, float]]]]:
    if class_order is None:
        class_order = list(models_dict.keys())

    first_model = next(iter(models_dict.values()))
    device = next(first_model.parameters()).device
    x_batch = x_batch.to(device)
    batch_size = x_batch.size(0)

    raw_outputs: Dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for cls_name in class_order:
            model = models_dict[cls_name]
            out = model(x_batch)
            raw_outputs[cls_name] = out.cpu()

    final_classes: List[str] = []
    via_list: List[str] = []
    all_scores_list: List[Dict[str, Dict[str, float]]] = []

    for i in range(batch_size):
        scores: Dict[str, Dict[str, float]] = {}
        for cls_name in class_order:
            out = raw_outputs[cls_name][i]
            neg = float(out[0].item())
            pos = float(out[1].item())
            scores[cls_name] = {"neg": neg, "pos": pos}

        decision = decide_from_binary_models(scores)
        final_classes.append(decision["final_class"])
        via_list.append(decision["via"])
        all_scores_list.append(decision["all_scores"])

    return final_classes, via_list, all_scores_list


def example_single_inference(models_dict: Dict[str, nn.Module]) -> None:
    x = torch.randn(1, 3, 224, 224, device=DEVICE)

    preds, via_list, all_scores_list = collective_predict_batch(
        models_dict=models_dict,
        x_batch=x,
        class_order=CLASS_ORDER,
    )

    print("Predicted class:", preds[0])
    print("Decision path:", via_list[0])
    print("Scores:", all_scores_list[0])
