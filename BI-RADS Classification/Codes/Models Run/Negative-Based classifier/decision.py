from typing import Dict, Any


def decide_from_binary_models(scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    positive_models = []
    negative_models = []

    for cls_name, sc in scores.items():
        neg = float(sc["neg"])
        pos = float(sc["pos"])

        if pos >= neg:
            positive_models.append((cls_name, pos, neg))
        else:
            negative_models.append((cls_name, pos, neg))

    if positive_models:
        positive_models.sort(key=lambda x: x[1], reverse=True)
        best_cls, best_pos, best_neg = positive_models[0]
        via = "positive"
    else:
        negative_models.sort(key=lambda x: x[2])
        best_cls, best_pos, best_neg = negative_models[0]
        via = "all_negative_min_neg"

    return {
        "final_class": best_cls,
        "via": via,
        "best_pos": best_pos,
        "best_neg": best_neg,
        "all_scores": scores,
    }
