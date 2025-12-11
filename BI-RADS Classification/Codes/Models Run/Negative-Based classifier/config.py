import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ARCH_NAME = "resnet34"

WEIGHT_PATHS = {
    "birads1": "resnet34_birads1.pt",
    "birads2": "resnet34_birads2.pt",
    "birads4": "resnet34_birads4.pt",
    "birads5": "resnet34_birads5.pt",
}

CLASS_ORDER = ["birads1", "birads2", "birads4", "birads5"]

NUM_CLASSES = 2
