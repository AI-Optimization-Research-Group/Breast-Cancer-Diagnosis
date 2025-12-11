import torch

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

NUM_CLASSES: int = 4
CLASS_NAMES = ["BIRADS1", "BIRADS2", "BIRADS4", "BIRADS5"]


IMAGE_SIZE: int = 224          
BATCH_SIZE: int = 16
NUM_WORKERS: int = 4           


SUPPORTED_MODELS = ["densenet121", "efficientnet_b2", "resnet34", "vgg16"]


WEIGHT_PATHS = {
    "densenet121": "weights/densenet121_birads.pth",
    "efficientnet_b2": "weights/efficientnet_b2_birads.pth",
    "resnet34": "weights/resnet34_birads.pth",
    "vgg16": "weights/vgg16_birads.pth",
}
