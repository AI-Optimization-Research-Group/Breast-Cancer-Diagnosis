from Models.ResNet_50 import ResNet50Model
from Models.ResNet_34 import ResNet34Model
from Models.VGG_16 import VGG16Model
from Models.EfficientNet_B2 import EfficientNetB2Model
from Models.DenseNet_121 import DenseNet121Model


AVAILABLE_MODELS = {
    'resnet50': ResNet50Model,
    'resnet34': ResNet34Model,
    'vgg16': VGG16Model,
    'efficientnet_b2': EfficientNetB2Model,
    'densenet121': DenseNet121Model
}


def get_model(model_name, num_classes):
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Geçersiz model adı: {model_name}. Kullanılabilir modeller: {list(AVAILABLE_MODELS.keys())}")
    
    model_class = AVAILABLE_MODELS[model_name]
    return model_class(num_classes)


def list_available_models():
    print("Kullanılabilir Modeller:")
    for i, model_name in enumerate(AVAILABLE_MODELS.keys(), 1):
        print(f"  {i}. {model_name}")
