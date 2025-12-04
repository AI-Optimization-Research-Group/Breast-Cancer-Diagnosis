import torch.nn as nn
from torchvision import models

class VGG16Model(nn.Module):
    def __init__(self, num_classes):
        super(VGG16Model, self).__init__()
        self.model = models.vgg16(pretrained=True)
        in_features = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
