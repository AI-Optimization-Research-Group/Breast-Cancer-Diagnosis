import torch.nn as nn
from torchvision import models


class DenseNet121Model(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet121Model, self).__init__()
        self.model = models.densenet121(pretrained=True)
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)