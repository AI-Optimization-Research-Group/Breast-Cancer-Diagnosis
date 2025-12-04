import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetB2Model(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB2Model, self).__init__()
        self.model = EfficientNet.from_pretrained('efficientnet-b2')
        in_features = self.model._fc.in_features
        self.model._fc = nn.Identity()
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        x = self.model.extract_features(x)
        x = self.model._avg_pooling(x)
        x = x.flatten(start_dim=1)
        x = self.classifier(x)
        return x