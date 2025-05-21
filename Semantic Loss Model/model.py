import torch.nn as nn
import torchvision

# ---- CNN model with ResNet18 backbone ----
class AttributeCNN(nn.Module):
    def __init__(self):
        super(AttributeCNN, self).__init__()
        
        # Load ResNet-18 and remove the final classification layer
        resnet = torchvision.models.resnet18()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])  # (B, 512, 1, 1)
        
        # Add a linear layer to project to 85-dimensional attribute space
        self.fc = nn.Linear(512, 85)

    def forward(self, x):
        x = self.backbone(x)  # Feature map: (B, 512, 1, 1)
        x = x.view(x.size(0), -1)  # Flatten to (B, 512)
        attr_pred = self.fc(x)  # Output: (B, 85)
        return attr_pred