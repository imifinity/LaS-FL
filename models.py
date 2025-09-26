import torch.nn as nn
import torchvision.models as models

class ResNet(nn.Module):
    """
    ResNet wrapper for CIFAR-10 and Tiny-ImageNet.
    Supports ResNet-18 and ResNet-50.
    """
    def __init__(self, depth=18, n_classes=10, pretrained=False):
        """
        Args:
            depth (int): ResNet depth (18 or 50)
            n_classes (int): Number of output classes (default 10 for CIFAR-10)
            pretrained (bool): Whether to load ImageNet pretrained weights
        """
        
        super(ResNet, self).__init__()
        
        # Choose ResNet depth
        if depth == 18:
            if pretrained:
                self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.model = models.resnet18(weights=None)
        elif depth == 50:
            if pretrained:
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            else:
                self.model = models.resnet50(weights=None)
        else:
            raise ValueError("Unsupported depth. Use 18 or 50.")
        
        # Modify conv1 and remove maxpool for small images
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        
        # Adjust final fc layer for n_classes
        self.model.fc = nn.Linear(self.model.fc.in_features, n_classes)

    def forward(self, x):
        """Forward pass through the ResNet backbone"""
        return self.model(x)