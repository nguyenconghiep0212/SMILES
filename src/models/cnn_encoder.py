import torch
import torch.nn as nn
import torchvision.models as models


class ResNet18Encoder(nn.Module):
    def __init__(self, pretrained=True, output_dim=512):
        super().__init__()

        # Load ResNet18
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # Remove final classifier layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-2])

        # Output feature map: (batch, 512, H/32, W/32)
        self.output_dim = output_dim

        # Flatten spatial dims into a sequence for transformer
        # Example: 512 x 8 x 8 => sequence of length 64 with 512-dim tokens

    def forward(self, x):
        """
        x: (B, 3, 256, 256)
        return: (sequence length B*T, 512)
        """

        features = self.encoder(x)
        # features shape: (B, 512, H/32, W/32)

        B, C, H, W = features.shape

        # Convert to sequence: (B, H*W, C)
        seq = features.permute(0, 2, 3, 1).reshape(B, H*W, C)

        return seq  # output tokens for transformer