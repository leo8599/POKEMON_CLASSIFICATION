"""
Contains PyTorch model code to instantiate TinyVGG and EfficientNet models.
"""
import torch
from torch import nn
import torchvision

class TinyVGG(nn.Module):
    """
    Creates the TinyVGG architecture.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # The input to Linear layer depends on the image size. 
            # For 64x64 input in TinyVGG, this math holds. 
            nn.Linear(in_features=hidden_units*13*13, out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

def create_efficientnet_b0(output_shape: int, device: torch.device):
    """
    Creates an EfficientNet_B0 feature extractor and a new classification head.
    """
    # 1. Get the base weights (pretrained on ImageNet)
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    
    # 2. Instantiate the model
    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    # 3. Freeze the base layers (so we don't destroy the patterns it already learned)
    for param in model.features.parameters():
        param.requires_grad = False

    # 4. Recreate the classifier head
    # EfficientNet_B0 has a 'classifier' block (Dropout + Linear). We replace it.
    torch.manual_seed(42)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280, # 1280 is the output size of EfficientNet_B0 features
                        out_features=output_shape) 
    ).to(device)
    
    return model