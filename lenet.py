'''
Modified from https://github.com/pytorch/vision.git
'''
import torch
import torch.nn as nn

__all__ = [
    'LeNet',
]


class LeNet(nn.Module):
    '''
    LeNet model
    '''
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(20, 50, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(1250, 500),
            nn.ReLU(True),
            # nn.Linear(256, 256),
            # nn.ReLU(True),
            # nn.Linear(256, 128),
            # nn.ReLU(True),
            nn.Linear(500, 1),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x