'''
Modified from https://github.com/pytorch/vision.git
'''
import torch
import torch.nn as nn

__all__ = [
    'SmallNetR', 'SmallNetC'
]


class SmallNetR(nn.Module):
    '''
    SmallNet model
    '''
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(44944, 100),
            nn.ReLU(True),
            # nn.Linear(256, 256),
            # nn.ReLU(True),
            # nn.Linear(256, 128),
            # nn.ReLU(True),
            nn.Linear(100, 1),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class SmallNet(nn.Module):
    '''
    SmallNet model
    '''
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            # nn.Dropout(),
            nn.Linear(400, 100),
            nn.ReLU(True),
            # nn.Linear(256, 256),
            # nn.ReLU(True),
            # nn.Linear(256, 128),
            # nn.ReLU(True),
            nn.Linear(100, 5),
            nn.ReLU(True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        # x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x