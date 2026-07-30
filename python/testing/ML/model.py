import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, n_classes, n_mels=40):
        super().__init__()

        def dsconv(cin, cout, stride=1):
            return nn.Sequential(
                nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
                nn.BatchNorm2d(cin), nn.ReLU(inplace=True),
                nn.Conv2d(cin, cout, 1, bias=False),
                nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )

        self.stem = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1, bias=False), nn.BatchNorm2d(8), nn.ReLU()
        )
        self.blocks = nn.Sequential(
            dsconv(8, 16, stride=2),
            dsconv(16, 16),           # frozen during fine-tuning
            dsconv(16, 32, stride=2),  # fine-tuned
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(32, n_classes)  # always retrained

    def forward(self, x):  # x: [B, 1, n_mels, T]
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.drop(x))
