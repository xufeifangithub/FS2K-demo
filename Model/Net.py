import torch
import torch.nn as nn
from Model.OutputLayer import OutputLayer


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 9, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(9, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(37 * 37 * 48, 512),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            nn.Linear(512, 120),
            nn.ReLU(inplace=True),
            # nn.Dropout(0.5),
            # nn.Linear(1024, 120),
        )
        self.output_layer = OutputLayer(120)

    def forward(self, x):
        # x = x.view((-1, x.shape[3], x.shape[1], x.shape[2]))
        x = self.features(x)
        # print(x.shape)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        hair, gender, earring, smile, front_face = self.output_layer(x)
        return hair, gender, earring, smile, front_face
