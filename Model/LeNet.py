import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.OutputLayer import OutputLayer


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=3,
                                             out_channels=18,
                                             kernel_size=5,
                                             padding=2),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, stride=2))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=18,
                                             out_channels=48,
                                             kernel_size=5),
                                   nn.ReLU(),
                                   nn.MaxPool2d(2, stride=2))

        self.fc1 = nn.Linear(in_features=35 * 35 * 48, out_features=512)
        self.fc2 = nn.Linear(512, 120)
        # self.fc3 = nn.Linear(120, 84)
        # self.fc4 = nn.Linear(84, 2)
        self.output_layer = OutputLayer(120)

    def forward(self, x):
        # print(x.shape)
        # x = x.view((-1, x.shape[3], x.shape[1], x.shape[2]))
        x = self.conv1(x)
        x = self.conv2(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        hair, gender, earring, smile, front_face = self.output_layer(x)
        return hair, gender, earring, smile, front_face

