import torch.nn as nn
import torch.nn.functional as F
import torch


class OutputLayer(nn.Module):
    def __init__(self, input_size):
        super(OutputLayer, self).__init__()

        self.hair_layer = nn.Sequential(nn.Linear(input_size, 32),
                                        nn.Dropout(.1),
                                        nn.Linear(32, 2),
                                        # nn.Sigmoid(),
                                        # nn.Softmax(dim=1)
                                        )
        self.gender_layer = nn.Sequential(nn.Linear(input_size, 32),
                                          nn.Dropout(.1),
                                          nn.Linear(32, 2),
                                          # nn.Sigmoid(),
                                          # nn.Softmax(dim=1)
                                          )
        self.earring_layer = nn.Sequential(nn.Linear(input_size, 32),
                                           nn.Dropout(.1),
                                           nn.Linear(32, 2),
                                           # nn.Sigmoid(),
                                           # nn.Softmax(dim=1)
                                           )
        self.smile_layer = nn.Sequential(nn.Linear(input_size, 32),
                                         nn.Dropout(.1),
                                         nn.Linear(32, 2),
                                         # nn.Sigmoid(),
                                         # nn.Softmax(dim=1)
                                         )
        self.front_face_layer = nn.Sequential(nn.Linear(input_size, 32),
                                              nn.Dropout(.1),
                                              nn.Linear(32, 2),
                                              # nn.Sigmoid(),
                                              # nn.Softmax(dim=1)
                                              )

    def forward(self, x):
        hair = self.hair_layer(x)
        gender = self.gender_layer(x)
        earring = self.earring_layer(x)
        smile = self.smile_layer(x)
        front_face = self.front_face_layer(x)
        return hair, gender, earring, smile, front_face