import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)
        self.relu = F.relu
        self.l_softmax = F.log_softmax
        self.max_pool = F.max_pool2d

        self.ordered_layers = [
            self.conv1,
            self.conv2,
            self.fc1,
            self.fc2
        ]

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x, 2, 2)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.l_softmax(x, dim=1)
