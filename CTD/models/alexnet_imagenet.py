import torch
import torchvision
from torch import nn as nn

from utils.builder import MODELS

@MODELS.register_module()
class AlexNetImageNet(torch.nn.Module):

    def __init__(self):
        super(AlexNetImageNet, self).__init__()
        alexnet = torchvision.models.alexnet(pretrained=True)
        self.backbone = torch.nn.Sequential(*(list(alexnet.children())[:-1]))
        self.fc = torch.nn.Sequential(*(list(alexnet.classifier.children())[:-1]))
        self.register_buffer(
            'mean',
            torch.tensor([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1))
        self.register_buffer(
            'std',
            torch.tensor([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1))

        # self.temp = torch.nn.Linear(4096, 256)
        # self.feature_dim = 256
        self.feature_dim = 4096

    def forward(self, x: torch.Tensor):
        x = (x - self.mean) / self.std
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # x = self.temp(x)
        return x

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    alex_net = AlexNetImageNet().to(device)
    x = torch.zeros(1, 3, 224, 224).to(device)
    y = torch.zeros(1, 3, 224, 224).to(device)
    f1 = alex_net(x)
    f2 = alex_net(y)
