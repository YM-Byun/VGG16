import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VGG11(nn.Module):
    def __init__(self):
        super(VGG11, self).__init__()

        self.layer1 = nn.Sequential(
            *self.make_conv_layer(in_channels=3, out_channels=64))

        self.layer2 = nn.Sequential(
            *self.make_conv_layer(in_channels=64, out_channels=128))

        self.layer3 = nn.Sequential(
            *self.make_conv_layer(in_channels=128, out_channels=256),
            *self.make_conv_layer(in_channels=256, out_channels=256))

        self.layer4 = nn.Sequential(
            *self.make_conv_layer(in_channels=256, out_channels=512),
            *self.make_conv_layer(in_channels=512, out_channels=512))

        self.layer5 = nn.Sequential(
            *self.make_conv_layer(in_channels=512, out_channels=512),
            *self.make_conv_layer(in_channels=512, out_channels=512))

        self.conv_layers = nn.Sequential(
            self.layer1,
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.layer2,
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.layer3,
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.layer4,
            nn.MaxPool2d(kernel_size=2, stride=2),
            self.layer5,
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(
            nn.Linear(512, 4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 10),
            nn.Softmax(dim=1))

        self.features = nn.Sequential(
            self.conv_layers,
            self.fc)

    def make_conv_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        layer = []

        layer.append(
            nn.Conv2d(in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding = padding))

        layer.append(nn.BatchNorm2d(out_channels))

        layer.append(nn.LeakyReLU())

        return layer

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 512)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    dummy_data = torch.rand(10, 3, 32, 32)
    vgg11 = VGG11()

    print ("VGG 11 network")
    print (vgg11)

    print ("\n--------------------------------------\n")

    x = vgg11(dummy_data)

    print (f"Result: {x.shape}")
