import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        self.layer1 = nn.Sequential(
            *self.make_conv_layer(in_channels=3, out_channels=64),
            *self.make_conv_layer(in_channels=64, out_channels=64),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            *self.make_conv_layer(in_channels=64, out_channels=128),
            *self.make_conv_layer(in_channels=128, out_channels=128),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            *self.make_conv_layer(in_channels=128, out_channels=256),
            *self.make_conv_layer(in_channels=256, out_channels=256),
            *self.make_conv_layer(in_channels=256, out_channels=256),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer4 = nn.Sequential(
            *self.make_conv_layer(in_channels=256, out_channels=512),
            *self.make_conv_layer(in_channels=512, out_channels=512),
            *self.make_conv_layer(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer5 = nn.Sequential(
            *self.make_conv_layer(in_channels=512, out_channels=512),
            *self.make_conv_layer(in_channels=512, out_channels=512),
            *self.make_conv_layer(in_channels=512, out_channels=512),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.conv_layers = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5)

        self.fc = nn.Sequential(
            nn.Linear(7*7*512, 4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 1000),
            nn.Softmax(dim=1))

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
        x = x.view(-1, 7*7*512)
        print (x.shape)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    dummy_data = torch.rand(10, 3, 224, 244)
    vgg16 = VGG16()

    print ("VGG 16 network")
    print (vgg16)

    print ("\n--------------------------------------\n")

    x = vgg16(dummy_data)

    print (f"Result: {x.shape}")
