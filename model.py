import torch
import torch.nn as nn
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
            nn.Linear(512, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.Dropout(),
            nn.Linear(256, 10),
            nn.Softmax(dim=1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_conv_layer(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        layer = []

        layer.append(
            nn.Conv2d(in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding = padding))

        layer.append(nn.BatchNorm2d(out_channels))

        layer.append(nn.ReLU(inplace=True))

        return layer

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 512)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    dummy_data = torch.rand(10, 3, 32, 32)

    from pretrain.model import VGG11

    pretrain = VGG11()
    vgg16 = VGG16(pretrain.features)

    print ("VGG 16 network")
    print (vgg16)

    print ("\n--------------------------------------\n")

    x = vgg16(dummy_data)

    print (f"Result: {x.shape}")
