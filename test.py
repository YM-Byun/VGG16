import torch
import torch.nn as nn

from pretrain.model import VGG11
from model import VGG16

def main():
    pt_model = VGG11()
    pt_model.load_state_dict('./pretrain/weight/best_weight.pth')

    vgg16 = VGG16(pt_model.features)
    vgg16.load_state_dict('./weight/best_weight.pth')

    print ("\nLoaded VGG16 network!")

    img = load_img('test.jpg')

    classify(img)


def load_img(path):

def classify(inputs):
    vgg16.eval()

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad()
        outputs = vgg16(inputs)

        print (outputs)
