import torch
import torchvision.transforms as transforms
import time
import sys
import torch.nn as nn

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from pretrain.model import VGG11
from model import VGG16

batch_size=256
momentum=0.9
weight_decay = 0.0005
learning_rate = 0.001
epochs = 350
is_cuda = torch.cuda.is_available()

class Adaptive_lr:
    def __init__(self, optimizer, lr):
        self.acc = []
        self.optimizer = optimizer
        self.update_cnt = 0
        self.lr = lr

    def save_acc(self, acc):
        self.acc.append(acc)

        if self.acc.count(acc) > 4:
            self.update_lr()
            self.acc.clear()
            self.update_cnt += 1
            print (f"\nLeaning Rate update to {self.lr}!")

        if len(self.acc) > 6:
            del self.acc[0]

    def update_lr(self):
        self.lr *= 0.1
        for param in self.optimizer.param_groups:
            param['lr'] = self.lr

    def get_update_cnt(self):
        return self.update_cnt

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    print ("\nLoading Cifar 10 Dataset...")

    train_dataset = CIFAR10(root='./dataset', train=True,
            download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=4)

    val_dataset = CIFAR10(root='./dataset', train=False,
            download = True, transform=transform)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck')

    print ("Loaded Cifar 10!\n")

    print ("========================================\n")

    pretrained_model = VGG11()
    pretrained_weight = torch.load('./pretrain/weight/best_weight.pth')
    pretrained_model.load_state_dict(pretrained_weight)
    
    print ("Loaded pretrained weight!")
    
    print ("\n========================================\n")

    vgg16 = VGG16(pretrained_model.features)

    optimizer = torch.optim.SGD(vgg16.parameters(), lr=learning_rate, momentum=momentum,
            weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    adaptive_lr = Adaptive_lr(optimizer, learning_rate)

    best_loss = 9.0

    if is_cuda:
        vgg16.cuda()
        criterion = criterion.cuda()

    for epoch in range(epochs):
        if adaptive_lr.get_update_cnt() > 3:
            sys.exit(0)

        train(train_loader, vgg16, criterion, optimizer, epoch)

        print ("")

        acc, loss = validate(val_loader, vgg16, criterion, epoch)

        adaptive_lr.save_acc(acc)

        is_best = loss < best_loss

        if is_best:
            best_loss = loss

            torch.save(vgg16.state_dict(), "./weight/best_weight.pth")

        print ("\n========================================\n")

        torch.save(vgg16.state_dict(), "./weight/lastest_weight.pth")

def train(train_loader, model, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(train_loader):
        inputs, label = data

        if is_cuda:
            inputs, label = inputs.cuda(), label.cuda()

        optimizer.zero_grad()

        outputs =  model(inputs)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        acc = accuracy(outputs, label)

        if (i % 20 == 19) or (i == len(train_loader) - 1):
            print (f"Epoch [{epoch+1}/{epochs}] | Train iter [{i+1}/{len(train_loader)}] | acc = {acc[0][0]:.5f} | loss = {(running_loss / float(i+1)):.5f}")

def validate(val_loader, model, criterion, epoch):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, label = data

            if is_cuda:
                inputs, label = inputs.cuda(), label.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, label)

            running_loss += loss.item()
            acc = accuracy(outputs, label)

            if (i % 10 == 9) or (i == len(val_loader) - 1):
                print (f"Epoch [{epoch+1}/{epochs}] | Val iter [{i+1}/{len(val_loader)}] | acc = {acc[0][0]:.5f} | loss = {(running_loss / float(i)):.5f}")

        return acc[0][0], (running_loss / float(i))

def accuracy(output, label, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = label.size(0)

        _,pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(label.view(1,-1).expand_as(pred))

        res = []

        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0/batch_size))

        return res


if __name__ == '__main__':
    main()
