import torch
import torchvision.transforms as transforms
import time
import torch.nn as nn

from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from model import VGG11
from datetime import datetime

batch_size=128
momentum=0.9
weight_decay = 0.0005
learning_rate = 0.0001
epochs = 150
is_cuda = torch.cuda.is_available()


def main():
    train_transform = transforms.Compose(
        [transforms.RandomCrop(32, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

    print ("\nLoading Cifar 10 Dataset...")

    train_dataset = CIFAR10(root='./dataset', train=True,
            download=True, transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
            shuffle=True, num_workers=4)

    val_dataset = CIFAR10(root='./dataset', train=False,
            download = True, transform=test_transform)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
            shuffle=False, num_workers=4)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
            'horse', 'ship', 'truck')

    print ("Loaded Cifar 10!\n")

    print ("========================================\n")

    vgg11 = VGG11()

    optimizer = torch.optim.SGD(vgg11.parameters(), lr=learning_rate, momentum=momentum,
            weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0

    if is_cuda:
        vgg11.cuda()
        criterion = criterion.cuda()

    for epoch in range(epochs):
        train(train_loader, vgg11, criterion, optimizer, epoch)

        print ("")

        acc = validate(val_loader, vgg11, criterion, epoch)

        is_best = acc > best_acc

        if is_best:
            best_acc = acc

            torch.save(vgg11.state_dict(), "./weight/best_weight.pth")

        print ("\n========================================\n")

    torch.save(vgg11.state_dict(), "./weight/lastest_weight.pth")


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
        acc1, acc5  = accuracy(outputs, label, topk=(1,5))

        if (i % 50 == 19) or (i == len(train_loader) - 1):
            print (f"Epoch [{epoch+1}/{epochs}] | Train iter [{i+1}/{len(train_loader)}] | acc_top1 = {acc1[0]:.5f} | acc_top5 = {acc5[0]:.5f} | loss = {(running_loss / float(i+1)):.5f}")

def validate(val_loader, model, criterion, epoch):
    model.eval()
    running_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0

    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, label = data

            if is_cuda:
                inputs, label = inputs.cuda(), label.cuda()

            outputs = model(inputs)
            loss = criterion(outputs, label)

            running_loss += loss.item()
            acc1, acc5 = accuracy(outputs, label, topk=(1,5))
            total_acc1 += acc1[0]
            total_acc5 += acc5[0]

    total_acc1 /= len(val_loader)
    total_acc5 /= len(val_loader)

    print (f"Epoch [{epoch+1}/{epochs}] | Validation | acc_top1 = {total_acc1:.5f} | acc_top5 = {total_acc5:.5f} | loss = {(running_loss / float(i)):.5f}")

    return total_acc1

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
