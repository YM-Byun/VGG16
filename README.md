# VGG16

## 1. Pre training
 - Using VGG 11
 - Cifar 10 dataset
 - learning rate 0.001, 74 epochs
 
## 2. Trainning
 - Using VGG 16
 - Cifar 10 dataset, RandomCrop, RandomHorizontalFlip 
 - learning rate 0.001 for 1~85 epoch
 - learning rate 0.0001 for 86 ~ 170 epoch
 - learning rate 0.00001 for 170~ 240 epoch

## 3. Result
 - Best model at 233 epoch
 - Test acc: 95.80% &nbsp;&nbsp;&nbsp;&nbsp; Test Loss: 1.504
 - Val acc: 87.51% &nbsp;&nbsp; Val Loss: 1.6282
 
## 4. Test
`` python3 test.py -i <IMAGE_PATH> ``

## Other models' acc
 1. 92.64% (https://github.com/kuangliu/pytorch-cifar)
 2. over 90% (https://www.kaggle.com/xhlulu/vgg-16-on-cifar10-with-keras-beginner)  
