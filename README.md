# VGG16

## 1. Dataset
 - Train dataset: Cifar 10 dataset with 2 augmentation
 
## 2. Trainning
 - Using VGG 16
 - learning rate 0.01 for 1~95 epoch
 - learning rate 0.001 for 96 ~ 110 epoch
 - learning rate 0.0001 for 110~ 150 epoch

## 3. Result
 - Best model at 115 epoch
 - Test acc: 96.67% &nbsp;&nbsp;&nbsp;&nbsp; Test Loss: 0.13332
 - Val acc: 93.06% &nbsp;&nbsp; Val Loss: 0.2888
 
## 4. Test
`` python3 test.py -i <IMAGE_PATH> ``

## Other models' acc
 1. 92.64% (https://github.com/kuangliu/pytorch-cifar)
 2. over 90% (https://www.kaggle.com/xhlulu/vgg-16-on-cifar10-with-keras-beginner)  
