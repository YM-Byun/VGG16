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
 - Best model at @@ epoch
 - Test acc: @@% &nbsp;&nbsp;&nbsp;&nbsp; Test Loss:
 - Val acc: @@% &nbsp;&nbsp; Val Loss:
 
## 4. Test
`` python3 test.py -i <IMAGE_PATH> ``

## Other models' acc
 1. 92.64% (https://github.com/kuangliu/pytorch-cifar)
 2. over 90% (https://www.kaggle.com/xhlulu/vgg-16-on-cifar10-with-keras-beginner)  
   Train Loss: 0.1477067740379274  
   Test Loss: 1.1987001258850098  
   Train F1 Score: 0.9678  
   Test F1 Score: 0.7195  
