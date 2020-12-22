## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        #ref https://arxiv.org/pdf/1710.00977.pdf
        #NaimishNet consists of 4 convolution2d layers,
        #4 maxpool-ing2d layers and 3 dense layers,
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
                
        ## Note that among the layers to add, consider including:
        # maxpooling layers - V
        # multiple conv layers - V
        # fully-connected layers V
        ##and other layers (such as 
        # dropout - V
        # or batch normalization -V
        ##to avoid overfitting
        
        #img 224x224
        #input image width/height, W, minus the filter size, F, divided by the stride, S, all + 1
        ## output size = (W-F)/S +1
        
        # Dropouts NaimishNet
        
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        ## output size = (W-F)/S +1 = (110-5)/1 +1 = 53
        ## output size = (W-F)/S +1 = (53-3)/1 +1 = 25
        ## output size = (W-F)/S +1 = (25-3)/1 +1 = 11
        
        ## output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        ## output size = (W-F)/S +1 = (110-3)/1 +1 = 54
        ## output size = (W-F)/S +1 = (54-3)/1 +1 = 26
        ## output size = (W-F)/S +1 = (26-2)/1 +1 = 12
        
        #Dropouts from 0.1 to 0.6 #changed
        #1
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool = nn.MaxPool2d(2, 2) # 1st good res was without MaxPool here
        self.bn2 = nn.BatchNorm2d(32)
        self.fc1_drop = nn.Dropout(p=0.1)
        #2
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc2_drop = nn.Dropout(p=0.2)
        #3
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc3_drop = nn.Dropout(p=0.2) #from 0.3 to 0.2
        #4
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn2 = nn.BatchNorm2d(256)
        self.fc4_drop = nn.Dropout(p=0.3) #from 0.4 to 0.3
        
        #256*11*11 = 30976
        #256*12*12 = 36864
        self.fc1 = nn.Linear(36864, 1000)
        self.fc5_drop = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(1000, 1000)
        self.fc6_drop = nn.Dropout(p=0.6)
        
        # finally, create 136 output channels (for the 10 classes)
        self.fc3 = nn.Linear(1000, 136) 
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        # conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.fc1_drop(x) 
        x = self.pool(F.relu(self.conv2(x)))
        x = self.fc2_drop(x) 
        x = self.pool(F.relu(self.conv3(x)))
        x = self.fc3_drop(x) 
        x = self.pool(F.relu(self.conv4(x)))
        x = self.fc4_drop(x) 
                      
        # prep for linear layer
        # this line of code is the equivalent of Flatten in Keras
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc5_drop(x)
        x = F.relu(self.fc2(x))
        x = self.fc6_drop(x)
        x = self.fc3(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
