import torch 
import torch.nn as nn 
import numpy as np 


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()

        # Start: convolution layers
        self._body = nn.Sequential(
            # Start: first convolution Layer            
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5), # input size = (32, 32), output size = (28, 28)
            nn.ReLU(inplace=True), # ReLU activation
            nn.MaxPool2d(kernel_size=2), # Max pool 2-d
            # End: first convolution Layer

            # Start: second convolution layer
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5), # input size = (14, 14), output size = (10, 10)
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # output size = (5, 5)            
            # End: second convolution layer
        )
        # End: convolution layers
        

        # Start: fully-connected layers
        self._head = nn.Sequential(
            # Start: first fully connected layer            
            nn.Linear(in_features=16 * 5 * 5, out_features=120), # in_features = total number of weight in last conv layer = 16 * 5 * 5
            nn.ReLU(inplace=True),# ReLU activation
            # End: first fully connected layer            

            # Start: second fully-connected layer
            nn.Linear(in_features=120, out_features=84),  # in_features = output of last linear layer = 120 
            nn.ReLU(inplace=True), # ReLU activation
            # End: second fully-connected layer
            
            # Start: Third fully connected layer. It is also output layer
            # in_features = output of last linear layer = 84
            # and out_features = number of classes = 10 (MNIST data 0-9)
            nn.Linear(in_features=84, out_features=10)
            # End: Third fully connected layer.
        )
        # End: fully-connected layers

    def forward(self, x):
        x = self._body(x)           # apply feature extractor
        x = x.view(x.size()[0], -1) # flatten the output of conv layers
                                    # dimension should be batch_size * number_of weight_in_last conv_layer
        x = self._head(x) # apply classification head
        return x

