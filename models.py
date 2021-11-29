## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Network(nn.Module):

    def __init__(self, input_size, flatten_input, output_size,  Conv_layers, filters, conv_dropout, Dense_layers, dense_dropout):
        
        super(Network, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # Defining the Conv Layer 
        
        self.Conv_layers = nn.ModuleList([nn.Conv2d(input_size, Conv_layers[0], filters[0])])
        
        Conv_layers_params = zip(Conv_layers[:-1], Conv_layers[1:], filters[1:])
        
        self.Conv_layers.extend([nn.Conv2d(c1, c2, f) for c1 ,c2, f in Conv_layers_params])
        
        
        # Defining the max - pool layer 
        
        self.maxPool = nn.MaxPool2d(2, stride = 2)
        
        # Creating the list for the Dense layers 
        
        self.Dense_layers = nn.ModuleList([nn.Linear(flatten_input, Dense_layers[0])])
        
        Dense_layer_size = zip(Dense_layers[:-1], Dense_layers[1:])
        
        self.Dense_layers.extend([nn.Linear(h1, h2) for h1, h2 in Dense_layer_size])
        
        
        #Defining the Convolution dropout layer
        
        self.Conv_Dropout = nn.ModuleList([nn.Dropout(p = conv_dropout[0])])                 
        
        self.Conv_Dropout.extend([nn.Dropout(p = i) for i in conv_dropout[1:]])
        
         #Defining the Dense dropout layer
        
        self.Dense_Dropout = nn.ModuleList([nn.Dropout(p = dense_dropout[0])])                 
        
        self.Dense_Dropout.extend([nn.Dropout(p = j) for j in dense_dropout[1:]])
        
        
        # Defining the ouput layer
                               
        self.output = nn.Linear(Dense_layers[-1], output_size)                       
 
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
         
        for conv, conv_drop in zip(self.Conv_layers, self.Conv_Dropout):
            
            x = F.relu(conv(x))
            
            x = self.maxPool(x)
            
            x = conv_drop(x)
        
        x = x.view(x.size(0), -1)
        
        for dense, den_drop in zip(self.Dense_layers, self.Dense_Dropout):
            
            x  = F.relu(dense(x))
            

            x = den_drop(x)
           
    
        x = self.output(x)

        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

    