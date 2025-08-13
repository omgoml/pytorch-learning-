import torch 
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, dropout_rate=0.3) -> None:
        super(CNN, self).__init__()
        
        self.features = nn.Sequential(
            #first convolutional block
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(dropout_rate),

            #second convolutional block 
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(dropout_rate),

            #Third convolutional block
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            #converting in to a 4x4 output regarless of the input size
            nn.AdaptiveAvgPool2d(output_size=(4,4))
        )
        
        #classifier layers (fully connected layers)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 10)
        )

    def forward(self, input_data):
        input_data = self.features(input_data)
        input_data = torch.flatten(input_data,1)
        input_data = self.classifier(input_data)

        return input_data
