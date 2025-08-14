import torch 
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self, dropout_rate:float = 0.3) -> None:
        super().__init__()

        self.features = nn.Sequential(
            #first convolutional block 
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #Second convolutional Block 
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64,out_channels=64,kernel_size=3,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate),
            nn.MaxPool2d(kernel_size=2,stride=2),

            #third convolutional Block 
            nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate * 0.5),
                    
            nn.AdaptiveAvgPool2d(output_size=((4,4))),
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=128 * 4 * 4, out_features=512),
            nn.BatchNorm1d(num_features=512),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=dropout_rate),
            nn.Linear(in_features=512, out_features=256),
            nn.BatchNorm1d(num_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout1d(p=dropout_rate * 0.5),
            nn.Linear(in_features=256, out_features=10),
        )
        #initialize weight and bias 
        self._initialize_parameters()

    def _initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.01)
                nn.init.constant_(m.bias,0)

    def forward(self, input_data: torch.Tensor):
        output = self.features(input_data)
        output = output.view(output.size(0),-1) 
        output = self.classifier(output)
        return output


