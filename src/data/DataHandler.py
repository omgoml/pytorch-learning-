from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from config.configer import *

def data_handler():
    """ loading the mnist dataset for the model"""

    #how the sample of the dataset can be transformed into for better training  
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1,0.1)),
    ])
    
    #making a validation and testing purpose data 
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081,)),
    ])
        
    #dowloading the data
    train_data = datasets.MNIST(root=ROOT_PATH,train=True,transform=train_transform,download=True)
    test_data = datasets.MNIST(root=ROOT_PATH, train=False, transform=test_transform, download=True)
    
    #load two set of train and test for training and inference
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKER,
        pin_memory=True if DEVICE["type"] == "cuda" else False
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKER,
        pin_memory=True if DEVICE["type"] == "cuda" else False
    )
    
    return train_loader, test_loader

