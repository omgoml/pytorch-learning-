import torch 
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np 
import os

def data_handler():
    torch.manual_seed(42)
    np.random.seed(42)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    train_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081)),
        transforms.RandomRotation(10),
        transforms.RandomAffine(0, translate=(0.1,0.1))
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,),(0.3081)),
    ])

    os.makedirs("mnist", exist_ok=True)

    train_data = datasets.MNIST(
        root="mnist",
        train=True,
        transform=train_transform,
        download=True,
    )

    test_data = datasets.MNIST(
        root="mnist",
        train=True,
        transform=test_transform,
        download=True,
    )
   
    BATCH_SIZE = 128
    NUM_WORKERS = 4 if torch.cuda.is_available() else 2

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    return (train_loader, test_loader)



