import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
from config.configer import * 
from tqdm import tqdm

def evaluate(model:nn.Module,test_loader:DataLoader, criterion):
    model.eval()
    test_loss = 0.0 
    correct = 0 
    total = 0
    
    pbar = tqdm(test_loader, desc="Evaluating...")

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(DEVICE["type"]), labels.to(DEVICE["type"])

            output:torch.Tensor = model(images)
            loss = criterion(output, labels)

            test_loss += loss.item()
            _, prediction = torch.max(output, 1)
            correct += prediction.eq(labels.view_as(prediction)).sum().item()
            total += labels.size(0) 

            pbar.set_postfix({"accuracy": f"{100 * correct / total:.2f}"})

    test_loss /= len(test_loader)
    accuracy = 100 * correct / total 

    return test_loss, accuracy


