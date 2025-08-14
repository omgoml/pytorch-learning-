import torch 
import os
import torch.nn as nn
from torch.optim import Optimizer 
from config.configer import *
from model.cnn import CNNModel

def ModelLoader():
    if not os.path.exists(MODEL_PATH):
        print(f"Model file {MODEL_PATH} not exit")
        return (None, None)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE["type"])

    model = CNNModel().to(DEVICE["type"])
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Model loaded from {MODEL_PATH}")
    print(f"Best test accuracy: {checkpoint['test_accuracy']:.2f}%")
    print(f"Trained for {checkpoint['epoch'] + 1} epochs")

    return model, checkpoint

def ModelSaver(model:nn.Module, optimizer: Optimizer, test_accuracy, epoch):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer":optimizer.state_dict(),
        "test_accuracy":test_accuracy,
    },MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")

