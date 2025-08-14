import torch
import torch.nn as nn 
import torch.optim as optim 
import numpy as np
from config.configer import *
from data.DataHandler import data_handler
from model.cnn import CNNModel
from inference.InferenceModel import InferenceModel
from training.train import TrainingModel

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    IModel = InferenceModel()
    
    try:
        if IModel.model is None:
            train_loader, _ = data_handler()
            model = CNNModel().to(DEVICE["type"])
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(),lr=0.001,weight_decay=WEIGHT_DECAY)
            scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer,
                max_lr=0.01,
                epochs=EPOCHS,
                steps_per_epoch= len(train_loader),
                pct_start=0.3,
            )
            TrainingProgress = TrainingModel(cnn_model=model,criterion=criterion,optimizer=optimizer,scheduler=scheduler)

            TrainingProgress.train()
        else:
            image_path = input("Enter your image path: ")
            predict, confidence, _ = IModel.predict_from_file(image_path)

            print(f"Predicted digit: {predict}")
            print(f"confidence: {confidence:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        
if __name__ == "__main__":
    main()
