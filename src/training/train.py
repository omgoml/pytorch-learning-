import torch 
import torch.nn as nn 
import torch.optim as optim 
from data.DataHandler import data_handler
from training.evaluate import evaluate
from config.configer import *
from tqdm import tqdm
from utils.utils import ModelSaver 

class TrainingModel:
    def __init__(self,cnn_model:nn.Module, optimizer: optim.Optimizer, criterion, scheduler: optim.lr_scheduler.OneCycleLR, epochs:int = EPOCHS) -> None:
        self.train_loader, self.test_loader = data_handler()
        self.epochs = epochs
        
        self.history = {
            "train_losses": [],
            "train_accuracies": [],
            "test_losses": [],
            "test_accuracies": []
        }

        self.model = cnn_model      
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion

    def train_epoch(self):
        self.model.train()
        training_loss = 0.0 
        correct = 0 
        total = 0 

        pbar = tqdm(self.train_loader, desc="Training...")

        for images, labels in pbar:
            images, labels = images.to(DEVICE["type"],non_blocking=True), labels.to(DEVICE["type"],non_blocking=True)

            self.optimizer.zero_grad()
            output = self.model(images)
            loss = self.criterion(output, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()
            self.scheduler.step()

            training_loss += loss.item()
            _, prediction = torch.max(output, 1)
            correct += prediction.eq(labels.view_as(prediction)).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Accuracy":f"{100 * correct / total:.2f}",
                "LR": f"{self.scheduler.get_last_lr()[0]:.6f}"
            })

        return training_loss / len(self.train_loader), 100 * correct/total

    def train(self):
        patience = 5
        patience_counter = 0
        best_accuracy = 0.0 

        for epoch in range(self.epochs):
            print(f"\n Epoch {epoch + 1} / {self.epochs}")
            
            train_loss, train_accuracy = self.train_epoch()
            self.history["train_losses"].append(train_loss)
            self.history["train_accuracies"].append(train_accuracy)

            test_loss, test_accuracy = evaluate(self.model, self.test_loader, self.criterion)
            self.history["test_losses"].append(test_loss)
            self.history["test_accuracies"].append(test_accuracy)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
            print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy
                ModelSaver(model=self.model,optimizer=self.optimizer,test_accuracy=best_accuracy,epoch=epoch)
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter == patience:
                print(f"Early stopping triggered after {epoch + 1} epochs") 
                break

