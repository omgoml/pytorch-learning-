import torch 
import torch.nn as nn 
import torch.optim as optim
import os
from CNN import CNN 
from evaluate import evaluate
from training import train_epoch
from  DataHandler import data_handler 
from utils import load_model, save_model

train_loader, test_loader = data_handler()

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CNN().to(device)

LOAD_EXISTING = False
MODEL_PATH = os.path.join("mnist","best_model.pth")

if LOAD_EXISTING and os.path.exists(MODEL_PATH):
    loaded_model, checkpoint = load_model(model_path=MODEL_PATH,device=device)
    if load_model is not None: 
        model = loaded_model
        print("Loaded existing model")
        
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=0.01,
    epochs=20,
    steps_per_epoch=len(train_loader),
    pct_start=0.3
)

EPOCHS = 20 
train_losses, train_accuracies = [], [] 
test_losses, test_accuracies = [], []
best_test_accuracies = 0 
patience = 5 
patience_counter = 0 

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1} / {EPOCHS}")
    
    train_loss, train_accuracy = train_epoch(model,train_loader,criterion, optimizer, scheduler, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracies)

    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
    print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

    if test_accuracy > best_test_accuracies:
        best_test_accuracies = test_accuracy
        save_model(model, optimizer,epoch,test_accuracy,MODEL_PATH)
        paitence_couter = 0
        print(f"New best model saved! Test Acc: {test_accuracy:.2f}%")
    else:
        patience_counter += 1
        
    if patience_counter >= patience:
        print(f"Early stopping triggered after {epoch + 1} epochs")
        break
