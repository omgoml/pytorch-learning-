import torch 
import torch.nn as nn 
import torch.optim as optim
import os
import matplotlib.pyplot as plt
from CNN import CNN 
from evaluate import evaluate
from training import train_epoch
from  DataHandler import data_handler 
from utils import load_model, save_model

def main():
    train_loader, test_loader = data_handler()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CNN().to(device)


    LOAD_EXISTING = True
    MODEL_PATH = os.path.join("mnist","best_model.pth")

    if LOAD_EXISTING and os.path.exists(MODEL_PATH):
        loaded_model, checkpoint = load_model(model_path=MODEL_PATH,device=device)
        if loaded_model is not None: 
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
        test_accuracies.append(test_accuracy)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

        if test_accuracy > best_test_accuracies:
            best_test_accuracies = test_accuracy
            save_model(model, optimizer,epoch,test_accuracy,MODEL_PATH)
            patience_counter = 0 
            print(f"New best model saved! Test Acc: {test_accuracy:.2f}%")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(test_losses, label='Test Loss', color='red')
    plt.title('Training and Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Train Accuracy', color='blue')
    plt.plot(test_accuracies, label='Test Accuracy', color='red')
    plt.title('Training and Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 3, 3)
    # Plot learning rate schedule
    lrs = []
    temp_optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    temp_scheduler = optim.lr_scheduler.OneCycleLR(
        temp_optimizer, 
        max_lr=0.01,
        epochs=len(train_losses),
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    for _ in range(len(train_losses) * len(train_loader)):
        lrs.append(temp_scheduler.get_last_lr()[0])
        temp_scheduler.step()

    # Sample learning rates to match epochs
    sampled_lrs = [lrs[i * len(train_loader)] for i in range(len(train_losses))]
    plt.plot(sampled_lrs)
    plt.title('Learning Rate Schedule')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


