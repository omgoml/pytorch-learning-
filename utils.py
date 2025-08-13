import torch 
import os 
from CNN import CNN

def load_model(model_path, device):
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not exists")
        return (None, None) 

    checkpoint = torch.load(model_path, map_location=device)

    model = CNN().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print(f"Model loaded from {model_path}")
    print(f"Best test accuracy: {checkpoint['test_accuracy']:.2f}%")
    print(f"Trained for {checkpoint['epoch'] + 1} epochs")

    return model, checkpoint

def save_model(model, optimizer, epoch, test_accuracy, file_path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "test_accuracy": test_accuracy,
    }, file_path)

    print(f"Model saved to {file_path}")


