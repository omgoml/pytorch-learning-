import torch 
import os 

DEVICE = {
    "type": torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

ROOT_PATH = "mnist"
MODEL_PATH = os.path.join(ROOT_PATH, "best_model.pth")
BATCH_SIZE = 128
NUM_WORKER = 4 if DEVICE["type"] == "cuda" else 2

EPOCHS = 20
WEIGHT_DECAY = 1e-4

