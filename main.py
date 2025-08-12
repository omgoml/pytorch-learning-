import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader
from torchvision import transforms, datasets 
import matplotlib.pyplot as plt

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_data = datasets.MNIST(
    root="mnist",
    train=True,
    transform=transform,
    download=True,
)

test_data = datasets.MNIST(
    root="mnist",
    train=False,
    transform=transform,
    download=True,
)

train_loader = DataLoader(train_data, batch_size=64,shuffle=True)
test_loader = DataLoader(test_data,batch_size=64,shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class NeuralNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            #first convolutional block 
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            #second convolutional block
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),

            #flatten data for the fully connected network 
            nn.Flatten(),

            #network 
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128,10),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, input_data):
        return self.model(input_data)

model = NeuralNetwork().to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for image, label in train_loader:
        image, label = image.to(device), label.to(device)

        output = model.forward(image)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    return avg_loss

def test(model, test_loader, device):
    model.eval()
    correct = 0.0 
    total = 0.0 

    with torch.no_grad():
        for image, label in test_loader:
            image, label = image.to(device), label.to(device)
            outputs = model.forward(image)

            _, prediction = torch.max(outputs.data,1)

            total += label.size(0)
            correct += (prediction == label).sum().item()

    accuracy = 100 * correct / total
    return accuracy

epochs = 10 
train_losses = []
test_accuracies = []

for epoch in range(epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)

    test_accuracy = test(model, test_loader, device)
    test_accuracies.append(test_accuracy)

    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%") 

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(test_accuracies)
plt.title('Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.show()

