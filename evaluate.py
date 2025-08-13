import torch 
from tqdm import tqdm

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0 
    correct = 0.0 
    total = 0 
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        
        for data, target in pbar:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True),
            output = model(data)
            test_loss += criterion(target,output)
            prediction = output.argmax(dim=1,keepdim=True)
            correct += prediction.eq(target.view_as(prediction)).sum().item()
            total += target.size(0)

            pbar.set_postfix({"accuracy": f"{100 * correct / total:.2f}%"})

    test_loss /= len(test_loader)
    accuracy = 100 * correct / total

    return test_loss, accuracy
