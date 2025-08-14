import os
from torchvision import transforms
from config.configer import *
from utils.utils import ModelLoader

class InferenceModel:
    def __init__(self) -> None:
        self.model = None 
        self.transform = transforms.Compose([ 
            transforms.ToTensor(),
            transforms.Normalize((0.1307,),(0.3081,))
        ])

    def load_trained_model(self):
        if not os.path.exists(MODEL_PATH):
            print(f"File path {MODEL_PATH} not exit")
            return "Fail"
        else:
            try:
                loaded_model, _ = ModelLoader()

                if loaded_model is not None:
                    self.model = loaded_model
                    self.model.eval()
                    print("Loaded model successfully")
                    return "Success"

                else:
                    print("There are no model which are loaded")
                    print("please retrain")
                    return "Fail"

            except Exception as e:
                print(f"Error: {e}")
                return "Fail"
    
    def predict_single_image(self, image_tensor:torch.Tensor):

        if self.model is None:
            print("Model not loaded")
            return None, None, None 
        
        if len(image_tensor.size()) == 3:
            image_tensor = image_tensor.unsqueeze(0)

        image_tensor = image_tensor.to(DEVICE["type"])
        
        with torch.no_grad():
            output = self.model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confience = probabilities.max().item()

        return prediction, confience, probabilities

    def predict_from_file(self, image_path):
        from PIL import Image 

        if self.model is None:
            print("Model not loaded")
            return None, None, None 

        image = Image.open(image_path).convert("L")
        image = image.resize((28,28))
        image_tensor =torch.Tensor(self.transform(image))

        return self.predict_single_image(image_tensor)
