import torch
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

def CovidPredictor(imgPath,label):
    model = torch.load("model.pkl",map_location=torch.device('cpu'))
    img = Image.open(imgPath)
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGRA2BGR)
    img = transforms.ToTensor()(img).unsqueeze_(0)
    output = model(img)
    pred = output.argmax(dim=1, keepdim=True)
    if pred.item() == 1 and label == 1:
        return "True Positive"
    elif pred.item() == 0 and label == 1:
        return "False Negative"
    elif pred.item() == 1 and label == 0:
        return "False Positive"
    elif pred.item() == 0 and label == 0:
        return "True Negative"

path = "covid.png"
label = 1 # 1 for Covid Expected and 0 for Non-Covid Expected
response = CovidPredictor(path,label)
print(response)
