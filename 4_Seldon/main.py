import torch
import urllib.request
from PIL import Image
from torchvision import transforms
import cv2
import numpy as np

class main(object):
    def __init__(self):
        print("Initialzing the class")
        print("Loading model")

    def predict(self, X, feature_names):
        print("In the prediction..")
        model = torch.load("model.pkl",map_location=torch.device('cpu'))
        urllib.request.urlretrieve(X[0],"temp.jpg")
        img = Image.open("temp.jpg")
        print("Image Downloaded")
        img = cv2.cvtColor(np.float32(img), cv2.COLOR_BGRA2BGR)
        img = transforms.ToTensor()(img).unsqueeze_(0)
        output = model(img)
        label = X[1]
        pred = output.argmax(dim=1, keepdim=True)
        res = ['True Positive'] #Default
        print("Going in for Call")
        if pred.item() == 1 and label == 1:
            res =  ["True Positive"]
        elif pred.item() == 0 and label == 1:
            rest = ["False Negative"]
        elif pred.item() == 1 and label == 0:
            res = ["False Positive"]
        elif pred.item() == 0 and label == 0:
            res = ["True Negative"]
        return res
