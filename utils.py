import torch
import torchvision
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import hyper
def load_model(device = 'cpu'):
    model = models.resnet50(pretrained=True)
    last_layer = model.fc.in_features
    model.fc = torch.nn.Linear(last_layer, 90)
    model.load_state_dict(torch.load("model.pth", map_location=torch.device(device)))
    return model
def img_preprocessing(path):
    img = Image.open(path)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize((0.488), (0.2172))
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    imgs = mpimg.imread(path)
    show = plt.imshow(imgs)
    return img

def predict(img, model, classes = hyper.classes):
    model.eval()
    outputs = model(img)
    _, index = torch.max(outputs, 1)
    return classes[index]

