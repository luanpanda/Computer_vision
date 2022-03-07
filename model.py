import utils
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
img_path = str(input('Image path: '))
img = utils.img_preprocessing(img_path)
model = utils.load_model(device)
print(utils.predict(img, model))
