import torch
from torch import nn
from torch import optim
import torch.nn.functional as f
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import argparse
import matplotlib.pyplot as plt
import json
import model_file

parser = argparse.ArgumentParser()
parser.add_argument("-path", "-p", action="store", default= "checkpoints.pth")
parser.add_argument("-data_dir", action="store", default= "flowers")
parser.add_argument("-img", action="store", default="flowers/test/10/image_07090.jpg")
parser.add_argument("-topk", action="store", default=5)
#parser.add_argument("-structure", action="store", default="densenet121")
#parser.add_argument("-hidden_layer1", action="store", default=500)
                    
pa = parser.parse_args()
path = pa.path
data_dir = pa.data_dir
img = pa.img
number_of_outputs = pa.topk
#structure = pa.structure
#hidden_layer1 = pa.hidden_layer1                    

trainloaders, validloaders, testloaders, train_datasets = model_file.loading_data(data_dir)
model = model_file.load_model(path)
model_file.process_image(img)


with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
probabilities = model_file.predict(img,model, number_of_outputs)     

labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])


i=0
while i < number_of_outputs:
    print("{} with a probability of {}".format(labels[i], probability[i]))
    i += 1

