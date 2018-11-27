import torch
from torch import nn
from torch import optim
import torch.nn.functional as f
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import argparse
import model_file
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("-dir", "-d", action="store", default="flowers")
parser.add_argument("-image", action="store", default='paind-project/flowers/test/1/image_06752.jpg')
parser.add_argument("-save_dir","-s_d", action="store", default="checkpoint.pth")
parser.add_argument("-dropout_ratio","-d_r", action="store", default=0.2)
parser.add_argument("-hidden_layer1", "-h_l1", action="store", default=500)
parser.add_argument("-print_every", "-p_e", action="store", default=5)
parser.add_argument("-struct", action="store", default="densenet121", type=str)
parser.add_argument("-epochs", action="store", default=12, type=int)

pa= parser.parse_args()
epochs = pa.epochs
data_dir = pa.dir
path= pa.save_dir
hidden_layer1 = pa.hidden_layer1
print_every = pa.print_every
structure = pa.struct
dropout_ratio = pa.dropout_ratio

trainloaders, validloaders, testloaders, train_datasets = model_file.loading_data(data_dir)
model = models.densenet121(pretrained=True)
model, optimizer, criterion = model_file.model_setup(model, structure, dropout_ratio, hidden_layer1)
model_file.train_network( model, criterion, optimizer, trainloaders,validloaders, epochs, print_every)
model_file.save_checkpoint(path, model, train_datasets)  
print("model has been trained")

