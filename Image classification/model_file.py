import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
from matplotlib.ticker import FormatStrFormatter
import argparse

import matplotlib.pyplot as plt
def loading_data( data_dir='flowers'):
#data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    train_transforms = transforms.Compose([transforms.RandomRotation(30), transforms.CenterCrop(224),transforms.RandomHorizontalFlip(), transforms.ToTensor(),transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    valid_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],        [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform =valid_transforms)
    test_dataset = datasets.ImageFolder(test_dir, transform =test_transforms)

    trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size=64, shuffle=True)
    validloaders = torch.utils.data.DataLoader(valid_dataset, batch_size=32, shuffle=True)
    testloaders = torch.utils.data.DataLoader(test_dataset, batch_size=20, shuffle=True)

    return trainloaders, validloaders, testloaders, train_datasets



# TODO: Build and train your network
def model_setup(model, structure = 'densenet121', dropout_ratio=0.2, hidden_layer1=500):
#load a pretrained network
    if structure == 'densenet121':
        model = models.densenet121(pretrained=True)


    #Freeze the parameters so we do not change them
    for param in model.parameters():
        param.requires_grad = True


    from collections import OrderedDict
    Classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(1024,hidden_layer1)), ('ReLU1', nn.ReLU()), ('dropout',nn.Dropout(dropout_ratio)), ('fc2', nn.Linear(hidden_layer1,200)), ('ReLU2', nn.ReLU()), ('fc3', nn.Linear(200,102)), ('output', nn.LogSoftmax(dim=1))]))
    model.classifier = Classifier
    model.cuda()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    return model, optimizer, criterion

def train_network( model, criterion, optimizer, loaders, loader, epochs=12, print_every=5):
    steps = 0
    loss_show=[]

# change to cuda
    model.to('cuda')

    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(loaders):
            steps += 1
            model.train()

            inputs,labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                vlost = 0
                accuracy=0


                for ii, (inputs2,labels2) in enumerate(loader):
                    optimizer.zero_grad()
                    model.eval()
                    inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                    model.to('cuda:0')
                    with torch.no_grad():
                        outputs = model.forward(inputs2)
                        vlost = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                vlost = vlost / len(loader)
                accuracy = accuracy /len(loader)



                print("Epoch: {}/{}... ".format(e+1, epochs),
                  "Loss: {:.4f}".format(running_loss/print_every),
                  "Validation Lost {:.4f}".format(vlost),
                   "Accuracy: {:.4f}".format(accuracy))


                running_loss = 0



def save_checkpoint(path, model, train_datasets):
    checkpoint = torch.save(model.state_dict(), path)
    state_dict = torch.load(path)
    model.class_to_idx = train_datasets.class_to_idx
    model.cpu
    torch.save({'structure' :'densenet121','model':'vgg16', 'hidden_layers':500, 'state_dict': state_dict, 'class_to_idx': model.class_to_idx}, 'checkpoints.pth')

# TODO: Write a function that loads a checkpoint and rebuilds the model
def load_model(path):
    checkpoint = torch.load(path)
    struct = checkpoint["structure"]
    hidden_layer1 = checkpoint["hidden_layers"]
    model = models.densenet121(pretrained=True)
    model,_,_  =  model_setup(model, struct , 0.2,hidden_layer1)
    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    return model

def process_image(image):
    img_pil = Image.open(image)

    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    img_tensor = adjustments(img_pil)

    return img_tensor

#model.class_to_idx =train_datasets.class_to_idx

#ctx = model.class_to_idx


def predict(image_path, model, topk=5):
    #if torch.cuda.is_available(): #and power==='gpu':
        #model.to('cuda:0')

    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()

    #if power == 'gpu':
    with torch.no_grad():
        output = model.forward(img_torch.cuda())
    #else:
    #with torch.no_grad():
        #output=model.forward(img_torch)

    probability = F.softmax(output.data,dim=1)

    return probability.topk(topk)
