# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Zekun Li  
# version ='1.0'
# ---------------------------------------------------------------------------
""" Spatial AI Assignment"""  
# ---------------------------------------------------------------------------

import argparse
import os 

import torch, gc
import torch.nn as nn
import torch.optim as optim 

from torch.utils.data import DataLoader, Dataset
import torch.utils.data as utils
from torchvision import transforms
import torch.nn.functional as F

from task1_dataset import RoadDataset
import task1_model_fcn 
from task1_model_unet import UNet_ResNet



def main(args):
    torch.cuda.empty_cache()

    train_dir = args.input_dir
    model_save_dir = args.model_dir
    epochs = args.epochs
    is_unet = args.is_unet

    train_prefix = 'SN3_roads_train_AOI_4_Shanghai_PS-RGB_'
    

    data_transf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Scale((256, 256))
                                 ])
    road_dataset_train = RoadDataset(root = train_dir, transforms = data_transf, img_name_prefix = train_prefix)
    train_set, val_set = torch.utils.data.random_split(road_dataset_train, [900, 128])
    train_loader = DataLoader(dataset = train_set, batch_size=4)
    #val_loader = DataLoader(dataset = val_set, batch_size=1)

    #train_size = int(0.8 * len(full_dataset))
    #test_size = len(full_dataset) - train_size
    #train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    if is_unet:
        model = UNet_ResNet(n_class=2)
    else:
        model = task1_model_fcn.fcn_model

    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), weight_decay=1e-3, lr = 0.01, momentum=0.9)


    for epoch in range(epochs):      
        model.train()
        running_loss = 0.0         
        for ii, (data, target) in enumerate(train_loader):
            inputs, labels = data.to(device), target.to(device)
            inputs = inputs.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            #print(outputs.shape)
            #print(type(outputs))
            loss = criterion(outputs, labels)
            print(loss)
            print(type(loss))
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            # TODO
            # 1. get data from train_loader          
            # 2. set all tensor gradients to be 0.
            # 3. feed the input to model and get prediction
            # 4. calculate the loss of ground-truth (GT) and prediction
            # 5. back propagation

        print('Epoch: {} - Loss: {:.6f}'.format(epoch + 1, running_loss/len(train_set)))
        running_loss = 0.0
        torch.save(model.state_dict(), os.path.join(model_save_dir, 'ep_' + str(epoch) +'.pth'))
"""
        gc.collect()
        torch.cuda.empty_cache()

        val_loss = []
        for ii, (data, target) in enumerate(val_loader):
            inputs, labels = data.to(device), target.to(device)
            inputs = inputs.float()
            outputs = model(inputs)
            loss = criterion(outputs['out'], labels)
            val_loss.append(loss)
        print("Validation loss is: ")
        print(val_loss)
        print(sum(val_loss)/len(val_set))
        print("------------------------")
"""


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../../data/task1_data/task1_train/',
                        help='the training folder that contain the geotiff and road vector data.')
    parser.add_argument('--epochs', type=int, default = 45,
                        help='the number of epochs to train the model.')
    parser.add_argument('--model_dir', type=str, default='fcn_weights/',
                        help='the output file path to save the trained model weights.')
    parser.add_argument('--is_unet', default=True, action='store_true',
                        help='use this param if you implemented UNet.')
    args = parser.parse_args()
    print(args)


    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)

    main(args)
