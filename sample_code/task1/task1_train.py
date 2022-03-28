# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Zekun Li  
# version ='1.0'
# ---------------------------------------------------------------------------
""" Spatial AI Assignment"""  
# ---------------------------------------------------------------------------

import argparse
import os 

import torch
import torch.nn as nn
import torch.optim as optim 

from torch.utils.data import DataLoader, Dataset
import torch.utils.data as utils
from torchvision import transforms
import torch.nn.functional as F

from task1_dataset import RoadDataset
import task1_model_fcn 
#from task1_model_unet import UNet_ResNet



def main(args):

    train_dir = args.input_dir
    model_save_dir = args.model_dir
    epochs = args.epochs
    is_unet = args.is_unet

    train_prefix = 'SN3_roads_train_AOI_4_Shanghai_PS-RGB_'
    

    data_transf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Scale((256, 256))
                                 ])
    road_dataset_train = RoadDataset(root = train_dir, transforms = data_transf, img_name_prefix = train_prefix)
    train_loader = DataLoader(dataset = road_dataset_train, batch_size=4)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
            print(type(outputs))
            print(type(labels))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # TODO
            # 1. get data from train_loader          
            # 2. set all tensor gradients to be 0.
            # 3. feed the input to model and get prediction
            # 4. calculate the loss of ground-truth (GT) and prediction
            # 5. back propagation

            if(ii%100==99):
                print('Epoch: {} - Loss: {:.6f}'.format(epoch + 1, loss.item()))
                #torch.save(model.state_dict(), os.path.join(model_save_dir, 'ep_' + str(epoch) +'.pth'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../../data/task1_data/task1_train/',
                        help='the training folder that contain the geotiff and road vector data.')
    parser.add_argument('--epochs', type=int, default = 45,
                        help='the number of epochs to train the model.')
    parser.add_argument('--model_dir', type=str, default='fcn_weights/',
                        help='the output file path to save the trained model weights.')
    parser.add_argument('--is_unet', default=False, action='store_true',
                        help='use this param if you implemented UNet.')
    args = parser.parse_args()
    print(args)


    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)

    main(args)
