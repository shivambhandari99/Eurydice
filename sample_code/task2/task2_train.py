# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Zekun Li  
# version ='1.0'
# ---------------------------------------------------------------------------
""" Spatial AI Assignment"""  
# ---------------------------------------------------------------------------
#/Users/shivambhandari/opt/anaconda3/envs/assignment4 
import argparse
import os 

import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate, evaluate_loss
from task2_dataset import PlaneDataset
import utils

import transforms as T


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)



def main(mode='train'):

    num_epochs = args.epochs
    input_dir = args.input_dir
    model_save_dir = args.model_dir


    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


    dataset = PlaneDataset(img_dir = os.path.join(input_dir, 'train_data'), 
                            annot_file_path = input_dir + 'train_list.csv',
                            transforms = get_transform(train=False))
    train_set, val_set = torch.utils.data.random_split(dataset, [70, 10])

    data_loader_train = torch.utils.data.DataLoader(
        train_set, batch_size=2, shuffle=True, num_workers=2,
        collate_fn=utils.collate_fn)
    data_loader_val = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    model.to(device)
    
    
        
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.0001)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader_train, device, epoch, 1, scaler=None)
        torch.save(model.state_dict(), os.path.join(model_save_dir, 'ep_run4' + str(epoch) +'.pth'))
        print("--------------------")
        evaluate_loss(model, data_loader_val, device=device)
        evaluate(model, data_loader_val, device=device)


        # TODO:
        # train for each epoch and update learning rate
        # save model for each epoch
        # Hint: you can use train_one_epoch() implemented for you


        
    
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../../data/Plane/',
                        help=' the input directory that contains train_data folder and train_list.csv')
    parser.add_argument('--model_dir', type=str, default='fasterrcnn_weights/',
                        help='the output file path to save the trained model weights.')
    parser.add_argument('--epochs', type=int, default = 40,
                        help='the number of epochs to train the model.')
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        os.makedirs(args.model_dir)


    main()
