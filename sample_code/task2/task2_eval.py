# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Zekun Li  
# version ='1.0'
# ---------------------------------------------------------------------------
""" Spatial AI Assignment"""  
# ---------------------------------------------------------------------------

import os 
import argparse
from PIL import Image
import cv2
import numpy as np

import torch
import torchvision
import transforms as T
from engine import  evaluate
import utils
from task2_dataset import PlaneDataset

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def main(args):
    data_dir = args.data_dir
    model_save_path = args.model_file

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset_test = PlaneDataset(img_dir = os.path.join(data_dir, 'test_data'), 
                                annot_file_path = os.path.join(data_dir, 'test_list.csv'),
                                transforms = get_transform(train=False))
    

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    model.to(device)
    model.load_state_dict(torch.load(model_save_path))

    model.eval()

    evaluate(model, data_loader_test, device=device)




if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../../data/task2_data/',
                        help='the evaluation data folder')
    parser.add_argument('--model_file', type=str, default='fasterrcnn_weights/task2.pth',
                        help='the model (weights) generated during the training process.')
    args = parser.parse_args()

    main(args)