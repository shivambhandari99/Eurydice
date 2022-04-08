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

from torchvision import transforms


def main(args):

    model_save_path = args.model_file 
    output_path = args.output_file
    input_dir = args.input_dir


    img_path_list = sorted(os.listdir(os.path.join(input_dir)))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    model.to(device)
    model.load_state_dict(torch.load(model_save_path))

    model.eval()

    with open(output_path, 'w') as out_f:

        line_idx = 0
        for img_path in img_path_list:
            img_abs_path = input_dir+'/'+ img_path
            img = Image.open(img_abs_path)
            print(type(img))
            convert_tensor = transforms.ToTensor()
            img = convert_tensor(img)
            img = img.to(device)
            print(type(img))
            #img = torch.LongTensor(img)
            output = model(img[None,:])
            print(output)
            print(output[0]['boxes'])
            for box in output[0]['boxes']:
                out_f.write(img_path+','+str(box)+'\n')

            # 1. read image from img_path and conver to tensor
            # 2. feed input tensor to model and get bbox with output[0]['bboxes']
            # 3. write to csv file
            # Hint: You can plot the output box on the image to visualize/verify the result

        
            



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../../data/Plane/test_data/',
                        help='the input directory that contains test_data folder')
    parser.add_argument('--model_file', type=str, default='fasterrcnn_weights/ep_39.pth',
                        help='the model (weights) generated during the training process.')
    parser.add_argument('--output_file', type=str, default='task2_out.csv',
                        help='the output csv file that has the same format as train_list.csv')
    args = parser.parse_args()

    main(args)
