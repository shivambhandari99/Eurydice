# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Zekun Li  
# version ='1.0'
# ---------------------------------------------------------------------------
""" Spatial AI Assignment """  
# ---------------------------------------------------------------------------

import os
import argparse
import cv2
import numpy as np

import torch 
import torchvision
from torchvision import transforms

import task1_dataset
import task1_model_fcn 
from osgeo import gdal
import sys
torch.set_printoptions(threshold=25000)

#from task1_model_unet import UNet_ResNet

#from matplotlib import pyplot as plt



def main(args):
    test_dir = args.input_dir
    model_save_path = args.model_file
    output_dir = args.output_dir
    is_unet = args.is_unet

    img_path_list = sorted(os.listdir(os.path.join(test_dir, "PS-RGB")))

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if is_unet:
        model = UNet_ResNet(n_class=1)
    else:
        model = task1_model_fcn.fcn_model

    #model.to(device)
    model.load_state_dict(torch.load(model_save_path))
    model.to(device)

    model.eval()

    data_transf = transforms.Compose([transforms.ToTensor(),
                                 transforms.Scale((256, 256))
                                 ])

    for img_path in img_path_list:
        #img = cv2.imread(os.path.join(test_dir,"PS-RGB",img_path))
        ds = gdal.Open(test_dir+'/PS-RGB/'+img_path)
        img = ds.ReadAsArray()
        img = img.astype(np.int16)
        img_0 = torch.squeeze(data_transf(img[0]))
        img_1 = torch.squeeze(data_transf(img[1]))
        img_2 = torch.squeeze(data_transf(img[2]))
        img = torch.stack([img_0,img_1,img_2])
        img = img[None, :]
        img = img.to(device)
        outputs = model(img.float())

        print(type(outputs['out']))
        print(outputs['out'].shape)
        # TODO
        # 1. load test image
        # 2. convert test image into tensor
        # 3. feed input to the model and get output
        # 4. save the output probability map to the output folder

        # Hint: need to prepare the image into 4-d tensor for model input
        # Hint: need to resize the image to the approporate size
        # Hint: the direct outputs of the model are logits, you need to convert them to probability
        # Hint: output range need to be rescaled to [0,255] to save with cv2.imwrite()


        
        cv2.imwrite(out_path, out_img)
        
        
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, default='../../data/task1_data/task1_test/',
                        help='the testing folder that contain the geotiff files.')
    parser.add_argument('--model_file', type=str, default='fcn_weights/task1.pth',
                        help='path to the model file generated after training process.')
    parser.add_argument('--output_dir', type=str, default='task1_out',
                        help='the output folder to save all the output probability maps')
    parser.add_argument('--is_unet', default=False, action='store_true',
                        help='use this param if you implemented UNet.')
    args = parser.parse_args()
    print(args)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    main(args)
