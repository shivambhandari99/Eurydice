# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Zekun Li  
# version ='1.0'
# ---------------------------------------------------------------------------
""" Spatial AI Assignment"""  
# ---------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cv2
from PIL import Image
import glob


import torch
import geojson
from osgeo import gdal

import geojson


class RoadDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms, img_name_prefix = 'SN3_roads_train_AOI_2_Vegas_PS-RGB_'):
        self.root = root
        self.transforms = transforms

        self.jsons = list(sorted(os.listdir(os.path.join(root, "geojson_roads"))))
        
        self.imgs = [ img_name_prefix + json_name.split('_')[-1].split('.geojson')[0]+'.tif'
                     for json_name in self.jsons]

    
        
    def __getitem__(self, idx):


        # TODO
        # 1. Get image and vector data path according to the idx
        # 2. Read image and calculate the mask 
        # 3. Apply transformations
        # 4. Return image and mask 



        return img, mask
    
    def caclulate_mask(self,tif_path, geojson_path, line_thickness = 30, color = (1,1,1)):
        
        
        with open(geojson_path, 'r') as f:
            vector_data = geojson.load(f)

        ds = gdal.Open(tif_path)
        geoTransform = ds.GetGeoTransform()
        
        minx = geoTransform[0]
        maxy = geoTransform[3]
        maxx = minx + geoTransform[1] * ds.RasterXSize
        miny = maxy + geoTransform[5] * ds.RasterYSize
        
        mask_img = np.zeros((ds.RasterXSize, ds.RasterYSize, 3)).astype(np.float32)

        
        
        # TODO
        # 1. Parse vector_data dictionary and process each feature
        # 2. Get the line geometry coordinates and convert from lat,lng to pixel coord system
        # 3. Plot the line on mask_img with cv2.polylines() function.
        # 4. Return the mask image
        # Hint: the number of channels for mask_img should be 2

        
                
        return mask_img

    def __len__(self):
        return len(self.imgs)