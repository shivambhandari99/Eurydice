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

import affine

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
        tif_path = self.root + '/PS-RGB/' + self.imgs[idx]
        geojson_path = self.root + '/geojson_roads/' + self.jsons[idx]
        ds = gdal.Open(tif_path)
        img = ds.ReadAsArray()
        img = img.astype(np.int16)
        img_0 = self.transforms(img[0])
        img_1 = self.transforms(img[1])
        img_2 = self.transforms(img[2])
        img = torch.stack(img_0,img_1,img_2)
        mask = self.caclulate_mask(tif_path, geojson_path, line_thickness = 30, color = (1,1,1))
        complementary_mask = mask
        where_0 = np.where(mask == 0)
        where_1 = np.where(mask == 1)

        complementary_mask[where_0] = 1
        complementary_mask[where_1] = 0

        complementary_mask = self.transforms(complementary_mask)
        mask = self.transforms(mask)
        mask = torch.stack([mask,complementary_mask])
        return img, mask

    @staticmethod
    def retrieve_pixel_value(geo_coord, data_source):
        """Return floating-point value that corresponds to given point."""
        x, y = geo_coord[0], geo_coord[1]
        forward_transform =  \
            affine.Affine.from_gdal(*data_source.GetGeoTransform())
        reverse_transform = ~forward_transform
        px, py = reverse_transform * (x, y)
        px, py = int(px + 0.5), int(py + 0.5)
        pixel_coord = px, py

        data_array = np.array(data_source.GetRasterBand(1).ReadAsArray())
        return [pixel_coord[0],pixel_coord[1]]

    
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

        

        isClosed = False

        # Line thickness of 2 px
        thickness = line_thickness


        for feature in vector_data['features']:
            if(feature['geometry']['type'] == "LineString"):
                master_line = []
                for coordinate in feature['geometry']['coordinates']:
                    try:
                        master_line.append(RoadDataset.retrieve_pixel_value(tuple(coordinate),ds))
                    except Exception as e:
                        print("Error caused by:")
                        print(e)
                        print(coordinate,master_line)
                        continue
                master_line = np.asarray(master_line,dtype=np.float32)
                master_line = master_line.reshape((-1, 1, 2))
                mask_img = cv2.polylines(mask_img, np.int32([master_line]), isClosed, color, thickness)
            else:
                for coordinate in feature['geometry']['coordinates']:
                    master_line = []
                    for inner_coordinate in coordinate:
                        try:
                            master_line.append(RoadDataset.retrieve_pixel_value(tuple(inner_coordinate),ds))
                        except Exception as e:
                            print("Error caused by:")
                            print(e)
                            print(inner_coordinate,master_line)
                            continue
                    master_line = np.asarray(master_line,dtype=np.float32)
                    master_line = master_line.reshape((-1, 1, 2))
                    mask_img = cv2.polylines(mask_img, np.int32([master_line]), isClosed, color, thickness)

        
        # TODO
        # 1. Parse vector_data dictionary and process each feature
        # 2. Get the line geometry coordinates and convert from lat,lng to pixel coord system
        # 3. Plot the line on mask_img with cv2.polylines() function.
        # 4. Return the mask image
        # Hint: the number of channels for mask_img should be 2

        
                
        return mask_img

    def __len__(self):
        return len(self.imgs)