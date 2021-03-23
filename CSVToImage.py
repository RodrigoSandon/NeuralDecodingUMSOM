#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 19:16:10 2021

@author: rodrigosandon
"""

import csv
from PIL import Image
from matplotlib import pyplot as plt
import os
import numpy as np

newdir = "/Volumes/Passport/ResearchDataChen/Code/Data/testingNoiseData/visp/"
path = "/Volumes/Passport/ResearchDataChen/Code/Data/allBrainRegionsnorm/visp/"

class CSVToImage:
    
    def __init__(self, folder_path):
        self.folder_path = folder_path
    
    def csvToImage(csv_path):
        
        results1 = []
        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile, quoting = csv.QUOTE_NONNUMERIC)
            for row in reader:
                results2 = []
                for col in row:
                    results2.append(col)
                results1.append(results2)
        arr = np.array(results1)
        #unflat_arr = CSVToImage.oneDimToTwoDim(arr, 32)
        #im = Image.fromarray(np.float64(arr), 'L')
        
        return arr
            
    def makeNameForNoisyIMG(csv_path):
        #/Volumes/Passport/ResearchDataChen/Code/Data/allBrainRegionsnorm/visrl/553568031_visrl_normalized_corrmap.csv
        pieces = csv_path.split("/")
        piece = pieces[8]
        pieces2 = piece.split("corr")
        piece2 = pieces2[0]
        
        new_name = piece2 + "im.png"
        
        return new_name
    
    def oneDimToTwoDim(lst, num_rows):
        return [lst[i:i+num_rows] for i in range(0, len(lst), num_rows)]
                    

# for i in os.listdir(path):
#     if not i.startswith("._"):
#         print(i)
#         im = CSVToImage.csvToImage(path + i)
#         #im.save(newdir + CSVToImage.makeNameForNoisyIMG(path + i))
        
array = CSVToImage.csvToImage("/Volumes/Passport/ResearchDataChen/Code/Data/allBrainRegionsnorm/visp/noisy_712178483_visp_normalized_corrmap.csv")
im = plt.imshow(array, interpolation = 'nearest')
plt.show(im)

#im1.save(newdir + CSVToImage.makeNameForNoisyIMG("/Volumes/Passport/ResearchDataChen/Code/Data/allBrainRegionsnorm/visp/noisy_501021421_visp_normalized_corrmap.csv"))
