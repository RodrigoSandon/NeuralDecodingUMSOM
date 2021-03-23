#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 00:47:36 2021

@author: rodrigosandon
"""

from itertools import chain
import dataloader
import pandas as pd
import numpy as np
import os, random
import csv


class DataAugmentation:
    
    def __init__(self, data_path, amount_to_make, regions):
        self.data_path = data_path
        self.amount_to_make = amount_to_make
        self.regions = regions
        
    
    def chooseRandomFile(list_of_csv_dirs):
        rand_csvfile = random.choice(list_of_csv_dirs)
        
        return rand_csvfile
        
    def random_noise(array): #random chosen file as input
        mu, sigma = 0, 0.1
        noise = np.random.normal(mu,sigma,array.shape) #a matrix
        
        noisy_arr = array + noise
        
        return noisy_arr
    
    def findRegionOfCSV(csv_path):
         #/Volumes/Passport/ResearchDataChen/Code/Data/allBrainRegionsnorm/visrl/553568031_visrl_normalized_corrmap.csv
        pieces = csv_path.split("/")
        piece = pieces[8]
        pieces2 = piece.split("_")
        region = pieces2[1]
        return region
    
    def random_noise2(array, region):
        rows = 32
        columns = 32
        mu, sigma = dataloader.summaries[region][0], dataloader.summaries[region][1]
        
        noise = np.zeros((rows, columns))
        for row in range(0, rows):
            for col in range(0, columns):
                noise[row][col] = max(-1, min(1, np.random.normal(mu, sigma)))
                
        return noise
        
    def combineDataOfFolders(base_path):
        list_of_csv_paths = []
        for folders in os.listdir(base_path):
            for csvs in os.listdir(base_path+ folders):
                if not csvs.startswith("._"):
                    list_of_csv_paths.append(base_path+ folders + "/" + csvs)
                
        return list_of_csv_paths
    
    def makeNameForNewNoisyCSV(csv_path):
        #/Volumes/Passport/ResearchDataChen/Code/Data/allBrainRegionsnorm/visrl/553568031_visrl_normalized_corrmap.csv
        pieces = csv_path.split("/")
        piece = pieces[8]
        
        new_name = "noisy_"+ piece
        
        return new_name
    
    def folderPathOfCSV(csv_path):
        pieces = csv_path.split("/")
        
        folder_path = pieces[7] + "/"
        
        return folder_path
    
    def oneDimToTwoDim(lst, num_rows):
        return [lst[i:i+num_rows] for i in range(0, len(lst), num_rows)]
    
            
            
        
#nvm its not a problem, this is just called additive noise
#if want to make 100 for each, 100 * 6
num_files_to_generate = 2000
regions = ['visal', 'visam', 'visl', 'visp', 'vispm','visrl' ] #this needs to be kept the same if we want unsupervised
#regions = ['visam']
data_path = "/Volumes/Passport/ResearchDataChen/Code/Data/mix_noisy_method1_copy/" #needs to be basename for the different regions folder

files = DataAugmentation.combineDataOfFolders(data_path)
mylist = list(files)

num_generated_files = 0

while num_generated_files < num_files_to_generate:
    
    csv_path = DataAugmentation.chooseRandomFile(mylist)
    region_of_csv = DataAugmentation.findRegionOfCSV(csv_path)
    
    reader = csv.reader(open(csv_path, "rt"), delimiter = ",")
    x = list(reader)
    flat = list(chain.from_iterable(x))
    csv_to_arr = np.array(flat).astype(float) #convert all numbers in list to floats
    
    print(num_generated_files)
    #print(csv_to_arr)
    #new_arr = DataAugmentation.random_noise2(csv_to_arr, region_of_csv)
    new_arr = DataAugmentation.random_noise(csv_to_arr)
    
    unflat_arr = DataAugmentation.oneDimToTwoDim(new_arr, 32)
    
    df = pd.DataFrame(unflat_arr, index = None, columns = None)
    df.to_csv(data_path + DataAugmentation.folderPathOfCSV(csv_path) + DataAugmentation.makeNameForNewNoisyCSV(csv_path), header=False, index=False)
    #np.savetxt(data_path + DataAugmentation.folderPathOfCSV(csv_path) + DataAugmentation.makeNameForNewNoisyCSV(csv_path), [unflat_arr], delimiter=',')
    
    num_generated_files += 1
    
num_generated_files = 0
    
            
            
            
        
        
    
       
        
