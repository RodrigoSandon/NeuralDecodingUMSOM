#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 01:55:15 2021

@author: rodrigosandon
"""
import dataloader
from itertools import chain
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
        
    def random_noise(self, number_to_be_generated, rows, columns): #random chosen file as input

        self.loader.generateClassSummaries()
        
        for region in self.loader.regions:
            
            region_mean_matrix = self.loader.summaries[region][0]
            region_std_matrix = self.loader.summaries[region][1]
            
            #mean and std matrix for summarizing each region
            
            for i in range(0, number_to_be_generated):
                generated = np.zeros((rows, columns))
                for row in range(0, rows):
                    for col in range(0, columns):  
                        generated[row][col] = max(-1, min(1, np.random.normal(region_mean_matrix[row][col], region_std_matrix[row][col])))
                #np.savetxt("Data_generated\\{Region}\\{Region}_generated_{index}.csv".format(Region=region, index=i), generated, delimiter=",")
                
    
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
num_files_to_generate = 245
regions = ['visal', 'visam', 'visl', 'visp', 'vispm','visrl' ] #this needs to be kept the same if we want unsupervised
data_path = "/Volumes/Passport/ResearchDataChen/Code/Data/allBrainRegionsnorm/" #needs to be basename for the different regions folder

files = DataAugmentation.combineDataOfFolders(data_path)
mylist = list(files)

num_generated_files = 0

while num_generated_files < num_files_to_generate:
    
    csv_path = DataAugmentation.chooseRandomFile(mylist)
    
    reader = csv.reader(open(csv_path, "rt"), delimiter = ",")
    x = list(reader)
    flat = list(chain.from_iterable(x))
    csv_to_arr = np.array(flat).astype(float) #convert all numbers in list to floats
    
    print(csv_to_arr)
    new_arr = DataAugmentation.random_noise(csv_to_arr)
    
    unflat_arr = DataAugmentation.oneDimToTwoDim(new_arr, 32)
    
    df = pd.DataFrame(unflat_arr, index = None, columns = None)
    df.to_csv(data_path + DataAugmentation.folderPathOfCSV(csv_path) + DataAugmentation.makeNameForNewNoisyCSV(csv_path), header=False, index=False)
    #np.savetxt(data_path + DataAugmentation.folderPathOfCSV(csv_path) + DataAugmentation.makeNameForNewNoisyCSV(csv_path), [unflat_arr], delimiter=',')
    
    num_generated_files += 1


# import dataloader 
# import numpy as np

# class DataGenerator:
    
#     def __init__(self, path, to_search):
#         self.loader = dataloader.DataLoader(path, to_search)
    
    
#     def generateData(self):
#         self.loader.generateClassSummaries()
#         total_number_of_generated = 300
#         rows = 32
#         columns = 32
        
#         for region in self.loader.regions:
#             print(region)
#             region_mean_matrix = self.loader.summaries[region][0]
#             region_std_matrix = self.loader.summaries[region][1]

#             for i in range(0, total_number_of_generated):
#                 generated = np.zeros((rows, columns))
#                 for row in range(0, rows):
#                     for col in range(0, columns):  
#                         generated[row][col] = max(-1, min(1, np.random.normal(region_mean_matrix[row][col], region_std_matrix[row][col])))
#                 np.savetxt("Data_generated\\{Region}\\{Region}_generated_{index}.csv".format(Region=region, index=i), generated, delimiter=",")

                
                
# generator = DataGenerator("Data\\allBrainRegionsraw", ["visal", "visam", "visl", "visp", "vispm", "visrl"])
# generator.generateData()