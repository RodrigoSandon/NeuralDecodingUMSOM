#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 01:57:31 2021

@author: rodrigosandon
"""

import numpy as np
import pandas as pd
import os
import csv
import pickle


class DataLoader:

    def __init__(self, path, to_search):
        self.regions = to_search
       # print("New Dataloader")
        self.path = path
        self.data = self.convertToDataFrame(self.path)
        #self.data, self.test = self.cross_validation_split(self.data, 0.75)
        self.summaries = dict()
        self.class_count = dict()
        self.imp_features = self.all_features() 
        
        #self.load_obj("features_mean") 
        
    
    def all_features(self):
        features = []
        for i in range(0, 1024):
            features.append((i, 0, 0))
        return features
            
    def load_obj(self, name):
        with open('obj/' + name + '.pkl', 'rb') as f:
            return pickle.load(f)
        
    def cross_validation_split(self, data, train_size):
        train = data.sample(frac=train_size)
        test = data.drop(train.index).sample(frac=1.0)
        
        return train, test
        
    def convertToDataFrame(self, path):
        df = pd.DataFrame(columns=("Class", "Data"))
        for subdir, dirs, files in os.walk(path):
        
        
            for file in files:
                reader = csv.reader(open(os.path.join(subdir, file), "rt"), delimiter=",")
                x = list(reader)
                result = np.array(x).astype("float")
                
                row = pd.DataFrame({"Class" : os.path.basename(os.path.normpath(subdir)), "Data" : [result] })
            
                df = df.append(row, ignore_index=True)
        
        return df
    
    
    def summarizeData(self, label, important_features):
        mean_matrix = np.zeros([32, 32])
        std_matrix = np.zeros([32, 32])
        label_count = 0
        
        for index, row in self.data.iterrows():   
            #if self.shouldAnalyze(row["Class"], label):

            if row["Class"] == label:
                label_count += 1
                data = row["Data"]
                
                for feature in important_features:
                    row = int(feature[0] / mean_matrix.shape[0])
                    col = feature[0] % std_matrix.shape[1]
                    mean_matrix[row][col] += (1 - feature[1]) * (1 - feature[2]) * data[row][col]
                
                        
        mean_matrix = mean_matrix / label_count
        for index, row in self.data.iterrows():
             #if self.shouldAnalyze(row["Class"], label):
            if row["Class"] == label:
           
                data = row["Data"]
                for feature in important_features:
                    row = int(feature[0] / std_matrix.shape[0])
                    col = feature[0] % std_matrix.shape[1]
                    std_matrix[row][col] += (((1 - feature[1]) * (1 - feature[2]) * data[row][col]) - mean_matrix[row][col]) ** 2
        std_matrix = std_matrix / (label_count - 1)
        std_matrix = np.sqrt(std_matrix)
        
        return mean_matrix, std_matrix, label_count
    
    def generateClassSummaries(self):
        for region in self.regions:
            means, stds, class_count = self.summarizeData(region, self.imp_features)
            self.summaries[region] = [means, stds]
            self.class_count[region] = class_count
        
                    
   
loader = DataLoader("Data\\allBrainRegionsraw", ["visal", "visam", "visl", "visp", "vispm", "visrl"])
loader.generateClassSummaries()
print(loader.summaries)



# import numpy as np
# import pandas as pd
# import os
# import csv
# import pickle

# class DataLoader:
    
#     def __init__(self, path):
#         self.path = path
#         self.data = self.csvToDataframe(self.path)
        
    
#     def load_object(self, name):
#         with open('obj/' + name + '.pkl', 'rb') as f:
#             return pickle.load(f)
    
#     def csvToDataframe(self, path):
#         df = pd.DataFrame(columns = ["Class", "Data"])
#         # we cans still label the csv but we don't have to follow through with it
        
#         for subdir, folders in os.walk(path): #path is the general input to the whole class
#             for csv_files in folders:
#                 reader = csv.reader(open(os.path.join(subdir,csv_files), "rt"), delimiter = ",")
#                 x = list(reader) #convert what was read into an iterable list
#                 result = np.array(x).astype("float") #convert all numbers in list to floats
                
#                 row = pd.DataFrame({ #adding on the values directly to the columns
#                     "Class": os.path.basename(os.path.normpath(subdir)), 
#                     "Data": [result] #result is the list of floats
#                     })
#                 df = df.append(row, ignore_index = True) 
#                 #we are going to have a large dataframe with
#                 #all of the matrices represented as a flattened list
        
#         return df
                    
            
                
#dataloader = DataLoader("/Volumes/Passport/ResearchDataChen/Code/Data")
#dataloader.generateClassSummaries()
#print(dataloader.summaries)
#print(dataloader.class_count)
