#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 19:32:19 2021

@author: rodrigosandon
"""
import numpy as np
import pickle

f = open("results4.txt", "a+")

class DataSummarizer: 
    
    def __init__(self, path, to_search):
        self.summaries = dict()
        self.class_count = dict()
        self.imp_features = self.load_obj("features_mean") 
        
        return pickle.load(f)
    
    def dataSummary(self, label, features): #to get some simple descriptive stats from each matrix
        mean_matrix = np.zeros([32,32]) #make a mean matrix for all matrices
        std_matrix = np.zeros([32,32]) #make a mean matrix for all matrices
        label_count = 0
        
        for row, col in self.data.iterrows():
            if row["Class"] == label: 
                #if the current name of class we're on matches that of the label that's assigned to it
                label_count += 1
                data = row["Data"]
                
                for important_feature in features:
                    row = int(important_feature[0] / mean_matrix.shape[0]) #normalizing feature value with number of cols?
                    col = important_feature[0] % std_matrix.shape[1] #remainder of dividing feature value with number of rows?
                    mean_matrix[row][col] += (1 - important_feature[1]) * (1 - important_feature[2]) * data[row][col]
                    #this doesn't look like its the mean features so what really is it?
        
        mean_matrix = mean_matrix / label_count #this is where we acc get the mean matrix for the whole dataset
        
        for row, col in self.data.iterrows():
            if row["Class"] == label: 
                #what is the label based on before the code written above, bc it wasn't defined before right?
                data = row["Data"]
                for important_feature in features:
                    row = int(important_feature[0] / mean_matrix.shape[0])
                    col = important_feature[0] % std_matrix.shape[1]
                    std_matrix[row][col] += (((1 - important_feature[1]) * (1 - important_feature[2]) * data[row][col]) - mean_matrix[row][col]) ** 2
        
        std_matrix = np.sqrt(std_matrix / (label_count - 1)) #when we acc get the stdev matrix
        
        return mean_matrix, std_matrix, label_count
    
    def generateDataSummary(self):
        for region in self.regions:
            means, stdevs, class_count = self.dataSummary(region, self.imp_features)
            self.summaries[region] = [means, stdevs]
            self.class_count[region] = class_count