#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 22:05:03 2021

@author: rodrigosandon
"""
import csv
from PIL import Image
import os, random
import numpy as np
import pandas as pd
from scipy import ndarray
import skimage as sk
import shutil

parent_dir = "/Volumes/Passport/ResearchDataChen/Code/Data/mix_noisy_method1_copy/"

class DataUtilities:
    
    def __init__(self, file_path):
        self.file_path = self
        
    def deleteFilesStartingWith(meta_folder_path, string):
        for folder in os.listdir(meta_folder_path):
            for csvs in os.listdir(meta_folder_path + folder):
                if csvs.startswith(string):
                    os.remove(meta_folder_path + folder + "/" + csvs)
                    
    def deleteFilesStartingWithDigit(meta_folder_path):
        for folder in os.listdir(meta_folder_path):
            for csvs in os.listdir(meta_folder_path + folder):
                if csvs.startswith(tuple('0123456789')):
                    os.remove(meta_folder_path + folder + "/" + csvs)
                    
    def csvToImage(csv_path):
        
        results1 = []
        with open(csv_path) as csvfile:
            reader = csv.reader(csvfile, quoting = csv.QUOTE_NONNUMERIC)
            for row in reader:
                results2 = []
                for col in row:
                    results2.append(col)
                results1.append(results2)
        #unflat_arr = CSVToImage.oneDimToTwoDim(arr, 32)
        #im = Image.fromarray(np.float64(arr), 'L')
        
        return np.array(results1)
    
    def addStringtoFileName(path):
        #Example: /Volumes/Passport/ResearchDataChen/Code/Data/csv_unnorm/vispm/510093797_corr_map.csv
        for root, dirs, files in os.walk(path):

            for file in files: #finds only the CSV files
                path = root + "/" + file
                #print(path)
                pieces = path.split("/")
                region = pieces[len(pieces) - 2]
                piece = pieces[len(pieces) - 1]
        
                pieces2 = piece.split("_")
                copy = pieces2[:]
                copy[1:1] = [region]
        
                separator = "_"
        
                new_name = separator.join(copy)
                #print(new_name)
                os.rename(root + "/" + file, root + "/" + new_name)
                
    def findRegionOfCSV(csv_path):
        #/Volumes/Passport/ResearchDataChen/Code/Data/allBrainRegionsnorm/visrl/553568031_visrl_normalized_corrmap.csv
        pieces = csv_path.split("/")
        region = pieces[len(pieces) - 2]
        return region
    
    def findFileNameOfCSV(csv_path):
        #/Volumes/Passport/ResearchDataChen/Code/Data/allBrainRegionsnorm/visrl/553568031_visrl_normalized_corrmap.csv
        pieces = csv_path.split("/")
        piece = pieces[len(pieces) - 1]
  
        return piece
    
    def csvsToHeatmap(path):
        DIM = 32

        for root, dirs, files in os.walk(path):
            
            for file in files:
                #print(file)
                filepath = root + "/" + file
                
                rel_path = os.path.join(*(filepath.split(os.path.sep)[1:]))
                #print(rel_path)
                #print(DataUtilities.findRegionOfCSV(rel_path))
                rel_path = rel_path[: len(rel_path) - 4]
                
                df = pd.read_csv(filepath, header=None)
                array = df.values
                img = Image.new("RGB", (DIM, DIM), (255, 255, 255))
                pixels = img.load()
                
                for col in range(array.shape[1]):
                    
                    for row in range(array.shape[0]):
                        r = 0
                        g = 0
                        b = 0
                        if array[col][row] >= 0:
                            r = array[col][row] * 255
                        else:
                            g = abs(array[col][row]) * 255
                            
                        b = abs(array[col][row]) * 255
                        pixels[col, row] = (int(r), int(g), int(b))

                img.save("/Volumes/Passport/ResearchDataChen/Code/Data/csv_unnorm_imgs/" + DataUtilities.findRegionOfCSV(rel_path) + "/ " + file + ".png")
 
    def listAllCSVsOfFolders(path):
        listOfFilePaths = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if not file.startswith("._"):
                    filepath = root + "/" + file
                    listOfFilePaths.append(filepath)
                
        return listOfFilePaths
    
    def chooseRandomFile(list_of_csv_dirs):
        rand_csvfile = random.choice(list_of_csv_dirs)
        
        return rand_csvfile
    
    def random_noise(image_array: ndarray):
    # add random noise to the image
        return sk.util.random_noise(image_array)
    
    '''
    Say we currently have 6 folders (that represent different regions of the brain), each folder contains different number of images that are correctly labelled to its folders. But the goal is to add noise to these images, so we'd have to make a new dataset that contains the modified images. We add noise in the method addAdditiveNoiseToCSVs in line 217. Where i'm struggling is getting the new dataset that is being made include the same amount of images per folder. A requirement is that I have to choose the images to add noise to randomly.
    
    Right now I'm thinking I randomly pick images and if the count for the number of new images that have been generated for that class (folder) is below the desired amount, keep randomly choosing. However, after running the function below, it generates different amount of images for each region. How could I fix this?
    
    '''
    #input is just one folder -one region
    def addAdditiveNoiseToCSVs(folder_path, num_files_desired):
        
        images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if not f.startswith("._") if os.path.isfile(os.path.join(folder_path, f))]
        
        num_generated_files = 0
        while num_generated_files <= num_files_desired:
            # random image from the folder
            
            image_path = random.choice(images)
            
            region = DataUtilities.findRegionOfCSV(image_path)
            
            # read image as an two dimensional array of pixels
            image_to_transform = sk.io.imread(image_path)
            
            #^num of transformations to one image, but the function that adds noise itself is adding it to every pixel
            transformed_image = None

            transformed_image = DataUtilities.random_noise(image_to_transform)
        
            new_file_path = "/Volumes/Passport/ResearchDataChen/Code/Data/just_noisy_imgs/" + region + "/" + str(num_generated_files) + "_" + DataUtilities.findFileNameOfCSV(image_path)
        
            # write image to the disk
            sk.io.imsave(new_file_path, transformed_image)
            num_generated_files += 1
    
    #going into the train data structure that it's currently on
    def randomShuffle(path):
        listOfLabels = []
        listOfImgPaths = []
        
        for root, folders, file in os.walk(path):
            for f in file:
                listOfImgPaths.append(os.path.join(root, f))
                pieces = f.split("_")
                label = pieces[2]
                listOfLabels.append(label)
        #print(listOfImgPaths)
        #print(listOfLabels)
        random.shuffle(listOfLabels) #now it's shuffled
        #/Volumes/Passport/ResearchDataChen/Code/Data/official_all_regions_input/train/visrl/200_ 603592541_visrl_corr_map.csv.png
        #now change the dir names and make a new directory accordingly
        shuffledListOfImgPaths = []
        sep = "/"
        for i in range(len(listOfImgPaths)): # listOfLabels should be same length
            pieces = listOfImgPaths[i].split("/")
            newLabel = listOfLabels[i]
            pieces[6] = "shuffled_official_all_regions_input"
            pieces[8] = newLabel
            shuffledListOfImgPaths.append(sep.join(pieces))
        
        for i in range(len(listOfImgPaths)):
            shutil.copy2(listOfImgPaths[i],shuffledListOfImgPaths[i])
        #print(shuffledListOfImgPaths)
        #now I should have a list of new
            
        

DataUtilities.randomShuffle("/Volumes/Passport/ResearchDataChen/Code/Data/official_all_regions_input/train/")

#DataUtilities.addAdditiveNoiseToCSVs("/Volumes/Passport/ResearchDataChen/Code/Data/non-noisy_csv_unnorm_imgs/visal/", 200)
#DataUtilities.addAdditiveNoiseToCSVs("/Volumes/Passport/ResearchDataChen/Code/Data/non-noisy_csv_unnorm_imgs/visam/", 200)
#DataUtilities.addAdditiveNoiseToCSVs("/Volumes/Passport/ResearchDataChen/Code/Data/non-noisy_csv_unnorm_imgs/visl/", 200)
#DataUtilities.addAdditiveNoiseToCSVs("/Volumes/Passport/ResearchDataChen/Code/Data/non-noisy_csv_unnorm_imgs/visp/", 200)
#DataUtilities.addAdditiveNoiseToCSVs("/Volumes/Passport/ResearchDataChen/Code/Data/non-noisy_csv_unnorm_imgs/vispm/", 200)
#DataUtilities.addAdditiveNoiseToCSVs("/Volumes/Passport/ResearchDataChen/Code/Data/non-noisy_csv_unnorm_imgs/visrl/", 200)

#DataUtilities.addAdditiveNoiseToCSVs("/Volumes/Passport/ResearchDataChen/Code/Data/csv_unnorm/", 200)
#DataUtilities.csvsToHeatmap("/Volumes/Passport/ResearchDataChen/Code/Data/csv_unnorm")
#DataUtilities.addStringtoFileName("/Volumes/Passport/ResearchDataChen/Code/Data/csv_unnorm")
#DataUtilities.deleteFilesStartingWith("/Volumes/Passport/ResearchDataChen/Code/Data/all_regions_input/train/", "noisy")
#DataUtilities.deleteFilesStartingWithDigit("/Volumes/Passport/ResearchDataChen/Code/Data/shuffled_official_all_regions_input/train/")
#DataUtilities.csvsToImage("/Volumes/Passport/ResearchDataChen/Code/Data/all_regions_input")
