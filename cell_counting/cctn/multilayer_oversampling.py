# ============================================================================================
# Multilayer cell handling with median centroid
# Author: Gerald M (edited by Balint Hodossy)
#
# This script loads a csv file which contains centroids that may belong to multilayer cells.
# It only keeps centroids that are the median of a given collection of centroids. These in 
# collections are defined by their proximity across layers.
# 
# cell defined as: (centroid, layer number)
#
# Instructions:
# 2) Make changes to parameters and paths as neccessary
# 3) Execute the code in a Python IDE
# ============================================================================================

################################################################################
#  Module import
################################################################################

import os, time, numpy, math, json, warnings, csv, sys, collections
import numpy as np
import nibabel as nib
import multiprocessing
from multiprocessing import Barrier, Lock, Process
import statistics as st
from natsort import natsorted
import itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

filename="LGd-sh_v6_count_INQUEUE.csv"
path="/home/bkh16/"
target="overs.csv"
dist=12.
D=dist**2

################################################################################
#  Function definitions
################################################################################


def distance(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2    

if __name__ == '__main__':

    
    print('Correcting for oversampling')
    
    centroids={}
    with open(path+filename, mode='r') as file:
        reader = csv.reader(file)
        for line in reader:
            centroids.setdefault(int(line[2]), []).append((int(line[0]), int(line[1]))) #Arrange all cells on the same layer into a list at the same key
        
        cell_collection=[]   #Collection of cells which may still have member centroids in the coming layers
        result={}            #Dictionary to be converted to csv file

        for key, layer in centroids.items():
            for cell in cell_collection:         #Cell is a collection centroids belonging to single cell across multiple layers
                if cell[-1][1]<key-1:            #If cell has ended
                    (middle_centroid, middle_key) = (st.median_low(cell)[0], st.median_low(cell)[1])  
                    result.setdefault(middle_key, []).append(middle_centroid) #Store its median in appropriate layer
                    cell_collection.remove(cell) #No need to keep track of it now
                    continue
                layer_copy=list(layer) #Shallow copy so we can change layer while looping through it
                for centroid in layer_copy:
                    if distance(cell[-1][0],centroid) < D:
                        cell.append((centroid, key))
                        layer.remove(centroid) #This leaves only cells that are the start of a new cell
                        
            for centroid in layer:
                cell_collection.append([(centroid, key)])
                
        for cell in cell_collection: #leftover cells stored
            (middle_centroid, middle_key) = (st.median_low(cell)[0], st.median_low(cell)[1])
            result.setdefault(middle_key, []).append(middle_centroid)
                  
        csv_file = path + target 
        with open(csv_file, 'w+') as f:
            for key in sorted(centroids.keys()):
                if len(centroids[key]) > 0:
                    csv.writer(f, delimiter=',').writerows(
                        np.round(np.concatenate(([((np.array(val))).tolist() for val in
                                                  centroids[key]],
                                                 np.ones((len(centroids[key]), 1)) * (
                                                         key )), axis=1)))                                                        
    print('~Fin~')

