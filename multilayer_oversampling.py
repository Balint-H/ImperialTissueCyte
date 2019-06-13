# ============================================================================================
# Cell Counting in Target Nuclei Script
# Author: Gerald M
#
# This script performs automated cell counting in anatomical structures of interest, or a
# a stack of TIFFs. It works by first determining an ideal threshold based on the circularity
# of objects. Then by tracking cells/objects over multiple layers to account for oversampling.
# The output provides a list of coordinates for identified cells. This should then be fed
# into the image predictor to confirm whether objects are cells or not.
#
# Version 2 - v2
# This version differes from original by removing all empty rows and columns to further
# crop each image. In addition, a rolling ball background subtration is used to remove
# uneven background and generally help the cell segmentation process.
#
# Instructions:
# 1) Go to the user defined parameters from roughly line 80
# 2) Make changes to those parameters as neccessary
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

benchmark = False

filename="LGd-sh_v6_count_INQUEUE.csv"
path="/home/bkh16/"
target="overs.csv"

################################################################################
#  Function definitions
################################################################################


def distance(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2    

if __name__ == '__main__':

    tstart = time.time()
    dist=12.
    D=dist**2
    print('Correcting for oversampling')
    
    centroids={}
    with open(path+filename, mode='r') as file:
        reader = csv.reader(file)
        for line in reader:
            centroids.setdefault(int(line[2]), []).append((int(line[0]), int(line[1])))
        
        cell_collection=[]
        result={}
                  

        for key, layer in centroids.items():
            layer_copy=list(layer)
            for cell in cell_collection: #cell is a collection centroids belonging to single cell across multiple layers
                if cell[-1][1]<key-1:
                    (middle_centroid, middle_key) = (st.median_low(cell)[0], st.median_low(cell)[1])
                    result.setdefault(middle_key, []).append(middle_centroid)
                    cell_collection.remove(cell)
                    continue
                for centroid in layer_copy:
                    if distance(cell[-1][0],centroid) < D:
                        cell.append((centroid, key))
                        layer.remove(centroid)
            for centroid in layer:
                cell_collection.append([(centroid, key)])
                
        for cell in cell_collection:
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

    minutes, seconds = divmod(time.time() - tstart, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    print(('Counting completed in %02d:%02d:%02d:%02d' % (days, hours, minutes, seconds)))
