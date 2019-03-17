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

import os, time, numpy, math, json, warnings, csv, sys, collections, cv2
import numpy as np
import nibabel as nib
import multiprocessing
from multiprocessing import Barrier, Lock, Process

import scipy.ndimage as ndimage
from filters.gaussmedfilt import gaussmedfilt
from filters.medfilt import medfilt
from filters.circthresh import circthresh
from filters.circthresh2 import circthresh2
from skimage.measure import regionprops, label
from PIL import Image
from skimage import io
from natsort import natsorted
from filters.rollingballfilt import rolling_ball_filter
import itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 1000000000

benchmark = True
parallel = True

new_circ = True
mask = True
over_sample = True
use_medfilt = False
count_path = "/mnt/TissueCyte80TB/181024_Gerald_HET/het-Mosaic/Ch2_Stitched_Sections"
mask_path = "/mnt/TissueCyte80TB/181024_Gerald_HET/het-Mosaic/Het_seg_10um.tif"
file, extension = os.path.splitext(mask_path)
if extension == '.nii':
    seg = nib.load(mask_path).get_data()
else:
    seg = io.imread(mask_path)
print('Loaded segmentation data')
print(seg.shape);



################################################################################
#  Function definitions
################################################################################


def count_process(slice_number_cp, count_file, structure_cp, circ_thresh=0.8, bg_thresh = 6., size = 85., radius = 12.):
    print(('%s working on:\t %s' % (multiprocessing.current_process().name, slice_number_cp)))
    print(seg.shape)
    # Load image and convert to dtype=float and scale to full 255 range
    current_cells = dict()
    with Image.open(count_path + '/' + count_file) as image:
        image = np.array(image).astype(float)
        image = np.multiply(np.divide(image, np.max(image)), 255.)
        print(('%s contrast in \t \t %s' % (multiprocessing.current_process().name, slice_number_cp)))
       
        # Apply mask if required
        if mask:
            print(('%s applying mask in \t \t  %s' % (multiprocessing.current_process().name, slice_number_cp)))
            mask_image = np.array(
                Image.fromarray(seg[slice_number_cp]).resize(tuple([int(x) for x in temp_size]), Image.NEAREST))
            mask_image[mask_image != structure_cp] = 0
            image[mask_image == 0] = 0

            # Crop image with idx
            mask_image = image > 0
            idx = np.ix_(mask_image.any(1), mask_image.any(0))
            mask_image = None
            row_idx = idx[0].flatten()
            col_idx = idx[1].flatten()
            image = image[idx]


        # Perform gaussian donut median filter

        if use_medfilt:
            print(('%s applying filter in \t \t \t%s' % (multiprocessing.current_process().name, slice_number_cp)))
            image = gaussmedfilt(image, 3, 1.5)
        else:
            image = cv2.GaussianBlur(image, ksize=(0,0), sigmaX=3)

        if image.shape[0] * image.shape[1] > (radius * 2) ** 2 and np.max(image) != 0.:
            # image = np.multiply(np.divide(image, np.max(image)), 255.)
            # Perform rolling ball background subtraction to remove uneven background signal
            print(('%s rolling in \t \t \t %s' % (multiprocessing.current_process().name, slice_number_cp)))
            image, background = rolling_ball_filter(np.uint8(image), 24)
            # Image.fromarray(image).save('/home/gm515/Documents/Temp3/Z_'+str(slice_number+1)+'.tif')

            if np.max(image) != 0.:
                # Perform circularity threshold
                print(('%s thresh in \t \t \t  %s' % (multiprocessing.current_process().name, slice_number_cp)))
                if new_circ:
                    image = image > circthresh2(image,size,bg_thresh,circ_thresh)
                else:
                    image = image > circthresh(image, size, bg_thresh)

                # Remove objects smaller than chosen size
                image_label = label(image, connectivity=image.ndim)

                def circfunc(r):
                    return (4 * math.pi * r.area) / ((r.perimeter * r.perimeter) + 0.00000001)

                # Centroids returns (row, col) so switch over
                circ = [circfunc(region) for region in regionprops(image_label)]
                areas = [region.area for region in regionprops(image_label)]
                labels = [region.label for region in regionprops(image_label)]
                centroids = [region.centroid for region in regionprops(image_label)]

            # Convert coordinate of centroid to coordinate of whole image if mask was used
                if mask:
                    def coordfunc(celly, cellx):
                        return row_idx[celly], col_idx[cellx]

                    # (row, col) or (y, x)
                    centroids = [coordfunc(int(c[0]), int(c[1])) for c in centroids]
                    # Threshold the objects based on size and circularity and store centroids
                    cells = []
                    for ii, _ in enumerate(areas):
                        if areas[ii] > size and areas[ii] < size * 10 and circ[ii] > 0.65:
                            cells.append(centroids[ii][::-1])
        else:
            cells = []
        current_cells.update({slice_number_cp: cells})
        print(('%s finished with \t \t \t \t%s!!!!!' % (multiprocessing.current_process().name, slice_number_cp)))
    return current_cells


def distance(a, b):
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def get_children(json_obj, acr, ids):
    for obj in json_obj:
        if obj['children'] == []:
            acr.append(obj['acronym'])
            ids.append(obj['id'])
        else:
            acr.append(obj['acronym'])
            ids.append(obj['id'])
            get_children(obj['children'], acr, ids)
    return (acr, ids)


def get_structure(json_obj, acronym):
    found = (False, None)
    for obj in json_obj:
        if obj['acronym'].lower() == acronym:
            # print obj['acronym'], obj['id']
            [acr, ids] = get_children(obj['children'], [], [])
            # print ids
            if ids == []:
                acr = [obj['acronym']]
                ids = [obj['id']]
                return (True, acr, ids)
            else:
                acr.append(obj['acronym'])
                ids.append(obj['id'])
                return (True, acr, ids)
        else:
            found = get_structure(obj['children'], acronym)
            if found:
                return found


if __name__ == '__main__':
    ################################################################################
    ## User defined parameters - please fill in the parameters in this section only
    ################################################################################

    # Do you want to use a mask taken from a registered segmentation atlas

    # Do you want to perform over sampling correction?
    # Cells within a radius on successive images will be counted as one cell

    # The following is redundant and will be included when considering volume and density
    # xy_res = 10
    # z_res = 5

    # If you are using a mask, input the mask path and the structures you want to count within
    # E.g. 'LGd, LGv, IGL, RT'
    if mask:
        
        structure_list = 'LGd'  # ,LGv,IGL,RT,LP,VPM,VPL,APN,ZI,LD'

    # Input details for the cell morphology
    # Can be left as default values

    # Input the directory path of the TIFF images for counting

    print(('\n' + count_path))
    # Of the images in the above directory, how many will be counted?
    # Number of files [None,None] for all, or [start,end] for specific range
    number_files = [None, None]

    # Do you want to use the custom donut median filter?

    # For the circularity threshold, what minimum background threshold should be set
    # You can estimate this by loading an image in ImageJ, perform a gaussian filter radius 3, then perform a rolling ball background subtraction radius 8, and choose a threshold which limits remaining background signal

    ################################################################################
    ## Initialisation
    ################################################################################

    # Create directory to hold the counts in same folder as the images
    if not os.path.exists(count_path + '/counts_v5_Balint'):
        os.makedirs(count_path + '/counts_v5_Balint')

    # List of files to count
    count_files = []
    count_files += [each for each in os.listdir(count_path) if each.endswith('.tif')]
    count_files = natsorted(count_files)
    print((str(len(count_files))))
    if number_files[0] is not None:
        count_files = count_files[number_files[0] - 1:number_files[1]]
    print(('Counting in files: ' + count_files[0] + ' to ' + count_files[-1]))

    ################################################################################
    ## Retrieving structures IDs
    ################################################################################

    ids = []
    acr = []
    if mask:
        anno_file = json.load(open('2017_annotation_structure_info.json'))
        structure_list = [x.strip() for x in structure_list.lower().split(",")]
        for elem in structure_list:
            a, i = get_structure(anno_file['children'], elem)[1:]
            ids.extend(i)
            acr.extend(a)
    else:
        ids.extend('n')
        acr.extend('n')
    print(('Counting in structures: ' + str(acr)))

    ################################################################################
    ## Counting
    ################################################################################

    tstart = time.time()

    temp = Image.open(count_path + '/' + count_files[0])
    temp_size = temp.size
    temp = None
    if mask:
        scale = float(temp_size[1]) / seg[1].shape[0]

    structure_index = 0

    for name, structure in zip(acr, ids):
        print(('Counting in ' + str(name)))
        proceed = True

        # Dictionary to store centroids - each key is a new slice number
        total_cells = dict()

        ################################################################################
        ## Obtain crop information for structure if mask required
        ################################################################################
        if mask:
            index = np.array([[], [], []])
            if structure in seg:
                index = np.concatenate((index, np.array(np.nonzero(structure == seg))), axis=1)
            else:
                proceed = False

            if index.size > 0:
                zmin = int(index[0].min())
                zmax = int(index[0].max())
            else:
                proceed = False
        else:
            zmin = 0
            zmax = len(count_files)

        ################################################################################
        ## Check whether to proceed with the count
        ################################################################################
        if proceed:
            if benchmark:
                zmax = zmin+multiprocessing.cpu_count()*2-1
                print('Counting in %s to %s' % (zmin, zmax))
            ################################################################################
            ## Loop through slices based on cropped boundaries
            ################################################################################
            nProcess = multiprocessing.cpu_count()//3 if parallel else 1
            with multiprocessing.Pool(nProcess) as pool:
                print('Started Pool!')
                slices = list(range(zmin, zmax ))
                structures = itertools.repeat(structure, len(slices))
                act_files = count_files[zmin:zmax]
                structures = itertools.repeat(structure, len(slices))
                proc_input = zip(list(slices), act_files, structures)
                results = pool.starmap(count_process, proc_input)

            for cur_cells in results:
                total_cells.update(cur_cells)

        dist=12.
        if over_sample:
            print('Correcting for oversampling')
            for index, (x, y) in enumerate(zip(list(total_cells.values())[1:], list(total_cells.values())[:-1])):
                if len(x) * len(y) > 0:
                    total_cells[zmin + index] = [ycell for ycell in y if
                                                 all(distance(xcell, ycell) > dist ** 2 for xcell in x)]

        num_cells = sum(map(len, total_cells.values()))

        print(str(name) + ' ' + str(num_cells))

        csv_file = count_path + '/counts_v5_Balint/' + str(name) + '_multi_v5_count.csv'
        with open(csv_file, 'w+') as f:
            for key in sorted(total_cells.keys()):
                if len(total_cells[key]) > 0:
                    csv.writer(f, delimiter=',').writerows(
                        np.round(np.concatenate(([((np.array(val))).tolist() for val in
                                                  total_cells[key]],
                                                 np.ones((len(total_cells[key]), 1)) * (
                                                         key + 1)), axis=1)))                                                        
                                                         
        if benchmark: break
        structure_index += 1
    print('~Fin~')

    minutes, seconds = divmod(time.time() - tstart, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    print(('Counting completed in %02d:%02d:%02d:%02d' % (days, hours, minutes, seconds)))
