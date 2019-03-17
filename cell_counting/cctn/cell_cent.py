import os, time, numpy, math, json, warnings, csv, sys, collections, cv2
import numpy as np
import nibabel as nib
import multiprocessing
from multiprocessing import Barrier, Lock, Process

import scipy.ndimage as ndimage
from filters.gaussmedfilt import gaussmedfilt
from filters.medfilt import medfilt
from filters.circthresh import circthresh
from skimage.measure import regionprops, label
from PIL import Image, ImageDraw, ImageColor
from skimage import io
from natsort import natsorted
from filters.rollingballfilt import rolling_ball_filter
import itertools

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.simplefilter('ignore', Image.DecompressionBombWarning)
Image.MAX_IMAGE_PIXELS = 1000000000


def process(im_in):
    print("Im in")
    size = 85.
    radius = 12.
    circ_thresh = 0.8
    bg_thresh = 6.

    im_out = np.array(im_in).astype(float)

    im_out = np.multiply(np.divide(im_out, np.max(im_out)), 255.)
    im_out=im_out.astype('uint8')
    print(np.max(im_out))
    im_out = Image.fromarray(im_out, mode='L')
    
    return im_out
   
  # im_out = ndimage.gaussian_filter(im_out, sigma=(3, 3))

    #im_out, background = rolling_ball_filter(np.uint8(im_out), 8)

   # im_out = im_out > circthresh2(im_out,size,bg_thresh,circ_thresh)


    #image_label = label(im_out, connectivity=im_out.ndim)

   # def circfunc(r):
       # return (4 * math.pi * r.area) / ((r.perimeter * r.perimeter) + 0.00000001)

    # Centroids returns (row, col) so switch over
    #circ = [circfunc(region) for region in regionprops(image_label)]
    #areas = [region.area for region in regionprops(image_label)]
    #labels = [region.label for region in regionprops(image_label)]
    #centroids = [region.centroid for region in regionprops(image_label)]
    #cells = []
    #for ii, _ in enumerate(areas):
       # if areas[ii] > size / 2.5 and areas[ii] < size * 10 and circ[ii] > 0.65:
          #  cells.append(centroids[ii][::-1])
    # im_out = np.floor(im_out*255)
    # im_out = Image.fromarray(im_out)
    #im_out = im_in.convert('RGB')
    #draw = ImageDraw.Draw(im_out)
   # for cur_cell in cells:
   #    draw.line([(cur_cell[0] - 20, cur_cell[1]), (cur_cell[0] + 20, cur_cell[1])], fill="red")
   #     draw.line([(cur_cell[0], cur_cell[1] - 20), (cur_cell[0], cur_cell[1] + 20)], fill="red")
    



if __name__ == '__main__':
    count_path = "/mnt/TissueCyte80TB/181024_Gerald_HET/het-Mosaic/Ch2_Stitched_Sections"
    sz = 5000
    x = 5960
    y = 7494
    with Image.open(count_path+"/Stitched_Z054.tif") as im:
        im1=im.crop((x-sz,y-sz,x+sz,y+sz))
        im1.save(count_path+"/counts_v5_Balint/test4.tif")
        im1 = process(im1)

    im1.save(count_path+"/counts_v5_Balint/test.tif")

