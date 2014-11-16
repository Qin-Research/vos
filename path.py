from pylab import *
import numpy as np
#from util import *
from sys import argv
from time import time
import os
from skimage import img_as_ubyte
from scipy.io import loadmat
from sklearn.preprocessing import scale
from skimage.color import rgb2gray,rgb2lab
from skimage.feature import hog
from joblib import Parallel, delayed
from skimage.segmentation import find_boundaries
from video_graph import *
from IPython.core.pylabtools import figsize
from video_util import *


class Path:
    def __init__(self, id, rows, cols, frame, imgs, vx, vy):
        self.id = id
        self.rows = rows
        self.cols = cols
        self.frame = frame
        self.n_frames = len(np.unique(self.frame))

        unique_frame = np.unique(self.frame)

        mags = []
        angs = []

        self.mean_rgb = np.zeros((len(unique_frame),3))
        self.mean_flows = np.zeros((len(unique_frame), 2))
        
        for (ii,i) in enumerate(unique_frame):
            im = imgs[i]
            rows = self.rows[self.frame == i]
            cols = self.cols[self.frame == i]
            self.mean_rgb[ii] = np.mean(im[rows, cols], axis=0)
            self.mean_flows[ii] = np.array([np.mean(vx[rows, cols, i]), np.mean(vy[rows, cols, i])])
        #     mags.append(np.mean(mag[rows, cols, i]))
        #     angs.append(np.mean(angle[rows, cols, i]))
                                    
        # self.median_mag = np.median(mags)
        # self.median_ang = np.median(angs)
        

    def plot(self, frames):
        unique_frame = np.unique(self.frame)
        for i in unique_frame:
            print i
            im = img_as_ubyte(imread(frames[i]))
            rows = self.rows[self.frame == i]
            cols = self.cols[self.frame == i]
            mask = np.zeros(im.shape[:2], np.bool)
            mask[rows, cols] = 1

            imshow(alpha_composite(im, mask_to_rgb(mask, (0,255,0))), cmap=gray())
            show()
