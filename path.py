from pylab import *
import numpy as np
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
from IPython.core.pylabtools import figsize
from util import *

class Trajectory:
    # Trajectory represented as video coordinates (x, y, z) of pixels in it
    # Path is the same as trajectory
    # z coordinate is frame index
    # For example, the first pixel's coordinate (x,y,z) is (rows[0], cols[0], frame[0])
    
    def __init__(self, rows, cols, frame):
        self.rows = rows # row index 
        self.cols = cols # col index
        self.frame = frame # frame index
        self.n_frames = len(np.unique(self.frame))

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

def get_paths(name):                        
    
    sp_file = "data/tsp/%s.mat" % name
    sp_label = loadmat(sp_file)["sp_labels"]
            
    paths = {}
    n = np.unique(sp_label)            

    height, width, n_frames = sp_label.shape

    from collections import defaultdict
    rows = defaultdict(list)
    cols = defaultdict(list)
    frames = defaultdict(list)
    
    for k in range(n_frames):
        for i in range(height):
            for j in range(width):
                l = sp_label[i,j,k]
                rows[l].append(i)
                cols[l].append(j)
                frames[l].append(k)
                
    for (i,id) in enumerate(n):
        # mask = sp_label == id
        # rows, cols, frame = np.nonzero(mask)
        paths[id] =  Trajectory(np.array(rows[id]), np.array(cols[id]), np.array(frames[id]))
        
    from cPickle import dump
    with open('data/trajs/%s.pickle' % name, 'w') as f:
        dump(paths,f)

    return paths
