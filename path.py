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


def get_paths():                        
    name = 'soldier'
    name = 'girl'
    imdir = '/home/masa/research/code/rgb/%s/' % name
    vx = loadmat('/home/masa/research/code/flow/%s/vx.mat' % name)['vx']
    vy = loadmat('/home/masa/research/code/flow/%s/vy.mat' % name)['vy']
    
    frames = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)) if f.endswith(".png")]
    imgs = [img_as_ubyte(imread(f)) for f in frames]
            
    from skimage.filter import vsobel,hsobel
    sp_file = "../TSP/results2/%s.mat" % name
    sp_label = loadmat(sp_file)["sp_labels"][:,:,:-1]
            
    paths = {}
    n = np.unique(sp_label)            

    for (i,id) in enumerate(n):
        mask = sp_label == id
        rows, cols, frame = np.nonzero(mask)
        paths[id] =  Path(n, rows, cols, frame, imgs, vx, vy)
        
    from cPickle import dump
    with open('paths_%s.pickle' % name, 'w') as f:
        dump(paths,f)

    return paths
