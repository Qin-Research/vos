from pylab import *
import numpy as np
from util import *
from sys import argv
from time import time
import os
from scipy.io import loadmat
from sklearn.preprocessing import scale
from skimage.color import rgb2gray,rgb2lab
from skimage.feature import hog
from joblib import Parallel, delayed
from skimage.segmentation import find_boundaries
from video_graph import *

def prepare(name, dir_name):
    imdir = '/home/masa/research/code/rgb/%s/' % name
    vx = loadmat('/home/masa/research/code/flow/%s/vx.mat' % name)['vx']
    vy = loadmat('/home/masa/research/code/flow/%s/vy.mat' % name)['vy']
    
    frames = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)[:-1]) if f.endswith(".png")]
    
    sp_file = "../TSP/results/%s.mat" % name
    sp_label = loadmat(sp_file)["sp_labels"]
    segs,adjs,mappings = get_tsp(sp_label)

    os.mkdir(dir_name)
    frames_dir = dir_name + "/frames/"
    tsp_dir = dir_name + "/tsp/"
    vx_dir = dir_name + "/vx/"
    vy_dir = dir_name + "/vy/"
    for d in [frames_dir, tsp_dir, vx_dir, vy_dir]:
        os.mkdir(d)

    for i in range(vx.shape[2]):
        imsave(frames_dir + "%05d.png" % i, imread(frames[i]))
        np.save(tsp_dir + "%05d.npy" % i, segs[i].astype(int32))
        np.save(vx_dir + "%05d.npy" % i, vx[:,:,i])
        np.save(vy_dir + "%05d.npy" % i, vy[:,:,i])
