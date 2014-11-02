from pylab import *
import numpy as np
from sys import argv
from time import time
import os
from scipy.io import loadmat,savemat
from sklearn.preprocessing import scale
from skimage.color import rgb2gray,rgb2lab
from skimage.feature import hog
from joblib import Parallel, delayed
from skimage.segmentation import find_boundaries
from video_graph import *
from IPython.core.pylabtools import figsize
from scipy.sparse import csr_matrix

name = 'bmx'

imdir = '/home/masa/research/code/rgb/%s/' % name
vx = loadmat('/home/masa/research/code/flow/%s/vx.mat' % name)['vx']
vy = loadmat('/home/masa/research/code/flow/%s/vy.mat' % name)['vy']

frames = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)[:-1]) if f.endswith(".png")] 
from skimage.filter import vsobel,hsobel

mag = np.sqrt(vx**2 + vy ** 2)
r,c,n_frames = mag.shape
sp_file = "../TSP/results/%s.mat" % name
sp_label = loadmat(sp_file)["sp_labels"]
segs,adjs,mappings = get_tsp(sp_label)

gt = get_segtrack_gt(name)
g = gt[0][0]

if len(gt)>1: g += gt[1][0]

transfered_label = np.zeros(mag.shape, dtype=np.bool)
transfered_label[:,:,0] = g
from scipy.interpolate import *


for i in range(1,n_frames):

    for y in range(r):
        for x in range(c):
            yy = round(y + vy[y,x,i-1])
            xx = round(x + vx[y,x,i-1])
            if yy >= 0 and yy < r and xx >= 0 and xx < c:
                  transfered_label[yy,xx,i] = transfered_label[y,x,i-1]
        

for i in range(n_frames):
    figure(figsize(21,18))
    subplot(1,2,1)
    imshow(transfered_label[:,:,i],gray())

    subplot(1,2,2)
    g = gt[0][i]

    if len(gt)>1: g += gt[1][i]
    
    imshow(g,gray())
    
    show()
                  
