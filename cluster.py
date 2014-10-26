from pylab import *
import numpy as np
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
from IPython.core.pylabtools import figsize
from scipy.sparse import csr_matrix

#name = 'girl'
name = 'cheetah'

imdir = '/home/masa/research/code/rgb/%s/' % name
vx = loadmat('/home/masa/research/code/flow/%s/vx.mat' % name)['vx']
vy = loadmat('/home/masa/research/code/flow/%s/vy.mat' % name)['vy']

frames = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)[:-1]) if f.endswith(".png")] 
from skimage.filter import vsobel,hsobel

mag = np.sqrt(vx**2 + vy ** 2)

from sklearn.cluster import KMeans

im = mag[:,:,0]
feature = []
dim = 3
for i in range(im.shape[0]):
    for j in range(im.shape[1]):
#        feature += [i,j,vx[i,j,0], vy[i,j,0]]
        feature += [i,j,mag[i,j,0]]

km = KMeans(2)
indx_image = np.zeros(im.shape,np.int)
indx = km.fit_predict(np.array(feature).reshape(-1,dim))
count = 0
for i in range(im.shape[0]):
    for j in range(im.shape[1]):
        indx_image[i,j] = indx[count]
        count += 1
