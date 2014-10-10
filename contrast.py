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

name = 'bmx'
#name = 'bmx'
name = 'hummingbird'

imdir = '/home/masa/research/code/rgb/%s/' % name
vx = loadmat('/home/masa/research/code/flow/%s/vx.mat' % name)['vx']
vy = loadmat('/home/masa/research/code/flow/%s/vy.mat' % name)['vy']

frames = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)[:-1]) if f.endswith(".png")]
from skimage.filter import vsobel,hsobel

mag = np.sqrt(vx**2 + vy ** 2)
angle = np.arctan2(vy,vx)
sp_file = "../TSP/results/%s.mat" % name
sp_label = loadmat(sp_file)["sp_labels"]
segs,adjs,mappings = get_tsp(sp_label)

for j in range(vx.shape[2]):
    uni = np.unique(segs[j])
    flow_mean = np.zeros(len(uni))
    angle_mean = np.zeros(len(uni))
    mean_pos = np.zeros((len(uni),2))
    
    mag_image = np.zeros(segs[j].shape)
    angle_image = np.zeros(segs[j].shape)

    for (i,u) in enumerate(uni):
        rows, cols = np.nonzero(segs[j] == u)
        fm = np.mean(mag[rows, cols,j])
        am = np.mean(angle[rows, cols,j])
        flow_mean[i] = fm
        angle_mean[i] = am
        mean_pos[i,0] = np.mean(rows)
        mean_pos[i,1] = np.mean(cols)
        mag_image[rows, cols] = fm
        angle_image[rows, cols] = am
        
    from IPython.core.pylabtools import figsize
    figure(figsize=(20,15))

        
    subplot(1,6,1)
    imshow(mag_image, cmap=gray())

    subplot(1,6,2)
    imshow(angle_image, cmap=gray())
        

    flow_sal = np.zeros(len(uni))
    angle_sal = np.zeros(len(uni))

    sal_image = np.zeros(segs[j].shape)
    sal_image2 = np.zeros(segs[j].shape)
    sal_image3 = np.zeros(segs[j].shape)
    sal_image4 = np.zeros(segs[j].shape)
    
    for (i,u) in enumerate(uni):
        fm = flow_mean[i]
        am = angle_mean[i]
        pm = mean_pos[i]
        sm = 0
        sm2 = 0
        sm3 = 0
        sm4 = 0
        for (ii,uu) in enumerate(uni):
            if ii == i: continue
            w = np.exp(-0.5 * 1.0/(0.25**2) * np.linalg.norm(pm - mean_pos[ii])**2)
            sm += np.abs(flow_mean[ii] - fm)
            sm2 += np.abs(angle_mean[ii] - am)
            sm3 += w * np.abs(flow_mean[ii] - fm)
            sm4 += w*np.abs(angle_mean[ii] - am)
            
        rows, cols = np.nonzero(segs[j] == u)
        sal_image[rows, cols] = sm
        sal_image2[rows, cols] = sm2
        sal_image3[rows, cols] = sm3
        sal_image4[rows, cols] = sm4

    print j
    subplot(1,6,3)
    imshow(sal_image3, cmap=gray())

    subplot(1,6,4)
    imshow(sal_image4, cmap=gray())
    
    subplot(1,6,5)
    imshow(sal_image, cmap=gray())

    subplot(1,6,6)
    imshow(sal_image2, cmap=gray())
    show()
                    
                


        

    
