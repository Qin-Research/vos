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

name = 'cheetah'
#name = 'hum'
imdir = '/home/masa/research/code/rgb/%s/' % name
vx = loadmat('/home/masa/research/code/flow/%s/vx.mat' % name)['vx']
vy = loadmat('/home/masa/research/code/flow/%s/vy.mat' % name)['vy']

frames = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)) if f.endswith(".png")]
from skimage.filter import vsobel,hsobel
sp_file = "../TSP/results2/%s.mat" % name
sp_label = loadmat(sp_file)["sp_labels"]

class Path:
    def __init__(self, id, rows, cols, frame):
        self.id = id
        self.rows = rows
        self.cols = cols
        self.frame = frame

    def plot_path(self, frames):
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
        
n = np.unique(sp_label)

gt = get_segtrack_gt(name)
n_gt = len(gt)
gt_label = np.zeros(sp_label.shape, np.bool)
for i in range(sp_label.shape[2]):
    gt_label[:,:,i] = gt[0][i].astype(np.bool)
    if n_gt > 1: gt_label[:,:,i] += gt[1][i].astype(np.bool)

inlier = []
inlier_count = []
outlier = []
outlier_count = []
paths = {}
label_count = {}
single_label = []
label_order = {}
labels = []
for (i,id) in enumerate(n):
    mask = sp_label == id
    rows, cols, frame = np.nonzero(mask)
    paths[id] =  Path(n, rows, cols, frame)
    c = len(np.unique(frame))

    if c == 1:
        single_label.append(id)
        labels.append(gt_label[rows, cols, frame[0]] > 0.6)
        continue
#    if c > 2: long_paths[id] = Path(n, rows, cols, frame)
    # if np.sum(gt_label[mask]) > 10:
    #     unique_frame = np.unique(frame)
    #     ok = True
    label_count[id] = np.zeros(2)    
    unique_frame = np.unique(frame)
    label_order[id] = []
    for u in unique_frame:
        rs = rows[frame == u]
        cs = cols[frame == u]
        if np.mean(gt_label[rs,cs,u]) > 0.6:
            label_count[id][0] += 1
            label_order[id].append(0)
        else:
            label_count[id][1] += 1
            label_order[id].append(1)
            
    if label_count[id][0] > label_count[id][1]:
        labels.append(1)
    else:
        labels.append(0)
            
            
outlier = {}
outlier_order = {}
x = []
y = []
long_fg = {}
long_bg = {}
long_n = 5

for i in label_count.keys():
    # x.append(label_count[i][0])
    # y.append(label_count[i][1])
    if label_count[i][0] >= long_n:
        long_fg[i] = label_count[i]        
    if label_count[i][1] >= long_n:
        long_bg[i] = label_count[i]
    if label_count[i][0] != 0 and label_count[i][1] != 0:
        outlier[i] = label_count[i]
        x.append(label_count[i][0])
        y.append(label_count[i][1])
        outlier_order[i] = label_order[i]

        
label_all = np.zeros(gt_label.shape, np.bool)        
for (i,id) in enumerate(paths.keys()):
    if labels[i] == 0:
        label_all[sp_label == id] = 0
    else:
        label_all[sp_label == id] = 1
        
gt_single = np.zeros(gt_label.shape,np.bool)

for s in single_label:
    gt_single[sp_label == s] = 1

for i in range(gt_single.shape[2]):
    imshow(gt_single[:,:,i])
    show()

for i in outlier_order.keys():
    if len(outlier_order[i]) > 3:
     print i, outlier_order[i]    
# inlier_count = []
# for i in inlier:
#   inlier_count.append()

# inlier_count = np.array(inlier_count)  

#hummingbird
#failure: 5438
