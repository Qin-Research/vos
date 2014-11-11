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
# from krahenbuhl2013 import DenseCRF
# #prop = proposals.Proposal( setupBaseline( 130, 5, 0.8 ) )
# #prop = proposals.Proposal( setupBaseline( 150, 7, 0.85 ) )
# prop = proposals.Proposal( setupLearned( 150, 5, 0.8 ) )
# #prop = proposals.Proposal( setupLearned( 160, 6, 0.85 ) )

# detector = contour.MultiScaleStructuredForest()
# detector.load( "sf.dat" )
#name = 'soldier'
name = 'hummingbird'
def get_dominant_motion(motion):
    hist,bins = np.histogram(motion.flatten(), bins=500)
    return bins[np.argmax(hist)]

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
paths = {}
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

for (i,id) in enumerate(n):
    mask = sp_label == id
    rows, cols, frame = np.nonzero(mask)
    paths[id] =  Path(n, rows, cols, frame)
    c = len(np.unique(frame))

    if c == 1: continue
    if np.sum(gt_label[mask]) > 10:
        unique_frame = np.unique(frame)
        ok = True
        for u in unique_frame:
            rs = rows[frame == u]
            cs = cols[frame == u]
            if np.mean(gt_label[rs,cs,u]) > 0.5:
                continue
            else:
                outlier.append(id)
                outlier_count.append(c)
                ok = False
                break
        if ok:
             inlier.append(id)
             inlier_count.append(c)
        
inlier = np.array(inlier)
outlier = np.array(outlier)
inlier_count = np.array(inlier_count)
outlier_count = np.array(outlier_count)

# inlier_count = []
# for i in inlier:
#   inlier_count.append()

# inlier_count = np.array(inlier_count)  
