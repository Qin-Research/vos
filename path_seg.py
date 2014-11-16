from video_util import *
from pylab import *
import numpy as np
from sys import argv
from time import time
import os
from scipy.io import loadmat
from sklearn.preprocessing import scale
from skimage.color import rgb2gray,rgb2lab
from skimage.feature import hog
from skimage import img_as_ubyte
from joblib import Parallel, delayed
from skimage.segmentation import find_boundaries
from video_graph import *
from video_util import *
from IPython.core.pylabtools import figsize
from scipy.sparse import csr_matrix
from sklearn.mixture import GMM
from sklearn.ensemble import RandomForestClassifier
from krahenbuhl2013 import DenseCRF
import ipdb


def segment(frames, unary, pair_features ,segs,potts_weights, sp_label,paths):
    n_nodes = unary.shape[0]
    potts =                np.array([[0,1],
                                     [1,0]], np.float32)

    crf = DenseCRF(n_nodes, 2)

    print 'Mean field inference ...'        
    crf.set_unary_energy(unary.astype(np.float32))
    
    for (i,f) in enumerate(pair_features):
      crf.add_pairwise_energy((potts_weights[i] * potts).astype(np.float32), np.ascontiguousarray(pair_features[i]).astype(np.float32))
    
    iteration = 10

    labels = crf.map(iteration)
    belief = crf.inference(iteration)
#    labels = np.max(belief, axis=1)
    print ' done.'

    count = 0
    mask = []

    label_all = np.zeros(sp_label.shape, np.bool)
    for (i,id) in enumerate(paths.keys()):
        if labels[i] == 0:
            label_all[sp_label == id] = 1
        else:
            label_all[sp_label == id] = 0
            
    for i in range(sp_label.shape[2]):
        mask.append(label_all[:,:,i])
    # for j in range(len(segs)-1):
    #     uni = np.unique(segs[j])

    #     new_mask = np.zeros(segs[j].shape)
    #     for u in uni:
    #         rows, cols = np.nonzero(segs[j] == u)
    #         if labels[count] == 0:
    #             new_mask[rows, cols] = 1
    #         else:
    #             new_mask[rows, cols] = 0
                
    #         count += 1
            
    #     mask.append(new_mask)
                
    return mask,belief


def compare(mask1,mask2):

    figure(figsize(21,18))
    for i in range(len(mask1)):
        subplot(1,2,1)
        imshow(mask1[i])
        subplot(1,2,2)
        imshow(mask2[i])

        show()
        
def path_unary(frames, segs, sp_label, sp_unary, mappings, paths):
    n_paths = len(paths)
    unary = np.zeros((n_paths, 2))
    n_frames = len(frames)

    id_mapping = {}
    for (i,id) in enumerate(paths.keys()):
        id_mapping[id] = i

    count = 0
    for i in range(n_frames):
        uni = np.unique(segs[i])

        for u in uni:
            orig_id = mappings[i][:u]
            p_id = id_mapping[orig_id]
            u_fg = sp_unary[count][0]
            u_bg = sp_unary[count][1]

            unary[p_id][0] = max(unary[p_id][0], u_fg)
            unary[p_id][1] = max(unary[p_id][1], u_bg)

            count += 1

    return unary

def path_feature(frames, paths):
    n_paths = len(paths)
    feature = np.zeros((n_paths, 6))

    id_mapping = {}
    for (i,id) in enumerate(paths.keys()):
        id_mapping[id] = i

    ims = []
    for f in frames:
        ims.append(img_as_ubyte(imread(f)))
        
    for (i,id) in enumerate(paths.keys()):
        path = paths[id]
        rows = path.rows
        cols = path.cols
        frame = path.frame
        unique_frame = np.unique(frame)

        mc = np.zeros(3)
        mean_f = np.mean(unique_frame)

        for f in unique_frame:
            rrows = rows[frame == f]
            ccols = cols[frame == f]
            mc += np.mean(ims[f][rrows, ccols], axis=0)
            if round(mean_f) == f:
                m_x = np.mean(rrows)
                m_y = np.mean(ccols)

        loc = [m_x, m_y, mean_f]
        feature[id_mapping[id], 3:] = loc
        feature[id_mapping[id], :3] = mc / len(unique_frame)
        
    return feature

def plot_value(paths, sp_label, values):

    val = np.zeros(sp_label.shape)
    for (i,id) in enumerate(paths.keys()):
        val[sp_label == id] = values[i]

    for i in range(sp_label.shape[2]):
        imshow(val[:,:,i])
        show()
        

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

name = 'bmx'
imdir = '/home/masa/research/code/rgb/%s/' % name
vx = loadmat('/home/masa/research/code/flow/%s/vx.mat' % name)['vx']
vy = loadmat('/home/masa/research/code/flow/%s/vy.mat' % name)['vy']

frames = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)) if f.endswith(".png")] 
from skimage.filter import vsobel,hsobel

mag = np.sqrt(vx**2 + vy ** 2)
r,c,n_frames = mag.shape
n_frames+=1
sp_file = "../TSP/results2/%s.mat" % name
sp_label = loadmat(sp_file)["sp_labels"]
segs,mappings = get_tsp(sp_label)
n = np.unique(sp_label)
paths = {}

for (i,id) in enumerate(n):
    mask = sp_label == id
    rows, cols, frame = np.nonzero(mask)
    paths[id] =  Path(n, rows, cols, frame)
    
unary = loadmat('/home/masa/research/FastVideoSegment/unary_%s.mat'% name)['unaryPotentials']

p_u =  path_unary(frames, segs, sp_label, unary, mappings, paths)
p_feature = path_feature(frames, paths)
potts_weights = [0.1]

color = p_feature[:,:3] / 30
loc = p_feature[:,3:] / 80
pair_features = [np.hstack((color, loc))]
mask,belief =  segment(frames, p_u, pair_features ,segs,potts_weights, sp_label,paths)
#mask =  segment(frames, p_u, pair_features ,segs,potts_weights, sp_label,paths)

for i in range(n_frames-1):
    figure(figsize(20,18))

    print i
    im = img_as_ubyte(imread(frames[i]))            
    subplot(1,2,1)
    imshow(im)
    axis("off")

    subplot(1,2,2)
    imshow(alpha_composite(im, mask_to_rgb(mask[i], (0,255,0))),cmap=gray())        
    axis("off")    
    
    show() 

