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
name = 'bmx'
#name = 'bmx'

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
lab_range = get_lab_range(frames)
feats = get_sp_feature_all_frames(frames,segs, lab_range)

node_id = []

id_count = 0
init_sal = np.load('sal_%s.npy' % name)
rhs = []
for i in range(n_frames):
    uni = np.unique(segs[i])
    id_dict = {}
    for u in uni:
        rs, cs = np.nonzero(segs[i] == u)
        rhs.append(np.mean(init_sal[:,:,i][rs,cs]))
        id_dict[u] = id_count
        id_count += 1
    node_id.append(id_dict)
    
rows = []
cols = []
values = []
n_node = 0

sigma2 = 10000
for i in range(n_frames):
    uni = np.unique(segs[i])
    n_node += len(uni)
    print i
    for u in uni:
        for adj in adjs[i][u]:
            if adj == False: continue
            if node_id[i][u] == node_id[i][adj]: continue
            rows.append(node_id[i][u])
            cols.append(node_id[i][adj])
            values.append(np.exp(-np.linalg.norm(feats[i][u] - feats[i][adj]) ** 2 / (2*sigma2)))
            values.append(values[-1])
            cols.append(node_id[i][u])
            rows.append(node_id[i][adj])

        if i < n_frames -1:
            if np.sum(sp_label[:,:,i+1] == mappings[i][:u]) > 0:
                id = node_id[i+1][mappings[i+1][mappings[i][:u]]]
                if node_id[i][u] == id: continue
                rows.append(node_id[i][u])
                cols.append(id)
                values.append(np.exp(-np.linalg.norm(feats[i][:u] - feats[i+1][mappings[i+1][mappings[i][:u]]]) ** 2 / sigma2))
                values.append(values[-1])
                cols.append(node_id[i][u])
                rows.append(id)

from scipy.sparse import csr_matrix, spdiags                                   
W = csr_matrix((values, (rows, cols)), shape=(n_node, n_node))

inv_D =spdiags(1.0/((W.sum(axis=1)).flatten()), 0, W.shape[0], W.shape[1])
from scipy.sparse import eye
lhs = eye(n_node) - (inv_D.dot(W))

from scipy.sparse.linalg import spsolve
sal = spsolve(lhs, np.array(rhs))

sal = (sal - np.min(sal)) / (np.max(sal) - np.min(sal))
sal_image = np.zeros((r,c))

count = 0
for i in range(n_frames):
    uni = np.unique(segs[i])
    for u in uni:
        rs, cs = np.nonzero(segs[i] == u)
        sal_image[rs,cs] = sal[count]
        count += 1

    figure(figsize(20,15))
    subplot(1,2,1)
    imshow(init_sal[:,:,i])

    subplot(1,2,2)
    imshow(sal_image)
