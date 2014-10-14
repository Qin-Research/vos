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
sp_file = "../code/TSP/results/%s.mat" % name
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

sigma2 = 20000
for i in range(n_frames):
    uni = np.unique(segs[i])
    n_node += len(uni)
    print i
    for u in uni:
        for (n_id,adj) in enumerate(adjs[i][u]):
            if adj == False: continue
            if node_id[i][u] == node_id[i][n_id]: continue
            rows.append(node_id[i][u])
            cols.append(node_id[i][n_id])
            values.append(np.exp(-np.linalg.norm(feats[i][u] - feats[i][n_id]) ** 2 / (2*sigma2)))
            values.append(values[-1])
            cols.append(node_id[i][u])
            rows.append(node_id[i][n_id])

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



sal = np.array(rhs)
A = inv_D.dot(W)

for i in range(10):
    sal = A.dot(sal)

#sal = (sal - np.min(sal)) / (np.max(sal) - np.min(sal))        
count = 0
from skimage import img_as_ubyte
thres = 0.5
for i in range(n_frames):
    sal_image = np.zeros((r,c))
    im = img_as_ubyte(imread(frames[i]))    
    uni = np.unique(segs[i])
    s = sal[count:count+len(uni)]
    s = (s - np.min(s)) / (np.max(s) - np.min(s))
    for (j,u) in enumerate(uni):
        rs, cs = np.nonzero(segs[i] == u)
        sal_image[rs,cs] = s[j]
        if s[j] < thres:
            im[rs,cs] = (0,0,0)
        count += 1


    figure(figsize(20,15))
    subplot(1,3,1)
    imshow(init_sal[:,:,i],cmap=gray())

    subplot(1,3,2)
    imshow(sal_image,cmap=gray())

    subplot(1,3,3)
    imshow(im,cmap=gray())
    
