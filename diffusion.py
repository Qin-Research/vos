from pylab import *
import numpy as np
from sys import argv
from time import time
import os
from skimage import img_as_ubyte
from scipy.io import loadmat,savemat
from sklearn.preprocessing import scale
from skimage.color import rgb2gray,rgb2lab
from skimage.feature import hog
from joblib import Parallel, delayed
from skimage.segmentation import find_boundaries
from IPython.core.pylabtools import figsize
from path import Path
from scipy.sparse import csr_matrix, spdiags

def diffuse_inprob(inratios,paths, segs):
                               
    name = 'bmx'
    
    imdir = 'data/rgb/%s/' % name
    
    frames = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)) if f.endswith(".png")]
    imgs = [img_as_ubyte(imread(f)) for f in frames]
            
    init_ratio = []
    id2index = []
    index = 0
    for i in range(len(inratios)):
        id2index.append({})
        for (jj,j) in enumerate(inratios[i][0]):
            init_ratio.append(j[0])
            id2index[i][jj] = index
            index += 1
    
    dist = []
    row_index = []
    col_index = []
    rgbs = np.zeros((len(init_ratio),3))
    n_frames = len(frames)        
    for (i,id) in enumerate(paths.keys()):
        frame = paths[id].frame
        rows = paths[id].rows
        cols = paths[id].cols
    
        unique_frame = np.unique(frame)
    
        for f in unique_frame:
    
            if f == n_frames-1: continue
            r1 = rows[frame == f]
            c1 = cols[frame == f]
            index1 = id2index[f][segs[f][r1[0],c1[0]]]
            
            rgb1 = np.mean(imgs[f][r1,c1],axis=0)
            rgbs[index1] = rgb1
            for f2 in unique_frame:
                if f >= f2: continue
                if f2 == n_frames-1: continue            
                r2 = rows[frame == f2]
                c2 = cols[frame == f2]
                rgb2 = np.mean(imgs[f2][r2,c2],axis=0)
                
                index2 = id2index[f2][segs[f2][r2[0],c2[0]]]
                rgbs[index2] = rgb2
                
                d = np.linalg.norm(rgb1-rgb2) **2
    
                row_index.append(index1)
                row_index.append(index2)
                col_index.append(index2)
                col_index.append(index1)
    
                dist.append(d)
                dist.append(d)
    
    adjs = []            
    for f in range(len(segs)-1):
        adjs.append(get_sp_adj(segs[f]))
        
    for i in range(n_frames-1):
        for j in range(adjs[i].shape[0]):
            index1 = id2index[i][j]
            rgb1 = rgbs[index1]
    
            row_index.append(index1)
            col_index.append(index1)
            dist.append(0)
            
            for k in np.nonzero(adjs[i][j])[0]:
    
                if j > k: continue
                index2 = id2index[i][k]
                rgb2 = rgbs[index2]
                
                d = np.linalg.norm(rgb1-rgb2) **2
    
                row_index.append(index1)
                row_index.append(index2)
                col_index.append(index2)
                col_index.append(index1)
    
                dist.append(d)
                dist.append(d)
    
    sigma = 30 
    #sigma2 = 1000
    values2 = np.exp(-np.array(dist) / (2*sigma**2))
    
    from scipy.sparse import csr_matrix, spdiags
    n_node = len(init_ratio)
    W = csr_matrix((values2, (row_index, col_index)), shape=(n_node, n_node))
    
    inv_D =spdiags(1.0/((W.sum(axis=1)).flatten()), 0, W.shape[0], W.shape[1])
    D =spdiags(W.sum(axis=1).flatten(), 0, W.shape[0], W.shape[1])
    lam = 10
    lhs = D + lam * (D - W)
    from scipy.sparse import eye
    
    from scipy.sparse.linalg import spsolve,lsmr
    diffused_ratio = spsolve(lhs, D.dot(np.array(init_ratio)))
    
    diffused_ratios = []
    
    count = 0
    for i in range(len(inratios)):
        diffused_ratios.append(diffused_ratio[count:len(inratios[i][0])+count])
        count += len(inratios[i][0])
            
    savemat('diffused_%s.mat' % name, {'diffused_inratio':diffused_ratios})

    u = np.zeros(len(paths))

    h,w = imgs[0].shape
    s = (h,w,len(imgs))
    inratio_image = np.zeros(s)
    diffused_image = np.zeros(s)
    
    for (i,id) in enumerate(paths.keys()):
        frame = paths[id].frame
        rows = paths[id].rows
        cols = paths[id].cols
    
        unique_frame = np.unique(frame)
    
        for f in unique_frame:
            r = rows[frame == f]
            c = cols[frame == f]
            inratio_image[r,c,f] = inratios[f][0][segs[f][r[0],c[0]]][0]
            diffused_image[r,c,f] = diffused_ratio[id2index[f][segs[f][r[0],c[0]]]]

    return diffused_ratios, diffused_image                                        
