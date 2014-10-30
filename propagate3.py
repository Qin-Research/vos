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
from video_util import *
from IPython.core.pylabtools import figsize
from scipy.sparse import csr_matrix

def superpixel_feature(image,seg,lab_range):
    uni = np.unique(seg)
    n = len(uni)
    dim = 0
    features = None
    lab_image = rgb2lab(image)
    gray = np.pad(rgb2gray(image), (7,7), 'symmetric')

    n_bins = 20    
    for (i,region) in enumerate(uni):
        rows, cols = np.nonzero(seg == region)
        rgbs = image[rows, cols, :]
        labs = lab_image[rows, cols,:]
        
        #feature = np.empty(0)
        feature = np.mean(rgbs, axis=0) / 13
        center_y = np.mean(rows) / 60
        center_x = np.mean(cols) / 60
        feature = np.concatenate((feature,np.array([center_y,center_x])))
                                  
#        feature = np.concatenate((feature,np.min(rgbs, axis=0)))
#       feature = np.concatenate((feature,np.max(rgbs, axis=0)))
        # for c in range(3):
        #     hist, bin_edges = np.histogram(rgbs[:,c], bins=n_bins, range=(0,256),normed=True )
        #     feature = np.concatenate((feature, hist))
        # for c in range(3):
        #      hist, bin_edges = np.histogram(labs[:,c], bins=n_bins, range=(lab_range[c,0], lab_range[c,1]))
        #      feature = np.concatenate((feature, hist))
        #center_y = round(np.mean(rows))
        #center_x = round(np.mean(cols))
        # patch = gray[center_y:center_y+15, center_x:center_x+15]
        # hog_feat = hog(patch,orientations=6,pixels_per_cell=(5,5), cells_per_block=(3,3))
        # feature = np.concatenate((feature, hog_feat))
        # feature = np.concatenate((feature, np.array([np.mean(rows)/image.shape[0], np.mean(cols)/image.shape[1]])))
 #       feature = np.concatenate((feature, np.mean(rgbs, axis=0)))
 #       feature = np.concatenate((feature, np.mean(labs, axis=0)))

        if features == None:
            dim = len(feature)
            features = np.zeros((n, dim))
            features[0] = feature
        else:
            features[i] = feature

 #   return scale(features)

    
    return (features)

def get_sp_feature_all_frames(frames, segs, lab_range):
    feats = []
    from skimage import img_as_ubyte
    for (ii,im) in enumerate(frames):
#        features = superpixel_feature((imread(im)), segs[ii], lab_range)
        features = superpixel_feature(img_as_ubyte(imread(im)), segs[ii], lab_range)
        feats.append(features)
        
    return feats

def feats2mat(feats):
    ret = feats[0]
    for feat in feats[1:]:
        ret = np.vstack((ret, feat))
    return ret

def get_feature_for_pairwise(frames, segs, adjs,lab_range):
    print 'foo'
    features = feats2mat(get_sp_feature_all_frames(frames, segs, lab_range))
    new_features = np.zeros(features.shape)

    return features

name = 'monkeydog'
#name = 'soldier'

imdir = '/home/masa/research/code/rgb/%s/' % name
vx = loadmat('/home/masa/research/code/flow/%s/vx.mat' % name)['vx']
vy = loadmat('/home/masa/research/code/flow/%s/vy.mat' % name)['vy']

frames = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)[:-1]) if f.endswith(".png")] 
from skimage.filter import vsobel,hsobel

mag = np.sqrt(vx**2 + vy ** 2)
r,c,n_frames = mag.shape
n_frames+=1
sp_file = "../TSP/results/%s.mat" % name
sp_label = loadmat(sp_file)["sp_labels"]
segs,adjs,mappings = get_tsp(sp_label)
sp_mat = np.empty((r,c,n_frames))
for i in range(n_frames):
    sp_mat[:,:,i] = segs[i]

savemat('sp_%s.mat' % name, {'superpixels':sp_mat})    
uni = np.unique(segs[0])
gt = get_segtrack_gt(name)

if len(gt) == 1:
    g = gt[0][0]
    f = []
    b = []
    
    for u in uni:
        rows, cols = np.nonzero(segs[0] == u)
        if np.mean(g[rows, cols]) > 0.6: f.append(u)
        else: b.append(u)
    
    savemat('mask_%s.mat' % name, {'fore':array(f), 'back':array(b), 'count':1})
else:    
    g1 = gt[0][0]
    g2 = gt[1][0]
    f1 = []
    b1 = []
    f2 = []
    b2 = []
    
    for u in uni:
        rows, cols = np.nonzero(segs[0] == u)
        if np.mean(g1[rows, cols]) > 0.6: f1.append(u)
        else: b1.append(u)
        if np.mean(g2[rows, cols]) > 0.6: f2.append(u)
        else: b2.append(u)
    
    savemat('mask_%s.mat' % name, {'fore':array(f1), 'back':array(b1), 'fore2':array(f2), 'back2':array(b2), 'count':2})
    
# lab_range = get_lab_range(frames)
# feats = get_sp_rgb_mean_all_frames(frames,segs, lab_range)


# if len(gt) > 1:
#     g += gt[1][0]

    
# node_id = []

# id_count = 0
# init_sal = np.load('sal_%s.npy' % name)
# rhs = []
# for i in range(2):
#     uni = np.unique(segs[i])
#     id_dict = {}
#     for u in uni:
#         rs, cs = np.nonzero(segs[i] == u)
#         if i == 0:
#             rhs.append(np.mean(g[rs,cs]))
#         else:
#             rhs.append(0)

#         id_dict[u] = id_count
#         id_count += 1
#     node_id.append(id_dict)
    
# rows = []
# cols = []
# values = []
# n_node = 0

# edges = []
# edge_cost = []
# n_temp = 0

# for i in range(2):
#     uni = np.unique(segs[i])
#     n_node += len(uni)
#     print i
#     for u in uni:
#         rs,cs = np.nonzero(segs[i] == u)

#         if i == 0 and rhs[node_id[0][u]] == 0: continue
#         for (n_id,adj) in enumerate(adjs[i][u]):
#             if adj == False: continue
#             if node_id[i][u] == node_id[i][n_id]: continue
#             if i == 0 and rhs[node_id[0][n_id]] == 0: continue            
#             rows.append(node_id[i][u])
#             cols.append(node_id[i][n_id])
# #            values.append(np.exp(-np.linalg.norm(feats[i][u] - feats[i][n_id]) ** 2 / (sigma2)))
#             values.append(np.linalg.norm(feats[i][u] - feats[i][n_id]) ** 2)
#             values.append(values[-1])
#             cols.append(node_id[i][u])
#             rows.append(node_id[i][n_id])

#             edges.append((node_id[i][u], node_id[i][n_id]))
#             edge_cost.append(values[-1])

#         if i == 0:
#             if rhs[node_id[0][u]] == 0: continue
#             if np.sum(sp_label[:,:,i+1] == mappings[i][:u]) > 0:

#                 id = node_id[i+1][mappings[i+1][mappings[i][:u]]]
#                 if node_id[i][u] == id: continue
#                 rows.append(node_id[i][u])
#                 cols.append(id)
#                 values.append(np.linalg.norm(feats[i][:u] - feats[i+1][mappings[i+1][mappings[i][:u]]]) ** 2)
#                 values.append(values[-1])
#                 cols.append(node_id[i][u])
#                 rows.append(id)

#                 edges.append((node_id[i][u], id))                
#                 edge_cost.append(values[-1])
#                 n_temp += 1

# sigma2 = 8000
# values = np.exp(-np.array(values) / sigma2)
# edge_cost = np.exp(-np.array(edge_cost) / sigma2)

# from scipy.sparse import csr_matrix, spdiags                                   
# W = csr_matrix((values, (rows, cols)), shape=(n_node, n_node))

# inv_D =spdiags(1.0/((W.sum(axis=1)).flatten()), 0, W.shape[0], W.shape[1])
# D =spdiags(W.sum(axis=1).flatten(), 0, W.shape[0], W.shape[1])
# lam = 100
# lhs = D + lam * (D - W)
# from scipy.sparse import eye

# #lhs = eye(n_node) - (inv_D.dot(W))

# from scipy.sparse.linalg import spsolve,lsmr
# sal = spsolve(lhs, D.dot(np.array(rhs)))


# # sal = np.array(rhs)
# # A = inv_D.dot(W)

# # for i in range(10000):
# #     sal = A.dot(sal)

# #sal = (sal - np.min(sal)) / (np.max(sal) - np.min(sal))        

# # from skimage import img_as_ubyte

# count = 0
# masks = []
# ims = []
# sal_images = []
# sal_image = np.zeros((r,c))
# uni = np.unique(segs[1])

# for u in uni:
#     rows, cols = np.nonzero(segs[1] == u)
#     sal_image[rows, cols] = sal[node_id[1][u]]
  
