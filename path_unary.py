from pylab import *
import numpy as np
#from util import *
from sys import argv
from time import time
import os
from skimage import img_as_ubyte
from scipy.io import loadmat,savemat
from sklearn.preprocessing import scale
from sklearn.cluster import spectral_clustering
from skimage.color import rgb2gray,rgb2lab
from skimage.feature import hog
from joblib import Parallel, delayed
from skimage.segmentation import find_boundaries
from video_graph import *
from IPython.core.pylabtools import figsize
from video_util import *
from path import Path
from scipy.sparse import csr_matrix, spdiags

def plot_value(paths, sp_label, values,cm):

    val = np.zeros(sp_label.shape)
    for (i,id) in enumerate(paths.keys()):
        val[paths[id].rows, paths[id].cols, paths[id].frame] = values[i]

    for i in range(sp_label.shape[2]):
        imshow(val[:,:,i], cm)
        show()
    return val

def path_neighbors(sp_label, n_paths, mapping, mapping2, edges):
#    adj = np.zeros((n_paths, n_paths), np.bool)
    row_index = []
    col_index = []
    edge_values =[]

    count =[]
    
    n_frames = sp_label.shape[2]

    edge_dists = []
    color_dists = []
    flow_dists = []
    edge_len = []

    from collections import defaultdict
    for i in range(n_paths):
        edge_dists.append(defaultdict(float))
        color_dists.append(defaultdict(float))
        flow_dists.append(defaultdict(float))
        edge_len.append(defaultdict(int))
        
    for k in range(n_frames):
        edge_dists_buf = []
        color_dists_buf = []
        flow_dists_buf = []
        edge_length = []
        adj_list = []
    
        for i in range(n_paths):
            edge_dists_buf.append(defaultdict(float))
            color_dists_buf.append(defaultdict(float))
            flow_dists_buf.append(defaultdict(float))
            edge_length.append(defaultdict(int))
            adj_list.append(set())

            
        for i in range(sp_label.shape[0]):
           for j in range(sp_label.shape[1]):
               l = sp_label[i,j,k]
               e = edges[i,j,k]
                         
               index = mapping[l]
               if i > 0:
                   ll = sp_label[i-1,j,k]
                   if l != ll:
                       edge_dists_buf[index][mapping[ll]] += edges[i-1,j,k] + e
                       edge_length[index][mapping[ll]] += 1
                       adj_list[index].add(mapping[ll])
                       
                       # adj[index, mapping[ll]] = 1
                       # edge_values.append(edges[i-1,j,k] + e)
                       # row_index.append(index)
                       # col_index.append(mapping[ll])
                       # count.append(1)
                                
                       
               if i < sp_label.shape[0]-1:
                   ll = sp_label[i+1,j,k]

                   if l != ll:
                       edge_dists_buf[index][mapping[ll]] += edges[i+1,j,k] + e
                       edge_length[index][mapping[ll]] += 1
                       adj_list[index].add(mapping[ll])
                   
                   # if l!=ll:
                   #     adj[index, mapping[ll]] = 1
                   #     edge_values.append(edges[i+1,j,k] + e)
                   #     row_index.append(index)
                   #     col_index.append(mapping[ll])

                   #     count.append(1)
                       
               if j > 0:
                   ll = sp_label[i,j-1,k]

                   if l != ll:
                       edge_dists_buf[index][mapping[ll]] += edges[i,j-1,k] + e
                       edge_length[index][mapping[ll]] += 1
                       adj_list[index].add(mapping[ll])
                   
                   # if l!=ll:
                   #     adj[index, mapping[ll]] = 1
                   #     edge_values.append(edges[i,j-1,k] + e)
                   #     row_index.append(index)
                   #     col_index.append(mapping[ll])
                   #     count.append(1)
                                              
               if j < sp_label.shape[1] -1:
                   ll = sp_label[i,j+1,k]

                   if l != ll:
                       edge_dists_buf[index][mapping[ll]] += edges[i,j+1,k] + e
                       edge_length[index][mapping[ll]] += 1
                       adj_list[index].add(mapping[ll])


        for i in range(n_paths):
            if len(adj_list[i]) == 0: continue
            p1 = paths[mapping2[i]]
            f_index1 = np.nonzero(np.unique(p1.frame) == k)[0][0]
            for a in adj_list[i]:
                
                p2 = paths[mapping2[a]]
                f_index2 = np.nonzero(np.unique(p2.frame) == k)[0][0]
                
                flow_dists_buf[i][a] = np.linalg.norm(p1.mean_flows[f_index1] - p2.mean_flows[f_index2])**2
                color_dists_buf[i][a] = np.linalg.norm(p1.mean_rgb[f_index1] - p2.mean_rgb[f_index2])**2

                flow_dists[i][a] = max(flow_dists[i][a],flow_dists_buf[i][a])
                color_dists[i][a] = max(color_dists[i][a],color_dists_buf[i][a])

                if edge_dists[i][a] <= edge_dists_buf[i][a]:
                    edge_dists[i][a] =edge_dists_buf[i][a]
                    edge_len[i][a] = edge_length[i][a]
                    
                   # if l!=ll:
                   #     adj[index, mapping[ll]] = 1
                   #     edge_values.append(edges[i,j+1,k] + e)
                   #     row_index.append(index)
                   #     col_index.append(mapping[ll])

                   #     count.append(1)
    return flow_dists, edge_dists, color_dists, edge_len
    # edge_strength = csr_matrix((edge_values, (row_index, col_index)), shape=(n_paths, n_paths))                       
    # count_matrix = csr_matrix((count, (row_index, col_index)), shape=(n_paths, n_paths))

    # rows, cols = count_matrix.nonzero()

    # for i in range(len(rows)):
    #     edge_strength[rows[i],cols[i]] /= count_matrix[rows[i],cols[i]]
    # return adj, edge_strength 
                       
#name = 'hummingbird'
#name = 'bmx'
name = 'soldier'

imdir = '/home/masa/research/code/rgb/%s/' % name
vx = loadmat('/home/masa/research/code/flow/%s/vx.mat' % name)['vx']
vy = loadmat('/home/masa/research/code/flow/%s/vy.mat' % name)['vy']

mag = np.sqrt(vx**2 + vy ** 2)
angle = np.arctan2(vx,vy)
from skimage.filter import vsobel,hsobel

frames = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)) if f.endswith(".png")]
imgs = [img_as_ubyte(imread(f)) for f in frames]
        
sp_file = "../TSP/results2/%s.mat" % name
sp_label = loadmat(sp_file)["sp_labels"][:,:,:-1]

from skimage.filter import vsobel,hsobel

#
#edges = loadmat('/home/masa/research/release/%s.mat' % name)['edge']
    
from cPickle import load
with open('paths_%s.pickle' % name) as f:
    paths = load(f)


#segs,adjs, mapping = get_tsp2(sp_label)
segs = loadmat('sp_%s2.mat'%name)['superpixels'].astype(np.int)
s = []
for i in range(segs.shape[2]):
    s.append(segs[:,:,i])
segs = s    

inratios = loadmat('/home/masa/research/FastVideoSegment/inratios_%s.mat' % name)['inRatios']    

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

#lhs = eye(n_node) - (inv_D.dot(W))

from scipy.sparse.linalg import spsolve,lsmr
diffused_ratio = spsolve(lhs, D.dot(np.array(init_ratio)))

diffused_ratios = []

count = 0
for i in range(len(inratios)):
    diffused_ratios.append(diffused_ratio[count:len(inratios[i][0])+count])
    count += len(inratios[i][0])
        
savemat('diffused_%s.mat' % name, {'diffused_inratio':diffused_ratios})                        
# bmaps = loadmat('/home/masa/research/FastVideoSegment/bmaps_%s.mat' % name)['boundaryMaps']
# bmaps = bmaps > 0.2

# new_vx = np.zeros(vx.shape)
# new_vy = np.zeros(vy.shape)
# 
# 

# data = []
# label = []
# weight = []
# colors = []
u = np.zeros(len(paths))
inratio_image = np.zeros(sp_label.shape)
diffused_image = np.zeros(sp_label.shape)

for (i,id) in enumerate(paths.keys()):
    frame = paths[id].frame
    rows = paths[id].rows
    cols = paths[id].cols

    unique_frame = np.unique(frame)

    values = []
    # new_vx[rows, cols, frame] = np.mean(vx[rows, cols,frame])
    # new_vy[rows, cols, frame] = np.mean(vy[rows, cols,frame])

    # colors2 = []
    # ratios = []
    for f in unique_frame:
        r = rows[frame == f]
        c = cols[frame == f]
        values.append(inratios[f][0][segs[f][r[0],c[0]]][0])
        ratio =  inratios[f][0][segs[f][r[0],c[0]]][0]
        inratio_image[r,c,f] = ratio
        diffused_image[r,c,f] = diffused_ratio[id2index[f][segs[f][r[0],c[0]]]]

#         color = np.mean(imgs[f][r,c],axis=0)
#         colors.append(color)
#         colors2.append(color)
#         ratios.append(ratio)
#         # if ratio == 0:
#         #     data.append(color)                    
#         #     label.append(1)
#         #     weight.append(1)
#         # elif ratio > 0.2:
#         #     data.append(color)                    
#         #     label.append(0)
#         #     weight.append(ratio)
# #        values.append(np.sum(bmaps[r,c,f]))

    u[i] = np.mean(values)
#     if u[i] == 0:
#         for c in colors2:
#             data.append(c)
#             label.append(1)
#             weight.append(1)
#     elif u[i] > 0.2:
#         for c in colors2:
#             data.append(c)
#             label.append(0)
#             weight.append(u[i])
        
val = plot_value(paths, sp_label, u, jet())

for i in range(val.shape[2]):
    figure(figsize(21,18))
    subplot(1,4,1)
    imshow(inratio_image[:,:,i])
    subplot(1,4,2)
    imshow(val[:,:,i])
    subplot(1,4,3)
    imshow(diffused_image[:,:,i])
    subplot(1,4,4)
    imshow(diffused_image[:,:,i] + inratio_image[:,:,i])    
    show()

#loc2 = loadmat('/home/masa/research/FastVideoSegment/loc.mat')['loc']                                
# from sklearn.ensemble import RandomForestClassifier
# from sklearn import svm

# #forest = RandomForestClassifier(20)
# forest = svm.SVC(probability=True)
# forest.fit(np.array(data),label,weight)
# prob = forest.predict_proba(np.array(colors))
# forest_image = np.zeros(sp_label.shape)

# count = 0
# u_forest = np.zeros((len(paths), 2))
# for (i,id) in enumerate(paths.keys()):
#     frame = paths[id].frame
#     rows = paths[id].rows
#     cols = paths[id].cols

#     unique_frame = np.unique(frame)

#     for f in unique_frame:
#         r = rows[frame == f]
#         c = cols[frame == f]
#         forest_image[r,c,f] = prob[count,0]
#         u_forest[i] += prob[count]
#         count+=1

#     u_forest[i] /= len(unique_frame)

# np.save("forest_%s" % name, u_forest)    
                        
# new_mag = np.sqrt(new_vx ** 2 + new_vy ** 2)

# id_mapping = {}
# id_mapping2 = {}
# for (i,id) in enumerate(paths.keys()):
#     id_mapping[id] = i
#     id_mapping2[i] = id

# n_paths = len(paths)        
# flow_dists, edge_dists, color_dists,edge_length  = path_neighbors(sp_label, n_paths, id_mapping, id_mapping2, edges)                         
# row_index = []
# col_index = []
# color = []
# edge = []
# flow = []

# for i in range(n_paths):
#     row_index.append(i)
#     col_index.append(i)
#     color.append(0)
#     edge.append(0)
#     flow.append(0)
#     for a in edge_dists[i].keys():
#         row_index.append(i)
#         col_index.append(a)
#         color.append(color_dists[i][a])
#         edge.append(edge_dists[i][a] / edge_length[i][a])
#         flow.append(flow_dists[i][a])

# sigma_c = 30
# sigma_flow = 10
# sigma_edge = 0.1

# color_affinity = np.exp(-np.array(color) / (2*sigma_c**2))
# flow_affinity = np.exp(-np.array(flow) / (2*sigma_flow**2))
# edge_affinity = np.exp(-np.array(edge) / (2*sigma_edge**2))
# w_e = 0
# w_c = 10
# w_f = 0

# edge_aff = csr_matrix((edge_affinity, (row_index, col_index)), shape=(n_paths, n_paths))
# color_aff = csr_matrix((color_affinity, (row_index, col_index)), shape=(n_paths, n_paths))
# flow_aff = csr_matrix((flow_affinity, (row_index, col_index)), shape=(n_paths, n_paths))

# W = w_e * edge_aff + w_f * flow_aff + w_c * color_aff
# inv_D =spdiags(1.0/((W.sum(axis=1)).flatten()), 0, W.shape[0], W.shape[1])
# D =spdiags(W.sum(axis=1).flatten(), 0, W.shape[0], W.shape[1])
# lam = 10
# lhs = D + lam * (D - W)

# from scipy.sparse import eye

# #lhs = eye(n_node) - (inv_D.dot(W))

# from scipy.sparse.linalg import spsolve,lsmr
# u_new = spsolve(lhs, D.dot(np.array(u)))
# np.save('loc_%s.npy' % name, u_new)

# #color_prob = loadmat('/home/masa/research/FastVideoSegment/appearance_%s.mat' % name)['prob']    

# color_u = np.zeros((len(paths),2))

# count = 0

# prob_image1 = np.zeros(sp_label.shape)
# prob_image2 = np.zeros(sp_label.shape)
# from skimage.filter import vsobel,hsobel
# val = plot_value(paths, sp_label, u_new, jet())

# for i in range(val.shape[2]):
#     grad_v = vsobel(val[:,:,i])
#     grad_h = hsobel(val[:,:,i])

#     imshow(np.sqrt(grad_v ** 2 + grad_h ** 2))
#     show()

# for i in range(sp_label.shape[2]):
    
#     for u in np.unique(segs[i]):

#         rows,cols = np.nonzero(segs[i] == u)
#         prob_image1[rows,cols,i] = color_prob[count, 0]
#         prob_image2[rows,cols,i] = color_prob[count,1]
#         path_id = id_mapping[sp_label[rows[0], cols[0],i]]
#         p = color_prob[count]

#         color_u[path_id] += p
        
#         count += 1

# for (i,id) in enumerate(paths.keys()): color_u[i] /= paths[id].n_frames        
