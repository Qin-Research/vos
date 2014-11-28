from pylab import *
import numpy as np
#from util import *
from sys import argv
from time import time
import os
from skimage import img_as_ubyte
from scipy.io import loadmat
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
                       
name = 'hummingbird'
#name = 'bmx'

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
edges = loadmat('/home/masa/research/release/%s.mat' % name)['edge']
    
from cPickle import load
with open('paths_%s.pickle' % name) as f:
    paths = load(f)



segs = loadmat('sp_%s2.mat'%name)['superpixels'].astype(np.int)
s = []
for i in range(segs.shape[2]):
    s.append(segs[:,:,i])
segs = s    

inratios = loadmat('/home/masa/research/FastVideoSegment/inratios_%s.mat' % name)['inRatios']    

bmaps = loadmat('/home/masa/research/FastVideoSegment/bmaps_%s.mat' % name)['boundaryMaps']
bmaps = bmaps > 0.2

new_vx = np.zeros(vx.shape)
new_vy = np.zeros(vy.shape)
u = np.zeros(len(paths))
inratio_image = np.zeros(sp_label.shape)

data = []
label = []
weight = []
colors = []
for (i,id) in enumerate(paths.keys()):
    frame = paths[id].frame
    rows = paths[id].rows
    cols = paths[id].cols

    unique_frame = np.unique(frame)

    values = []
    new_vx[rows, cols, frame] = np.mean(vx[rows, cols,frame])
    new_vy[rows, cols, frame] = np.mean(vy[rows, cols,frame])

    colors2 = []
    ratios = []
    for f in unique_frame:
        r = rows[frame == f]
        c = cols[frame == f]
        values.append(inratios[f][0][segs[f][r[0],c[0]]][0])
        ratio =  inratios[f][0][segs[f][r[0],c[0]]][0]
        inratio_image[r,c,f] = ratio

        color = np.mean(imgs[f][r,c],axis=0)
        colors.append(color)
        colors2.append(color)
        ratios.append(ratio)
        # if ratio == 0:
        #     data.append(color)                    
        #     label.append(1)
        #     weight.append(1)
        # elif ratio > 0.2:
        #     data.append(color)                    
        #     label.append(0)
        #     weight.append(ratio)
#        values.append(np.sum(bmaps[r,c,f]))

    u[i] = np.mean(values)
    if u[i] == 0:
        for c in colors2:
            data.append(c)
            label.append(1)
            weight.append(1)
    elif u[i] > 0.2:
        for c in colors2:
            data.append(c)
            label.append(0)
            weight.append(u[i])
        

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(20)
forest.fit(np.array(data),label)
prob = forest.predict_proba(np.array(colors))
forest_image = np.zeros(sp_label.shape)

count = 0
u_forest = np.zeros((len(paths), 2))
for (i,id) in enumerate(paths.keys()):
    frame = paths[id].frame
    rows = paths[id].rows
    cols = paths[id].cols

    unique_frame = np.unique(frame)

    for f in unique_frame:
        r = rows[frame == f]
        c = cols[frame == f]
        forest_image[r,c,f] = prob[count,0]
        u_forest[i] += prob[count]
        count+=1

    u_forest[i] /= len(unique_frame)

np.save("forest_%s" % name, u_forest)    

bins = 20
color_hist = []
for (i,id) in enumerate(paths.keys()):
    frame = paths[id].frame
    rows = paths[id].rows
    cols = paths[id].cols

    unique_frame = np.unique(frame)

    colors = []
    for f in unique_frame:
        r = rows[frame == f]
        c = cols[frame == f]
        colors.append(imgs[f][r,c])

    colors = np.vstack(colors)
    histo = np.empty(0)
    for c in range(3):
        h,bin_edges = np.histogram(colors[:,c], bins=bins, range = (0,256))
        hi = h / float(np.sum(h))
        histo = np.concatenate((histo, hi))
    color_hist.append(histo)

color_hist = np.array(color_hist)

from scipy.spatial.distance import pdist,squareform

dist = squareform(pdist(color_hist))

sigma = 0.5
W =  np.exp(-dist / (2*sigma**2))                                        
inv_D =diag(1.0/((W.sum(axis=1)).flatten()))
D =diag(W.sum(axis=1).flatten())
lam = 10
lhs = D + lam * (D - W)

from scipy.sparse import eye

#lhs = eye(n_node) - (inv_D.dot(W))

from scipy.linalg import solve

u_new = solve(lhs, D.dot(np.array(u)))
np.save('loc_%s.npy' % name, u_new)

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
