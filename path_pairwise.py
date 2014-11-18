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

def cluster_check(paths, cluster_label,true_label):
    unique_label = np.unique(cluster_label)

    n_wrong = 0
    bad_cluster = {}
    good_cluster = {}
    for l in unique_label:
        ids = []
        for (i,id) in enumerate(paths.keys()):
            if cluster_label[i] == l:
                ids.append(i)
        ls = np.array(true_label)[ids]                
        if np.all(ls == 0) or np.all(ls == 1):
            if np.all(ls == 1): good_cluster[l] = ls
            continue
        else:
            bad_cluster[l] = ls
            n_wrong+=1

    print n_wrong
    return good_cluster, bad_cluster

        
def plot_cluster(frames, paths, sp_label, cluster_label, n):

    ids = []
    for (i,id) in enumerate(paths.keys()):
        if cluster_label[i] == n:
            ids.append(id)

    print ids
    mask = np.zeros(sp_label.shape, np.bool)
    for id in ids:
        mask[paths[id].rows, paths[id].cols, paths[id].frame] =1

    masks = []
    for i in range(sp_label.shape[2]):
        masks.append(mask[:,:,i])

    for i in range(sp_label.shape[2]):
        if np.sum(masks[i]) == 0: continue
        figure(figsize(12,9))

        print i
        
        im = img_as_ubyte(imread(frames[i]))            
    
        imshow(alpha_composite(im, mask_to_rgb(masks[i], (0,255,0))),cmap=gray())        
        axis("off")    
    
        show() 
    
        
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
                       
                           
name = 'bmx'

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

edges = loadmat('/home/masa/research/release/%s.mat' % name)['edges']
from skimage.filter import vsobel,hsobel
    
from cPickle import load
with open('paths_%s.pickle' % name) as f:
    paths = load(f)
    
gt = get_segtrack_gt(name)
n_gt = len(gt)
gt_label = np.zeros(sp_label.shape, np.bool)
for i in range(sp_label.shape[2]):
    gt_label[:,:,i] = gt[0][i].astype(np.bool)
    if n_gt > 1: gt_label[:,:,i] += gt[1][i].astype(np.bool)
    
labels = []
gt_thres = 0.5
label_count = {}

for (i,id) in enumerate(paths.keys()):
    frame = paths[id].frame
    rows = paths[id].rows
    cols = paths[id].cols
    c = len(np.unique(frame))

    if c == 1:
        labels.append(np.mean(gt_label[rows, cols, frame[0]]) > gt_thres)
        continue

    label_count[id] = np.zeros(2)    
    unique_frame = np.unique(frame)

    for u in unique_frame:
        rs = rows[frame == u]
        cs = cols[frame == u]
        if np.mean(gt_label[rs,cs,u]) > gt_thres:
            label_count[id][0] += 1

        else:
            label_count[id][1] += 1

            
    if label_count[id][0] > label_count[id][1]:
        labels.append(1)
    else:
        labels.append(0)
    
n_paths = len(paths)

id_mapping = {}
id_mapping2 = {}
for (i,id) in enumerate(paths.keys()):
    id_mapping[id] = i
    id_mapping2[i] = id


# adj,edge_strength = path_neighbors(sp_label, n_paths, id_mapping,edges)

# color_dists = []
# #mag_dists = []
# #ang_dists = []
# row_index = []
# col_index = []
# edge_values = []
# flow_dists = []
# for (i,id) in enumerate(paths.keys()):
#     p1 = paths[id]
#     color_dists.append(0)
# #    mag_dists.append(0)
#  #   ang_dists.append(0)
#     flow_dists.append(0)
#     row_index.append(i)
#     col_index.append(i)
#     edge_values.append(0)

#     for neighbor in np.nonzero(adj[i])[0]:
#         neighbor_id = id_mapping2[neighbor]

#         p2 = paths[neighbor_id]

#         row_index.append(i)
#         col_index.append(neighbor)        
#         color_dists.append(np.linalg.norm(p1.mean_rgb - p2.mean_rgb)**2)
#         flow_dists.append(np.linalg.norm(p1.mean_flow - p2.mean_flow)**2)
# #        mag_dists.append((p1.median_mag - p2.median_mag) ** 2)
#  #       ang_dists.append((p1.median_ang - p2.median_ang) ** 2)
#         edge_values.append(edge_strength[i,neighbor])

flow_dists, edge_dists, color_dists,edge_length  = path_neighbors(sp_label, n_paths, id_mapping, id_mapping2, edges)                         
row_index = []
col_index = []
color = []
edge = []
flow = []

for i in range(n_paths):
    row_index.append(i)
    col_index.append(i)
    color.append(0)
    edge.append(0)
    flow.append(0)
    for a in edge_dists[i].keys():
        row_index.append(i)
        col_index.append(a)
        color.append(color_dists[i][a])
        edge.append(edge_dists[i][a] / edge_length[i][a])
        flow.append(flow_dists[i][a])

                    
sigma_c = 70
sigma_flow = 10
sigma_edge = 0.3

color_affinity = np.exp(-np.array(color) / (2*sigma_c**2))
flow_affinity = np.exp(-np.array(flow) / (2*sigma_flow**2))
edge_affinity = np.exp(-np.array(edge) / (2*sigma_edge**2))

color_affinity[edge > 0.8] = 0
flow_affinity[edge > 0.8] = 0
edge_affinity[edge > 0.8] = 0

w_e = 1
w_c = 5
w_f = 1
#w_m = 1
#w_a = 1
edge_index = np.hstack((np.array(row_index)[:,np.newaxis], np.array(col_index)[:,np.newaxis]))                
#affinity = w_e * edge_affinity + w_c * color_affinity + w_m * mag_affinity + w_a * ang_affinity

edge_aff = csr_matrix((edge_affinity, (row_index, col_index)), shape=(n_paths, n_paths))
color_aff = csr_matrix((color_affinity, (row_index, col_index)), shape=(n_paths, n_paths))
flow_aff = csr_matrix((flow_affinity, (row_index, col_index)), shape=(n_paths, n_paths))
affinity = w_e * edge_aff + w_f * flow_aff + w_c * color_aff

paths_per_cluster = 5
n_clusters = n_paths / paths_per_cluster
import time
t = time.time()
from sklearn.cluster import affinity_propagation
cluster_labels = spectral_clustering(affinity, n_clusters=n_clusters, eigen_solver='arpack')
print time.time() - t
#plot_value(paths, sp_label, cluster_labels,jet())

good_cluster,bad_cluster = cluster_check(paths, cluster_labels,labels)
import shelve
shelf = shelve.open('%s_cluster.shelve' %name,'n')
shelf['cluster_labels'] = cluster_labels
shelf['labels'] = labels
shelf['sp_label'] = sp_label
shelf.close()
