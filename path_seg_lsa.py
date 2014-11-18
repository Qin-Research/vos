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
    
        
# def path_neighbors(sp_label, n_paths, mapping, edges):
#     adj = np.zeros((n_paths, n_paths), np.bool)
#     row_index = []
#     col_index = []
#     edge_values =[]

#     count =[]
    
#     n_frames = sp_label.shape[2]

#     for i in range(sp_label.shape[0]):
#        for j in range(sp_label.shape[1]):
#            for k in range(n_frames):
#                l = sp_label[i,j,k]
#                e = edges[i,j,k]
                         
#                index = mapping[l]
#                if i > 0:
#                    ll = sp_label[i-1,j,k]
#                    if l!=ll:
#                        adj[index, mapping[ll]] = 1
# #                       adj[mapping[ll], index] = 1
#                        edge_values.append(edges[i-1,j,k] + e)
#                        row_index.append(index)
#                        col_index.append(mapping[ll])

#                        count.append(1)
                                
                       
#                if i < sp_label.shape[0]-1:
#                    ll = sp_label[i+1,j,k]
#                    if l!=ll:
#                        adj[index, mapping[ll]] = 1
#  #                      adj[mapping[ll], index] = 1
#                        edge_values.append(edges[i+1,j,k] + e)
#                        row_index.append(index)
#                        col_index.append(mapping[ll])

#                        count.append(1)
                       
#                if j > 0:
#                    ll = sp_label[i,j-1,k]
#                    if l!=ll:
#                        adj[index, mapping[ll]] = 1
#   #                     adj[mapping[ll], index] = 1
#                        edge_values.append(edges[i,j-1,k] + e)
#                        row_index.append(index)
#                        col_index.append(mapping[ll])

#                        count.append(1)
                                              
#                if j < sp_label.shape[1] -1:
#                    ll = sp_label[i,j+1,k]
#                    if l!=ll:
#                        adj[index, mapping[ll]] = 1
#    #                    adj[mapping[ll], index] = 1
#                        edge_values.append(edges[i,j+1,k] + e)
#                        row_index.append(index)
#                        col_index.append(mapping[ll])

#                        count.append(1)
                       
#     #            if k < n_frames-1:
#     #                ll = sp_label[i,j,k+1]
#     #                if l!=ll:
#     #                    adj[index, mapping[ll]] = 1
#     # #                   adj[mapping[ll], index] = 1
#     #                    edge_values.append(edges[i,j,k+1] + e)
#     #                    row_index.append(index)
#     #                    col_index.append(mapping[ll])

#     #                    count.append(1)

#     #            if k > 0:
#     #                ll = sp_label[i,j,k-1]
#     #                if l!=ll:
#     #                    adj[index, mapping[ll]] = 1
#     #  #                  adj[mapping[ll], index] = 1
#     #                    edge_values.append(edges[i,j,k-1] + e)
#     #                    row_index.append(index)
#     #                    col_index.append(mapping[ll])
#     #                    count.append(1)                       

#     edge_strength = csr_matrix((edge_values, (row_index, col_index)), shape=(n_paths, n_paths))                       
#     count_matrix = csr_matrix((count, (row_index, col_index)), shape=(n_paths, n_paths))

#     rows, cols = count_matrix.nonzero()

#     for i in range(len(rows)):
#         edge_strength[rows[i],cols[i]] /= count_matrix[rows[i],cols[i]]
#     return adj, edge_strength 
def path_neighbors(sp_label, n_paths, mapping, mapping2, edges, paths):
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
    n_overlap = []

    from collections import defaultdict
    for i in range(n_paths):
        edge_dists.append(defaultdict(float))
        color_dists.append(defaultdict(float))
        flow_dists.append(defaultdict(float))
        edge_len.append(defaultdict(int))
        n_overlap.append(defaultdict(int))
        
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
               if not mapping.has_key(l):continue
                                            
               index = mapping[l]
               if i > 0:
                   ll = sp_label[i-1,j,k]
                   if l != ll:

                       if not mapping.has_key(ll):continue                       
                       edge_dists_buf[index][mapping[ll]] += edges[i-1,j,k] + e
                       edge_length[index][mapping[ll]] += 1
                       if not mapping[ll] in adj_list[index]: n_overlap[mapping[l]][mapping[ll]] += 1                       
                       adj_list[index].add(mapping[ll])

                       
                       # adj[index, mapping[ll]] = 1
                       # edge_values.append(edges[i-1,j,k] + e)
                       # row_index.append(index)
                       # col_index.append(mapping[ll])
                       # count.append(1)
                                
                       
               if i < sp_label.shape[0]-1:
                   ll = sp_label[i+1,j,k]

                   if l != ll:
                       if not mapping.has_key(ll):continue                       
                       edge_dists_buf[index][mapping[ll]] += edges[i+1,j,k] + e
                       edge_length[index][mapping[ll]] += 1
                       if not mapping[ll] in adj_list[index]: n_overlap[mapping[l]][mapping[ll]] += 1                                              
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
                       if not mapping.has_key(ll):continue                       
                       edge_dists_buf[index][mapping[ll]] += edges[i,j-1,k] + e
                       edge_length[index][mapping[ll]] += 1
                       if not mapping[ll] in adj_list[index]: n_overlap[mapping[l]][mapping[ll]] += 1                                              
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
                       if not mapping.has_key(ll):continue                       
                       edge_dists_buf[index][mapping[ll]] += edges[i,j+1,k] + e
                       edge_length[index][mapping[ll]] += 1
                       if not mapping[ll] in adj_list[index]: n_overlap[mapping[l]][mapping[ll]] += 1                                              
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

                # flow_dists[i][a] += flow_dists_buf[i][a]
                # color_dists[i][a] += color_dists_buf[i][a]
                # edge_dists[i][a] += edge_dists_buf[i][a] / edge_length[i][a]

                   # if l!=ll:
                   #     adj[index, mapping[ll]] = 1
                   #     edge_values.append(edges[i,j+1,k] + e)
                   #     row_index.append(index)
                   #     col_index.append(mapping[ll])

                   #     count.append(1)
    return flow_dists, edge_dists, color_dists, edge_len, n_overlap
                       
def path_unary(frames, segs, sp_label, sp_unary, mappings, paths):
    n_paths = len(paths)
    unary = np.zeros((n_paths, 2))
    n_frames = len(frames)-1

    id_mapping = {}
    for (i,id) in enumerate(paths.keys()):
        id_mapping[id] = i

    count = 0
    for i in range(n_frames):
        uni = np.unique(segs[i])

        for u in uni:
            orig_id = mappings[i][:u]

            if not id_mapping.has_key(orig_id):
                print 'foo'
                continue
            p_id = id_mapping[orig_id]
            u_fg = sp_unary[count][0]
            u_bg = sp_unary[count][1]

#           unary[p_id][0] = max(unary[p_id][0], u_fg)
#           unary[p_id][1] = max(unary[p_id][1], u_bg)
            unary[p_id][0] += u_fg
            unary[p_id][1] += u_bg

            count += 1
            
    for (i,id) in enumerate(paths.keys()):
        unary[i] /= paths[id].n_frames

    return unary

def segment(frames, unary,source, target, value, segs, potts_weight,paths):

    os.system("matlab -nodisplay -nojvm -nosplash < /home/masa/research/LSA/optimize.m");
    labels = loadmat('labeling.mat')['labels']
    count = 0
    mask = []
    r,c = segs[0].shape
    mask_label = np.zeros((r,c,len(segs)))

    for (i,path) in enumerate(paths.values()):
        if labels[i][0] == 0:
            mask_label[path.rows, path.cols, path.frame] = 1
            
    for j in range(len(segs)):
        mask.append(mask_label[:,:,j])
    
    return mask

    import opengm

    n_nodes = unary.shape[0]

    gm = opengm.graphicalModel(np.ones(n_nodes, dtype=opengm.index_type) * 2, operator="adder")
    fids=gm.addFunctions(unary.astype(np.float32))
    vis=np.arange(0,unary.shape[0],dtype=np.uint64)
# adl unary factors at once
    gm.addFactors(fids,vis)
    
    potts = potts_weight * np.array([[0,1],
                                     [1,0]])

    import time
    t = time.time()            
    for i in range(len(source)):
        e = [source[i], target[i]]
        fid = gm.addFunction(opengm.PottsFunction([2,2], valueEqual =0, valueNotEqual=potts_weight*value[i]))
        s = np.sort(e)
        gm.addFactor(fid, [s[0], s[1]])
    print time.time() - t

    opengm.hdf5.saveGraphicalModel(gm, "model.h5", "gm")
    solve_lsa = "/home/masa/research/vos/lsa/build/solve_lsa model.h5 gm"
    os.system(solve_lsa)

    labels = np.loadtxt("label")

    from opengm.inference import GraphCut
   
    inf = GraphCut(gm)
    inf.infer()
    labels = inf.arg()


    count = 0
    mask = []
    r,c = segs[0].shape
    mask_label = np.zeros((r,c,len(segs)))

    for (i,path) in enumerate(paths.values()):
        if labels[i] == 0:
            mask_label[path.rows, path.cols, path.frame] = 1
            
    for j in range(len(segs)):
        mask.append(mask_label[:,:,j])
        # uni = np.unique(segs[j])
        # new_mask = np.zeros(segs[j].shape)
        # for u in uni:
        #     rows, cols = np.nonzero(segs[j] == u)
        #     if labels[count] == 0:
        #         new_mask[rows, cols] = 1
        #     else:
        #         new_mask[rows, cols] = 0
                
        #     count += 1
            
        # mask.append(new_mask)
                
    return mask
                           
#name = 'hummingbird'
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
segs,mappings = get_tsp(sp_label)
edges = loadmat('/home/masa/research/release/%s.mat' % name)['edges']
from skimage.filter import vsobel,hsobel
    
from cPickle import load
with open('paths_%s.pickle' % name) as f:
    paths = load(f)

long_paths = {}
len_thres = 5
for id in paths.keys():
    if paths[id].n_frames >= 5:
        long_paths[id] = paths[id]
        
long_paths = paths
n_paths = len(long_paths)

id_mapping = {}
id_mapping2 = {}
for (i,id) in enumerate(long_paths.keys()):
    id_mapping[id] = i
    id_mapping2[i] = id
    
flow_dists, edge_dists, color_dists,edge_length,n_overlap  = path_neighbors(sp_label, n_paths, id_mapping, id_mapping2, edges, long_paths)
                         
row_index = []
col_index = []
color = []
edge = []
flow = []
overlaps = []

for i in range(n_paths):
    for a in edge_dists[i].keys():
        row_index.append(i)
        col_index.append(a)
        color.append(color_dists[i][a])
        edge.append(edge_dists[i][a] / edge_length[i][a])
        flow.append(flow_dists[i][a])
        overlaps.append(n_overlap[i][a])

                    
# sigma_c = 70
# sigma_flow = 10
# sigma_edge = 0.3

# color_affinity = np.exp(-np.array(color) / (2*sigma_c**2)) 
# flow_affinity = np.exp(-np.array(flow) / (2*sigma_flow**2))
# edge_affinity = np.exp(-np.array(edge) / (2*sigma_edge**2))

def func(x,lam): return 2*np.exp(-lam*x) - 1
lam_c = 0.05    
lam_flow = 0.05    
lam_edge = 3

#color[edge > 0.8] = 1e10
#flow[edge > 0.8] = 1e10
#edge[edge > 0.8] = 1e10

lam_c = 0.00003
lam_flow = 0.05    
lam_edge = 2

color_affinity = func(np.array(color), lam_c )
flow_affinity = func(np.array(flow),lam_flow )
edge_affinity = func(np.array(edge),lam_edge )

w_e = 5
w_c = 5
w_f = 0
affinity = w_e * edge_affinity + w_c * color_affinity + w_f * flow_affinity

#affinity *= np.array(overlaps)

unary = loadmat('/home/masa/research/FastVideoSegment/unary_%s.mat'% name)['unaryPotentials']
p_u = path_unary(frames, segs, sp_label, unary, mappings, long_paths)
potts_weight = 1

source = []
target = []
aff = []
for (r,c,a) in zip(row_index, col_index, affinity):
    if r != c:
        source.append(r)
        target.append(c)
        aff.append(a)

PE = np.zeros((len(source), 6))
potts_weight = 1
PE[:,0] = np.array(target)+1
PE[:,1] = np.array(source)+1
PE[:,3] = np.array(aff)* potts_weight
PE[:,4] = np.array(aff)* potts_weight

from scipy.io import savemat

savemat('energy.mat', {'UE':p_u.transpose(), 'PE':PE})

mask =  segment(frames, p_u, source, target, aff, segs, 0.01,long_paths)

for i in range(len(mask)):
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
    
# paths_per_cluster = 5
# n_clusters = n_paths / paths_per_cluster
# import time
# t = time.time()
# cluster_labels = spectral_clustering(affinity_matrix, n_clusters=n_clusters, eigen_solver='arpack')
# print time.time() - t
# plot_value(paths, sp_label, cluster_labels,jet())

# good_cluster,bad_cluster = cluster_check(paths, cluster_labels,labels)
