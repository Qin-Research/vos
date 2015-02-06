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
from skimage.morphology import *
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
        
def path_neighbors(sp_label, n_paths, mapping, mapping2, edges, flow_edges,paths):
#    adj = np.zeros((n_paths, n_paths), np.bool)
    row_index = []
    col_index = []
    edge_values =[]

    count =[]
    
    n_frames = sp_label.shape[2]

    edge_dists = []
    flow_edge_dists = []
    color_dists = []
    flow_dists = []
    edge_len = []
    n_overlap = []

    from collections import defaultdict
    for i in range(n_paths):
        edge_dists.append(defaultdict(float))
        flow_edge_dists.append(defaultdict(float))
        color_dists.append(defaultdict(float))
        flow_dists.append(defaultdict(float))
        edge_len.append(defaultdict(int))
        n_overlap.append(defaultdict(int))
        
    for k in range(n_frames):
        edge_dists_buf = []
        flow_edge_dists_buf = []
        color_dists_buf = []
        flow_dists_buf = []
        edge_length = []
        adj_list = []
    
        for i in range(n_paths):
            edge_dists_buf.append(defaultdict(float))
            flow_edge_dists_buf.append(defaultdict(float))
            color_dists_buf.append(defaultdict(float))
            flow_dists_buf.append(defaultdict(float))
            edge_length.append(defaultdict(int))
            adj_list.append(set())

            
        for i in range(sp_label.shape[0]):
           for j in range(sp_label.shape[1]):
               l = sp_label[i,j,k]
               e = edges[i,j,k]
               flow_e = flow_edges[i,j,k]
               if not mapping.has_key(l):continue
                                            
               index = mapping[l]
               if i > 0:
                   ll = sp_label[i-1,j,k]
                   if l != ll:

                       if not mapping.has_key(ll):continue                       
                       edge_dists_buf[index][mapping[ll]] += edges[i-1,j,k] + e
                       flow_edge_dists_buf[index][mapping[ll]] += flow_edges[i-1,j,k] + flow_e
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
                       flow_edge_dists_buf[index][mapping[ll]] += flow_edges[i-1,j,k] + flow_e                       
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
                       flow_edge_dists_buf[index][mapping[ll]] += flow_edges[i-1,j,k] + flow_e                       
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
                       flow_edge_dists_buf[index][mapping[ll]] += flow_edges[i-1,j,k] + flow_e                       
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
                if flow_edge_dists[i][a] <= flow_edge_dists_buf[i][a]:
                    flow_edge_dists[i][a] =flow_edge_dists_buf[i][a]
                    
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
    return flow_dists, edge_dists, flow_edge_dists, color_dists, edge_len, n_overlap
                       
def path_unary(frames, segs, sp_label, sp_unary, mappings, paths,forest,forest2):
    n_paths = len(paths)

    n_frames = len(frames)-1

    id_mapping = {}
    for (i,id) in enumerate(paths.keys()):
        id_mapping[id] = i

    
    mapping = {}
    count = 0
    ims = []
    for i in range(len(segs)):
        im = img_as_ubyte(imread(frames[i]))
        ims.append(im)
        uni = np.unique(segs[i])
        for j in uni:
            mapping[(i,j)] = count
            count += 1
        
    count = 0
    rgb_data = np.zeros((sp_unary.shape[0],3))

    for (i,id) in enumerate(paths.keys()):
        frame = paths[id].frame
        rows = paths[id].rows
        cols = paths[id].cols
    
        unique_frame = np.unique(frame)
    
        values = []
        for f in unique_frame:
            index = mapping[(f,segs[f][rows[frame == f][0], cols[frame == f][0]])]

            rgb_data[index] = np.mean(ims[f][rows[frame == f],cols[frame == f]], axis=0)
            
    prob = -np.log(forest.predict_proba(rgb_data) + 1e-7)
    prob2 = -np.log(forest2.predict_proba(rgb_data) + 1e-7)
    
        
    count = 0
    unary = np.zeros((n_paths, 2))    
    unary_forest = np.zeros((n_paths, 2))    
    unary_forest2 = np.zeros((n_paths, 2))    
    for i in range(n_frames):
        uni = np.unique(segs[i])

        for u in uni:
            orig_id = mappings[i][:u]

            if not id_mapping.has_key(orig_id):
                count += 1
                continue
            p_id = id_mapping[orig_id]
            u_fg = sp_unary[count][0]
            u_bg = sp_unary[count][1]

#           unary[p_id][0] = max(unary[p_id][0], u_fg)
#           unary[p_id][1] = max(unary[p_id][1], u_bg)
            unary[p_id][0] += u_fg
            unary[p_id][1] += u_bg
            unary_forest[p_id][0] += prob[count][0]
            unary_forest[p_id][1] += prob[count][1]
            unary_forest2[p_id][0] += prob2[count][0]
            unary_forest2[p_id][1] += prob2[count][1]

            count += 1
            
    for (i,id) in enumerate(paths.keys()):
        unary[i] /= paths[id].n_frames
        unary_forest[i] /= paths[id].n_frames
        unary_forest2[i] /= paths[id].n_frames

    return unary, unary_forest,unary_forest2

def plot_job(path1, path2, v, r,c,n_frame):
    
    aff = np.zeros((r,c,n_frame))
    rows1 = path1.rows
    cols1 = path1.cols
    frame1 = path1.frame

    rows2 = path2.rows
    cols2 = path2.cols
    frame2 = path2.frame

    overlap_frame = np.intersect1d(np.unique(frame1), np.unique(frame2))

    for f in overlap_frame:

        rws1 = rows1[frame1 == f]
        cls1 = cols1[frame1 == f]

        mask1 = np.zeros((r,c))
        mask1[rws1, cls1] = 1
        mask1 = binary_dilation(mask1, square(3))

        rws2 = rows2[frame2 == f]
        cls2 = cols2[frame2 == f]
        mask2 = np.zeros((r,c))
        mask2[rws2, cls2] = 1
        
        row_index,col_index = np.nonzero(np.logical_and(mask1, mask2))
        
        aff[row_index, col_index, f] = v

    return aff
        
def plot_affinity(p1, p2, affinity, frames, sp_label, paths, id_mapping, id_mapping2,n_jobs=1):

    r,c,n_frame = sp_label.shape    
        
    affs = Parallel(n_jobs= n_jobs)(delayed(plot_job)(paths[id_mapping2[i]], paths[id_mapping2[j]], v, r, c, n_frame) for (iter,(i,j,v)) in zip(p1, p2, affinity))

    for aff in affs[1:]:
        affs[0] += aff

    aff = affs[0]

    for i in range(sp_label.shape[2]):

        figure(figsize(21,18))
        im = imread(frames[i])

        subplot(1,2,1)
        imshow(im)

        subplot(1,2,2)
        imshow(aff[:,:,i])

        show()

def plot_affinity2(affinity, frames, sp_label, paths, id_mapping, id_mapping2):

    r,c,n_frame = sp_label.shape
    aff = np.ones((r,c,n_frame)) * inf

    for k in range(n_frame):
        for j in range(c):
            for i in range(r):
               l = sp_label[i,j,k]
               index = id_mapping[l]
               
               if i > 0:
                   ll = sp_label[i-1,j,k]
                   if l != ll:
                       aff[i,j,k] = affinity[index][id_mapping[ll]]

               if i < sp_label.shape[0]-1:
                   ll = sp_label[i+1,j,k]

                   if l != ll:
                       aff[i,j,k] = affinity[index][id_mapping[ll]]                       
                       
               if j > 0:
                   ll = sp_label[i,j-1,k]

                   if l != ll:
                       aff[i,j,k] = affinity[index][id_mapping[ll]]                       
                                              
               if j < sp_label.shape[1] -1:
                   ll = sp_label[i,j+1,k]

                   if l != ll:
                       aff[i,j,k] = affinity[index][id_mapping[ll]]                       



    for i in range(sp_label.shape[2]):

        figure(figsize(21,18))
        im = imread(frames[i])

        subplot(1,2,1)
        imshow(im)
        axis("off")

        subplot(1,2,2)
        imshow(aff[:,:,i],jet())
        axis("off")
        colorbar()

        show()
                                       
    return aff
          
# def path_unary2(frames, segs, sp_label, sp_unary, mappings, paths,forest):
#     n_paths = len(paths)
#     unary = np.zeros((n_paths, 2))
#     new_unary = np.zeros((n_paths, 2))
#     n_frames = len(frames)-1

#     id_mapping = {}
#     for (i,id) in enumerate(paths.keys()):
#         id_mapping[id] = i

#     count = 0
#     rgb_data = np.zeros((sp_unary.shape[0],3))
#     for i in range(n_frames):
#         uni = np.unique(segs[i])

#         im = img_as_ubyte(imread(frames[i]))
#         for u in uni:
#             rgb_data[count] = np.mean(im[segs[i] == u],axis=0)
#             count += 1
            
#     prob = -np.log(forest.predict_proba(rgb_data) + 1e-7)            
#     count = 0            
#     for i in range(n_frames):
#         uni = np.unique(segs[i])

#         im = img_as_ubyte(imread(frames[i]))
#         for u in uni:
            
#             orig_id = mappings[i][:u]

#             p_id = id_mapping[orig_id]
#             u_fg = sp_unary[count][0]
#             u_bg = sp_unary[count][1]

# #           unary[p_id][0] = max(unary[p_id][0], u_fg)
# #           unary[p_id][1] = max(unary[p_id][1], u_bg)
#             unary[p_id][0] += u_fg
#             unary[p_id][1] += u_bg

#             new_unary[p_id] += prob[count]
#             count += 1
            
#     # for (i,id) in enumerate(paths.keys()):
#     #     unary[i] /= paths[id].n_frames

#     return unary, new_unary

def segment(frames, unary,source, target, value, segs, potts_weight,paths):

    os.system("matlab -nodisplay -nojvm -nosplash < /home/masa/research/LSA/optimize.m");
    labels = loadmat('labeling.mat')['labels']
    count = 0
    mask = []
    r,c = segs[0].shape
    mask_label = np.ones((r,c,len(segs))) * 0.5

    for (i,path) in enumerate(paths.values()):
        if labels[i][0] == 0:
            mask_label[path.rows, path.cols, path.frame] = 1
        else:
            mask_label[path.rows, path.cols, path.frame] = 0            
            
    for j in range(len(segs)):
        mask.append(mask_label[:,:,j])
    
    return mask,labels

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

def plot_affinities(frames, affs):
    n = len(affs)
    n_frames = affs[0].shape[2]

    for i in range(n_frames):
        im = imread(frames[i])
        subplot(1,n+1,1)
        imshow(im)
        for j in range(n):
            subplot(1,n+1,j+2)
            imshow(affs[j][:,:,i],jet())
            axis('off')

        show()

        
name = 'bmx'
#name = 'hummingbird'

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
edges = loadmat('/home/masa/research/release/%s.mat' % name)['edge']
flow_edges = loadmat('/home/masa/research/FastVideoSegment/bmaps_%s.mat' % name)['boundaryMaps']
    
from skimage.filter import vsobel,hsobel
    
import cPickle 
with open('paths_%s.pickle' % name) as f:
    paths = cPickle.load(f)

diffused_image = load("diffused_image_%s.npy" % name)
data = []
all_data = []
lbl = []
ims = []
for f in frames: ims.append(img_as_ubyte(imread(f))        )

bin_edges = []    
for i in range(diffused_image.shape[2]):
    hist, bin_edge = np.histogram(diffused_image[:,:,i].flatten(), bins = 20)
    bin_edges.append(bin_edge)

fg_image = np.zeros(diffused_image.shape)        
bg_image = np.zeros(diffused_image.shape)        
for (i,id) in enumerate(paths.keys()):
    frame = paths[id].frame
    rows = paths[id].rows
    cols = paths[id].cols
    
    unique_frame = np.unique(frame)

    for (j,f) in enumerate(unique_frame):
        im = ims[f]
        mean_rgb = np.mean(im[rows[frame == f], cols[frame == f]], axis=0)

        all_data.append(mean_rgb)
        inratio = diffused_image[rows[frame == f][0], cols[frame == f][0], f]
        if inratio > bin_edges[f][5]:
            fg_image[rows[frame == f], cols[frame == f], f] = 1, 
            data.append(mean_rgb)
            lbl.append(0)
        elif inratio < bin_edges[f][1]:
            bg_image[rows[frame == f], cols[frame == f], f] = 1,             
            data.append(mean_rgb)
            lbl.append(1)
    
data = np.vstack(data)
labels = np.array(lbl)        

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(20)
forest.fit(data,labels)

long_paths = {}
len_thres = 5
loc_long = {}
color_long = {}
for (i,id) in enumerate(paths.keys()):
    if paths[id].n_frames >= 5:
        long_paths[id] = paths[id]

n_paths = len(long_paths)

id_mapping = {}
id_mapping2 = {}

for (i,id) in enumerate(long_paths.keys()):
    id_mapping[id] = i
    id_mapping2[i] = id
    
flow_dists, edge_dists, flow_edge_dists,color_dists,edge_length,n_overlap  = path_neighbors(sp_label, n_paths, id_mapping, id_mapping2, edges, flow_edges,long_paths)
color = loadmat('/home/masa/research/FastVideoSegment/color_%s.mat'% name)['prob']
locprior = loadmat('/home/masa/research/FastVideoSegment/locprior_%s.mat'% name)['locationUnaries']
loc_unary = -np.log(locprior+1e-7)
p_u, p_u_forest,_ = path_unary(frames, segs, sp_label, loc_unary, mappings, long_paths,forest,forest)
loc_weight = 5
row_index = []
col_index = []
color = []
edge = []
flow_edge = []
flow = []
overlaps = []

for i in range(n_paths):
    for a in edge_dists[i].keys():
        row_index.append(i)
        col_index.append(a)
        color.append(color_dists[i][a])
        edge.append(edge_dists[i][a] / edge_length[i][a])
        flow_edge.append(flow_edge_dists[i][a] / edge_length[i][a])
        flow.append(flow_dists[i][a])
        overlaps.append(n_overlap[i][a])

def func(x,lam): return 2*np.exp(-lam*x) - 1
lam_c = 0.05    
lam_flow = 0.05    
lam_edge = 3

lam_c = 0.00003
lam_flow = 0.05    
lam_edge = 2

color_affinity = func(np.array(color), lam_c )
flow_affinity = func(np.array(flow),lam_flow )
edge_affinity = func(np.array(edge),lam_edge )

w_e = 2
w_c = 0
w_f = 1
affinity = w_e * edge_affinity + w_c * color_affinity + w_f * flow_affinity
aff_weighted = affinity * np.array(overlaps)
#affinity *= np.array(overlaps)

potts_weight = 50

source = []
target = []
aff = []
#for (r,c,a) in zip(row_index, col_index, aff_weighted):
for (r,c,a) in zip(row_index, col_index, affinity):
    if r != c:
        source.append(r)
        target.append(c)
        aff.append(a)

PE = np.zeros((len(source), 6))
potts_weight = 0.5
PE[:,0] = np.array(target)+1
PE[:,1] = np.array(source)+1
PE[:,3] = np.array(aff)* potts_weight
PE[:,4] = np.array(aff)* potts_weight

loc_weight = 1
color_weight = 0

# p_u = loc_weight * u_loc + color_weight * u_color
# p_u_long = -loc_weight * np.log(ls) - color_weight * np.log(cs)

from scipy.io import savemat

unary = loc_weight * p_u + p_u_forest
#unary = loc_weight * p_u
savemat('energy.mat', {'UE': unary.transpose(), 'PE':PE})

mask,labels =  segment(frames, p_u, source, target, aff, segs, 0.01,long_paths)

for i in range(len(mask)):
    figure(figsize(21,18))

    print i
    im = img_as_ubyte(imread(frames[i]))            
    subplot(1,3,1)
    imshow(im)
    axis("off")
    
    subplot(1,3,2)
    imshow(alpha_composite(im, mask_to_rgb(mask[i], (0,255,0))),cmap=gray())        
    axis("off")

    subplot(1,3,3)
    imshow(mask[i],gray())
    axis("off")

    show() 


#############################################################################################
    
data = []
lbl = []

ims = []
for f in frames: ims.append(img_as_ubyte(imread(f))        )

for (i,id) in enumerate(long_paths.keys()):
    frame = long_paths[id].frame
    rows = long_paths[id].rows
    cols = long_paths[id].cols
    
    unique_frame = np.unique(frame)
    mean_rgbs = np.zeros((len(unique_frame),3))
    for (j,f) in enumerate(unique_frame):
        im = ims[f]
        mean_rgbs[j] = np.mean(im[rows[frame == f], cols[frame == f]], axis=0)

    data.append(mean_rgbs)        
    if labels[i] == 0:
        lbl.append(np.zeros(len(unique_frame)))
    else:
        lbl.append(np.ones(len(unique_frame)))

data = np.vstack(data)
labels = np.concatenate(lbl)        

from sklearn.ensemble import RandomForestClassifier
forest2 = RandomForestClassifier(20)
forest2.fit(data,labels)

id_mapping = {}
id_mapping2 = {}
for (i,id) in enumerate(paths.keys()):
    id_mapping[id] = i
    id_mapping2[i] = id

n_paths = len(paths)    
flow_dists, edge_dists, flow_edge_dists,color_dists,edge_length,n_overlap  = path_neighbors(sp_label, n_paths, id_mapping, id_mapping2, edges,flow_edges, paths)



row_index = []
col_index = []
color = []
edge = []
flow_edge = []
flow = []
overlaps = []

for i in range(n_paths):
    for a in edge_dists[i].keys():
        row_index.append(i)
        col_index.append(a)
        color.append(color_dists[i][a])
        edge.append(edge_dists[i][a] / edge_length[i][a])
        flow_edge.append(flow_edge_dists[i][a] / edge_length[i][a])
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
flow_edge_affinity = func(np.array(flow_edge),10)
edge_affinity = func(np.array(edge),lam_edge )
w_e = 2
w_c = 0
w_f = 1
affinity = w_e * edge_affinity + w_c * color_affinity + w_f * flow_edge_affinity

aff_weighted = affinity *np.array(overlaps)*0.2
#affinity *= np.array(overlaps)

potts_weight = 1

source = []
target = []
aff = []
aff2 = []

for (r,c,a,a2) in zip(row_index, col_index, affinity, aff_weighted):
    if r != c:
        source.append(r)
        target.append(c)
        aff.append(a)
        aff2.append(a2)

aff_dict = []
aff_dict2 = []
for i in range(n_paths):
    aff_dict.append({})
    aff_dict2.append({})

for (s,t,a,a2) in zip(source, target, aff,aff2):
    aff_dict[s][t] = a   
    aff_dict2[s][t] = a2   

#aff_vis = plot_affinity2(aff_dict, frames, sp_label, paths, id_mapping, id_mapping2)
new_p_u, p_u_forest, new_p_u_forest = path_unary(frames, segs, sp_label, loc_unary, mappings, paths,forest, forest2)
            
PE = np.zeros((len(source), 6))
potts_weight = 0.5
PE[:,0] = np.array(target)+1
PE[:,1] = np.array(source)+1
PE[:,3] = np.array(aff)* potts_weight
PE[:,4] = np.array(aff)* potts_weight

loc_weight = 0.5
u = loc_weight * new_p_u + 2* new_p_u_forest + 2*p_u_forest
#u = loc_weight * new_p_u
#u = new_p_u_forest
savemat('energy.mat', {'UE':u.transpose(), 'PE':PE})


new_mask, labeling = segment(frames, u, source, target, aff, segs, 0.01,paths)

n = len(new_mask)
r,c = new_mask[0].shape
m = np.zeros((r,c,n))
#new_mask =  segment(frames, u, source, target, aff, segs, 0.01,paths)

for i in range(len(new_mask)):
    m[:,:,i] = new_mask[i]
    figure(figsize(21,18))

    print i
    im = img_as_ubyte(imread(frames[i]))            
    subplot(1,4,1)
    imshow(im)
    axis("off")
    
    subplot(1,4,2)
    imshow(alpha_composite(im, mask_to_rgb(mask[i], (0,255,0))),cmap=gray())        
    axis("off")

    subplot(1,4,3)
    imshow(mask[i],gray())
    axis("off")

    subplot(1,4,4)
    imshow(alpha_composite(im, mask_to_rgb(new_mask[i], (0,255,0))),cmap=gray())        
    axis("off")
        
    show() 
    
#################################################################################
# AP 0.76

save("%s_mask.npy" % name, m)
gt = get_segtrack_gt(name)
g = gt[0]
if len(gt) > 1:
    for i in range(1,len(gt)):
        for j in range(len(gt[i])):
            g[j] += gt[i][j]


res = []
for i in range(len(g)-1):
    print i
    figure(figsize(15,12))
    result = np.ones((r,c,3), dtype=ubyte) * 125
    rs,cs = np.nonzero(new_mask[i] == 1)
#    rs,cs = np.nonzero(g[i] == 1)
    result[rs,cs] = imgs[i][rs,cs]
    res.append(np.hstack((imgs[i],result)))
    # subplot(1,3,1)
    # imshow(imgs[i])
    # axis("off")
    # subplot(1,3,2)
    # imshow(mask[i])
    # axis("off")
    # subplot(1,3,3)
    # imshow(result)
    # axis("off")    
    # show()

                         
