from pylab import *
import numpy as np
from sys import argv
from time import time
import os
from skimage import img_as_ubyte
from scipy.io import loadmat
from sklearn.preprocessing import scale
from skimage.color import rgb2gray,rgb2lab
from skimage.feature import hog
from IPython.core.pylabtools import figsize
from util import *
from path import Path
from scipy.sparse import csr_matrix, spdiags
from scipy.io import savemat

def struct_edge_detect(name):
    savemat('name.mat', {'name':name})
    os.system("matlab -nodisplay -nojvm -nosplash < matlab_func/edge_dir.m");
    
    edges = loadmat('edges.mat')['edges']
    return edges

def compute_flow_edge(name):
    savemat('name.mat', {'name':name})
    os.system("matlab -nodisplay -nojvm -nosplash < matlab_func/flow_edge.m");
    
    edges = loadmat('flow_edges.mat')['boundaryMaps']
    return edges

def compute_inprob(name,segs):
    savemat('name.mat', {'name':name})
    n = len(segs)
    r,c = segs[0].shape
    sp = np.zeros((r,c,n),dtype=np.int)
    for i in range(n): sp[:,:,i] = segs[i]

    savemat('sp.mat', {'superpixels':sp})
    os.system("matlab -nodisplay -nojvm -nosplash < matlab_func/inprobs.m");
    
    inprobs = loadmat('inprobs.mat')['inRatios']
    return inprobs

def compute_locprior(name, segs, diffused_prob):
    savemat('name.mat', {'name':name})
    n = len(segs)
    r,c = segs[0].shape
    sp = np.zeros((r,c,n),dtype=np.int)
    for i in range(n): sp[:,:,i] = segs[i]

    savemat('sp.mat', {'superpixels':sp})
    savemat('diffused.mat', {'diffused_inprobs':diffused_prob})
    os.system("matlab -nodisplay -nojvm -nosplash < matlab_func/compute_locprior.m");
    
    locprior = loadmat('locprior.mat')['locationUnaries']
    return locprior
    
    
def plot_paths_value(paths, sp_label, values,cm):

    val = np.zeros(sp_label.shape)
    for (i,id) in enumerate(paths.keys()):
        val[paths[id].rows, paths[id].cols, paths[id].frame] = values[i]

    for i in range(sp_label.shape[2]):
        imshow(val[:,:,i], cm)
        show()

    return val


def path_neighbors(sp_label, n_paths, id2ind, ind2id, edges, flow_edges,paths):
    row_index = []
    col_index = []
    edge_values =[]

    count =[]
    
    n_frames = sp_label.shape[2] - 1

    edge_dists = []
    flow_edge_dists = []
    edge_len = []

    from collections import defaultdict
    for i in range(n_paths):
        edge_dists.append(defaultdict(float))
        flow_edge_dists.append(defaultdict(float))
        edge_len.append(defaultdict(int))
        
    for k in range(n_frames):
        edge_dists_buf = []
        flow_edge_dists_buf = []
        edge_length = []
        adj_list = []
    
        for i in range(n_paths):
            edge_dists_buf.append(defaultdict(float))
            flow_edge_dists_buf.append(defaultdict(float))
            edge_length.append(defaultdict(int))
            adj_list.append(set())

        for i in range(sp_label.shape[0]):
           for j in range(sp_label.shape[1]):
               l = sp_label[i,j,k]
               e = edges[i,j,k]
               flow_e = flow_edges[i,j,k]
               if not id2ind.has_key(l): continue
                                            
               index = id2ind[l]
               if i > 0:
                   ll = sp_label[i-1,j,k]
                   if l != ll:

                       if not id2ind.has_key(ll):continue                       
                       edge_dists_buf[index][id2ind[ll]] += edges[i-1,j,k] + e
                       flow_edge_dists_buf[index][id2ind[ll]] += flow_edges[i-1,j,k] + flow_e
                       edge_length[index][id2ind[ll]] += 1
                       adj_list[index].add(id2ind[ll])

                       
               if i < sp_label.shape[0]-1:
                   ll = sp_label[i+1,j,k]

                   if l != ll:
                       if not id2ind.has_key(ll):continue                       
                       edge_dists_buf[index][id2ind[ll]] += edges[i+1,j,k] + e
                       flow_edge_dists_buf[index][id2ind[ll]] += flow_edges[i-1,j,k] + flow_e                       
                       edge_length[index][id2ind[ll]] += 1
                       adj_list[index].add(id2ind[ll])
                       
               if j > 0:
                   ll = sp_label[i,j-1,k]

                   if l != ll:
                       if not id2ind.has_key(ll):continue                       
                       edge_dists_buf[index][id2ind[ll]] += edges[i,j-1,k] + e
                       flow_edge_dists_buf[index][id2ind[ll]] += flow_edges[i-1,j,k] + flow_e                       
                       edge_length[index][id2ind[ll]] += 1
                       adj_list[index].add(id2ind[ll])
                   
               if j < sp_label.shape[1] -1:
                   ll = sp_label[i,j+1,k]

                   if l != ll:
                       if not id2ind.has_key(ll):continue                       
                       edge_dists_buf[index][id2ind[ll]] += edges[i,j+1,k] + e
                       flow_edge_dists_buf[index][id2ind[ll]] += flow_edges[i-1,j,k] + flow_e                       
                       edge_length[index][id2ind[ll]] += 1
                       adj_list[index].add(id2ind[ll])

        for i in range(n_paths):
            if len(adj_list[i]) == 0: continue
            p1 = paths[ind2id[i]]
            f_index1 = np.nonzero(np.unique(p1.frame) == k)[0][0]
            for a in adj_list[i]:
                
                p2 = paths[ind2id[a]]
                f_index2 = np.nonzero(np.unique(p2.frame) == k)[0][0]
                
                if edge_dists[i][a] <= edge_dists_buf[i][a]:
                    edge_dists[i][a] =edge_dists_buf[i][a]
                if flow_edge_dists[i][a] <= flow_edge_dists_buf[i][a]:
                    flow_edge_dists[i][a] =flow_edge_dists_buf[i][a]
                    
                    edge_len[i][a] = edge_length[i][a]

    return edge_dists, flow_edge_dists, edge_len

                       
def path_unary(frames, segs, sp_unary, label_mappings, paths,initial_forest,refined_forest):
    n_paths = len(paths)

    n_frames = len(frames)-1

    id2ind = {}
    for (i,id) in enumerate(paths.keys()):
        id2ind[id] = i

    mapping = {}
    count = 0
    ims = []
    for i in range(len(segs)-1):
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
            
    prob = -np.log(initial_forest.predict_proba(rgb_data) + 1e-7)
    prob2 = -np.log(refined_forest.predict_proba(rgb_data) + 1e-7)
    
        
    count = 0
    unary = np.zeros((n_paths, 2))    
    unary_forest = np.zeros((n_paths, 2))    
    unary_forest2 = np.zeros((n_paths, 2))    
    for i in range(n_frames):
        uni = np.unique(segs[i])

        for u in uni:
            orig_id = label_mappings[i][:u]

            if not id2ind.has_key(orig_id):
                count += 1
                continue
            p_id = id2ind[orig_id]
            u_fg = sp_unary[count][0]
            u_bg = sp_unary[count][1]

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

          
def optimize_lsa(unary,pairwise, segs,paths):

    savemat('energy.mat', {'UE': unary.transpose(), 'PE':pairwise})    
    os.system("matlab -nodisplay -nojvm -nosplash < matlab_func/optimize.m");
    labels = loadmat('labeling.mat')['labels']
    count = 0
    mask = []
    r,c = segs[0].shape
    mask_label = np.ones((r,c,len(segs)-1)) * 0.5

    for (i,path) in enumerate(paths.values()):
        if labels[i][0] == 0:
            mask_label[path.rows, path.cols, path.frame] = 1
        else:
            mask_label[path.rows, path.cols, path.frame] = 0            
            
    for j in range(len(segs)-1):
        mask.append(mask_label[:,:,j])
    
    return mask,labels

### which video to segment ###
#name = 'soldier'
#name = 'bmx'
#name = 'girl'
name = 'hummingbird'

### load required precomputed data ###
data_dir = "data"
imdir = "%s/rgb/%s/" % (data_dir,name)

frames = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)) if f.endswith(".png")] 

imgs = [img_as_ubyte(imread(f)) for f in frames]
        
sp_file = "%s/tsp/%s.mat" % (data_dir,name)

#load precomputed temporal superpixels (tsp)
sp_label = loadmat(sp_file)['sp_labels']

# relabel segment labels to 0,1,2, ...
# mappings is a mapping from original superpixel label to relabeled ones and vice versa
print 'relabel segment labels...'
segs,label_mappings = relabel(sp_label)

# path here refers to each tsp trajectory
print 'load precomputed TSP trajectories...'
import cPickle 
with open("%s/trajs/%s.pickle" % (data_dir,name) ) as f:
    paths = cPickle.load(f) # see path.py


### Compute color and flow edges ###     
edges = struct_edge_detect(name) # structured forest edge detector (Dollar et al. ICCV2013)

flow_edges = compute_flow_edge(name) # flow edge


######## Diffusion ##########

#from diffusion import diffuse_inprob
print 'Diffusion...'

inprobs = compute_inprob(name, segs)

from diffusion import diffuse_inprob
diffused_prob,diffused_image = diffuse_inprob(inprobs, paths, segs,imgs)

###### Random forest ########
print 'Random Forest...'
# see my thesis, p.14
# prepare training data based on diffused prob.

mean_rgbs = []
lbl = []

bin_edges = []    
for i in range(diffused_image.shape[2]):
    hist, bin_edge = np.histogram(diffused_image[:,:,i].flatten(), bins = 20)
    bin_edges.append(bin_edge)

for (i,id) in enumerate(paths.keys()):
    frame = paths[id].frame
    rows = paths[id].rows
    cols = paths[id].cols
    
    unique_frame = np.unique(frame)

    for (j,f) in enumerate(unique_frame):
        im = imgs[f]
        mean_rgb = np.mean(im[rows[frame == f], cols[frame == f]], axis=0)

        inprob = diffused_image[rows[frame == f][0], cols[frame == f][0], f]
        if inprob > bin_edges[f][5]:
            mean_rgbs.append(mean_rgb)
            lbl.append(0)
        elif inprob < bin_edges[f][1]:
            mean_rgbs.append(mean_rgb)
            lbl.append(1)
    
mean_rgbs = np.vstack(mean_rgbs)
labels = np.array(lbl)        

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(20)
forest.fit(mean_rgbs,labels)

####### Segment long trajs #######
# Gather longer paths (more than 5 frame) and segment them first.

long_paths = {}
len_thres = 5
loc_long = {}
color_long = {}
for (i,id) in enumerate(paths.keys()):
    if paths[id].n_frames >= len_thres:
        long_paths[id] = paths[id]

n_paths = len(long_paths)

id2ind = {}
ind2id = {}

for (i,id) in enumerate(long_paths.keys()):
    id2ind[id] = i
    ind2id[i] = id

######### Unary ###########

# Compute unary potetential of paths by averaging superpixel unary

locprior = compute_locprior(name, segs, diffused_prob)
loc_unary = -np.log(locprior+1e-7)
unary_loc, unary_forest,_ = path_unary(frames, segs,loc_unary, label_mappings, long_paths,forest,forest) #second forest is dummy


######### Pairwise #######
# Compute color edge distance and flow edge distance between neighboring trajectories.
edge_dists, flow_edge_dists,edge_length  = path_neighbors(sp_label, n_paths, id2ind, ind2id, edges, flow_edges,long_paths)

row_index = []
col_index = []
edge = []
flow_edge = []

for i in range(n_paths):
    for a in edge_dists[i].keys():
        row_index.append(i)
        col_index.append(a)
        edge.append(edge_dists[i][a] / edge_length[i][a])
        flow_edge.append(flow_edge_dists[i][a] / edge_length[i][a])

def func(x,lam): return 2*np.exp(-lam*x) - 1
    
lam_flow_edge = 10
lam_edge = 2

flow_affinity = func(np.array(flow_edge),lam_flow_edge)
edge_affinity = func(np.array(edge),lam_edge )

w_e = 2
w_f = 1
affinity = w_e * edge_affinity + w_f * flow_affinity

source = []
target = []
aff = []

for (r,c,a) in zip(row_index, col_index, affinity):
    if r != c:
        source.append(r)
        target.append(c)
        aff.append(a)

PE = np.zeros((len(source), 6))

param = {"bmx":0.5, "girl":0.1, "hummingbird":1, "soldier":1}
potts_weight = param[name]
PE[:,0] = np.array(target)+1
PE[:,1] = np.array(source)+1
PE[:,3] = np.array(aff)* potts_weight
PE[:,4] = np.array(aff)* potts_weight

loc_weight = 1

unary = loc_weight * unary_loc + unary_forest

######### Optimize ##########

mask,labels =  optimize_lsa(unary, PE, segs,long_paths)

########## Reestimate unary ###########    
data = []
lbl = []

for (i,id) in enumerate(long_paths.keys()):
    frame = long_paths[id].frame
    rows = long_paths[id].rows
    cols = long_paths[id].cols
    
    unique_frame = np.unique(frame)
    mean_rgbs = np.zeros((len(unique_frame),3))
    for (j,f) in enumerate(unique_frame):
        im = imgs[f]
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

### Segment all trajectories ###

id2ind = {}
ind2id = {}
for (i,id) in enumerate(paths.keys()):
    id2ind[id] = i
    ind2id[i] = id

n_paths = len(paths)    
edge_dists, flow_edge_dists,edge_length = path_neighbors(sp_label, n_paths, id2ind, ind2id, edges,flow_edges, paths)

row_index = []
col_index = []
edge = []
flow_edge = []

for i in range(n_paths):
    for a in edge_dists[i].keys():
        row_index.append(i)
        col_index.append(a)
        edge.append(edge_dists[i][a] / edge_length[i][a])
        flow_edge.append(flow_edge_dists[i][a] / edge_length[i][a])
                    
flow_edge_affinity = func(np.array(flow_edge),lam_flow_edge)
edge_affinity = func(np.array(edge),lam_edge )

affinity = w_e * edge_affinity + w_f * flow_edge_affinity

source = []
target = []
aff = []

for (r,c,a) in zip(row_index, col_index, affinity):
    if r != c:
        source.append(r)
        target.append(c)
        aff.append(a)

unary_loc, unary_forest, unary_forest_refined = path_unary(frames, segs,loc_unary, label_mappings, paths,forest, forest2)
            
PE = np.zeros((len(source), 6))
param = {"bmx":0.5, "girl":1, "hummingbird":0.1, "soldier":0.1}
potts_weight = param[name]
PE[:,0] = np.array(target)+1
PE[:,1] = np.array(source)+1
PE[:,3] = np.array(aff)* potts_weight
PE[:,4] = np.array(aff)* potts_weight

w1 = {"bmx":0.5, "girl":0.5, "hummingbird":0.5, "soldier":0.5}
w2 = {"bmx":2, "girl":0.5, "hummingbird":0.5, "soldier":1}
w3 = {"bmx":2, "girl":1.5, "hummingbird":1.5, "soldier":2}
u = w1[name] * unary_loc + w2[name] * unary_forest + w3[name] * unary_forest_refined

new_mask,labeling =  optimize_lsa(u, PE,segs, paths)

n = len(new_mask)
r,c = new_mask[0].shape
m = np.zeros((r,c,n))

for i in range(len(new_mask)):
    m[:,:,i] = new_mask[i]
    figure(figsize(21,18))

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

gt = get_segtrack_gt(name)
g = gt[0]
if len(gt) > 1:
    for i in range(1,len(gt)):
        for j in range(len(gt[i])):
            g[j] += gt[i][j]

res = []
for i in range(len(g)-1):

    figure(figsize(15,12))
    result = np.ones((r,c,3), dtype=ubyte) * 125
    rs,cs = np.nonzero(new_mask[i] == 1)
#    rs,cs = np.nonzero(g[i] == 1)
    result[rs,cs] = imgs[i][rs,cs]
    res.append(np.hstack((imgs[i],result)))
    imshow(res[-1])
    axis('off')
    show()

print "Average precision score:", compute_ap(g, new_mask)

os.system('rm *.mat')
