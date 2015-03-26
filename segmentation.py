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

def plot_value(paths, sp_label, values,cm):

    val = np.zeros(sp_label.shape)
    for (i,id) in enumerate(paths.keys()):
        val[paths[id].rows, paths[id].cols, paths[id].frame] = values[i]

    for i in range(sp_label.shape[2]):
        imshow(val[:,:,i], cm)
        show()

    return val

def path_neighbors(sp_label, n_paths, mapping, mapping2, edges, flow_edges,paths):
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
               if not mapping.has_key(l): continue
                                            
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

                       
               if i < sp_label.shape[0]-1:
                   ll = sp_label[i+1,j,k]

                   if l != ll:
                       if not mapping.has_key(ll):continue                       
                       edge_dists_buf[index][mapping[ll]] += edges[i+1,j,k] + e
                       flow_edge_dists_buf[index][mapping[ll]] += flow_edges[i-1,j,k] + flow_e                       
                       edge_length[index][mapping[ll]] += 1
                       if not mapping[ll] in adj_list[index]: n_overlap[mapping[l]][mapping[ll]] += 1                                              
                       adj_list[index].add(mapping[ll])
                       
               if j > 0:
                   ll = sp_label[i,j-1,k]

                   if l != ll:
                       if not mapping.has_key(ll):continue                       
                       edge_dists_buf[index][mapping[ll]] += edges[i,j-1,k] + e
                       flow_edge_dists_buf[index][mapping[ll]] += flow_edges[i-1,j,k] + flow_e                       
                       edge_length[index][mapping[ll]] += 1
                       if not mapping[ll] in adj_list[index]: n_overlap[mapping[l]][mapping[ll]] += 1                                              
                       adj_list[index].add(mapping[ll])
                   
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

def plot_affinity(affinity, frames, sp_label, paths, id_mapping, id_mapping2):

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
          
def segment(frames, unary,source, target, value, segs,paths, lsa_path):

    os.system("matlab -nodisplay -nojvm -nosplash < %s/optimize.m" % lsa_path);
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

name = 'hummingbird'
data_dir = "data"
imdir = "%s/rgb/%s/" % (data_dir,name)

frames = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)) if f.endswith(".png")]
imgs = [img_as_ubyte(imread(f)) for f in frames]
        
sp_file = "%s/tsp/%s.mat" % (data_dir,name)

sp_label = loadmat(sp_file)['sp_labels'][:,:,:-1]
segs,mappings = get_tsp(sp_label)
edges = loadmat("%s/edges/%s.mat" % (data_dir,name))['edges']
flow_edges = loadmat("%s/flow_edges/%s.mat" % (data_dir,name))['boundaryMaps']
    
import cPickle 
with open("%s/trajs/%s.pickle" % (data_dir,name) ) as f:
    paths = cPickle.load(f)

######## Diffusion ##########

#from diffusion import diffuse_inprob
inratios = loadmat("%s/inprobs/%s.mat" % (data_dir,name))['inRatios']
from diffusion import diffuse_inprob
diffused_ratio,diffused_image = diffuse_inprob(inratios, paths, segs,imgs)

###### Random forest ########

data = []
all_data = []
lbl = []
ims = imgs

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
        im = ims[f]
        mean_rgb = np.mean(im[rows[frame == f], cols[frame == f]], axis=0)

        all_data.append(mean_rgb)
        inratio = diffused_image[rows[frame == f][0], cols[frame == f][0], f]
        if inratio > bin_edges[f][5]:
            data.append(mean_rgb)
            lbl.append(0)
        elif inratio < bin_edges[f][1]:
            data.append(mean_rgb)
            lbl.append(1)
    
data = np.vstack(data)
labels = np.array(lbl)        

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(20)
forest.fit(data,labels)

####### Segment long trajs #######
long_paths = {}
len_thres = 5
loc_long = {}
color_long = {}
for (i,id) in enumerate(paths.keys()):
    if paths[id].n_frames >= len_thres:
        long_paths[id] = paths[id]

n_paths = len(long_paths)

id_mapping = {}
id_mapping2 = {}

for (i,id) in enumerate(long_paths.keys()):
    id_mapping[id] = i
    id_mapping2[i] = id

######### Unary ###########
    
locprior =loadmat("%s/locprior/%s.mat" % (data_dir,name))['locationUnaries']
loc_unary = -np.log(locprior+1e-7)
p_u, p_u_forest,_ = path_unary(frames, segs, sp_label, loc_unary, mappings, long_paths,forest,forest)


######### Pairwise #######        
flow_dists, edge_dists, flow_edge_dists,color_dists,edge_length,n_overlap  = path_neighbors(sp_label, n_paths, id_mapping, id_mapping2, edges, flow_edges,long_paths)

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
potts_weight = 0.5
PE[:,0] = np.array(target)+1
PE[:,1] = np.array(source)+1
PE[:,3] = np.array(aff)* potts_weight
PE[:,4] = np.array(aff)* potts_weight

loc_weight = 1

unary = loc_weight * p_u + p_u_forest
savemat('energy.mat', {'UE': unary.transpose(), 'PE':PE})

######### Optimize ##########

lsa_path = "external/LSA/"
mask,labels =  segment(frames, p_u, source, target, aff, segs,long_paths, lsa_path)

########## Reestimate unary ###########    
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

aff_dict = []
for i in range(n_paths):
    aff_dict.append({})

for (s,t,a) in zip(source, target, aff):
    aff_dict[s][t] = a   

new_p_u, p_u_forest, new_p_u_forest = path_unary(frames, segs, sp_label, loc_unary, mappings, paths,forest, forest2)
            
PE = np.zeros((len(source), 6))
potts_weight = 0.5
PE[:,0] = np.array(target)+1
PE[:,1] = np.array(source)+1
PE[:,3] = np.array(aff)* potts_weight
PE[:,4] = np.array(aff)* potts_weight

loc_weight = 0.5
u = loc_weight * new_p_u + 2* new_p_u_forest + 2*p_u_forest

savemat('energy.mat', {'UE':u.transpose(), 'PE':PE})

new_mask, labeling = segment(frames, u, source, target, aff, segs,paths,lsa_path)

n = len(new_mask)
r,c = new_mask[0].shape
m = np.zeros((r,c,n))

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

gt = get_segtrack_gt(name)
g = gt[0]
if len(gt) > 1:
    for i in range(1,len(gt)):
        for j in range(len(gt[i])):
            g[j] += gt[i][j]


res = []
for i in range(len(g)-1):
    print i
#    figure(figsize(15,12))
    result = np.ones((r,c,3), dtype=ubyte) * 125
    rs,cs = np.nonzero(new_mask[i] == 1)
#    rs,cs = np.nonzero(g[i] == 1)
    result[rs,cs] = imgs[i][rs,cs]
    res.append(np.hstack((imgs[i],result)))

print compute_ap(g, new_mask)
