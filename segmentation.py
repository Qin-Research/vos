from pylab import *
import numpy as np
from sys import argv
import sys
from time import time
import os
from skimage import img_as_ubyte
from scipy.io import loadmat,savemat
from skimage.color import rgb2gray,rgb2lab
from IPython.core.pylabtools import figsize

from diffusion import *
from util import *
from vis import *
from matlab_call import *
    
def path_neighbors(sp_label, n_paths, id2ind, ind2id, edges, flow_edges,paths):
    #Return a list of dict, where each dict is something like: {'neighbor_id1':edge_distance, ...}
    #The length of the list is n_paths, one dict for one trajectory
    
    row_index = []
    col_index = []
    edge_values =[]

    count =[]
    
    n_frames = sp_label.shape[2]

    edge_dists = []
    flow_edge_dists = []
    edge_len = []

    from collections import defaultdict
    for i in range(n_paths):
        edge_dists.append(defaultdict(float))
        flow_edge_dists.append(defaultdict(float))
        edge_len.append(defaultdict(int))
        
    for k in range(n_frames-1):
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

                #Take maximum over adjacent frames
                #See 'Segmentation of moving objects by long term video analysis', PAMI2014, Section 4. 
                if edge_dists[i][a] <= edge_dists_buf[i][a]:
                    edge_dists[i][a] =edge_dists_buf[i][a]
                if flow_edge_dists[i][a] <= flow_edge_dists_buf[i][a]:
                    flow_edge_dists[i][a] =flow_edge_dists_buf[i][a]
                    
                    edge_len[i][a] = edge_length[i][a]

    return edge_dists, flow_edge_dists, edge_len


def get_pairwise(sp_label, edges, flow_edges, paths, potts_weight, vis=False):
    #Return pairwise potential (and pairwise affinity)
    #See README of external/LSA for the shape of pairwise potential
    
    n_paths = len(paths)
    id2ind = {}
    ind2id = {}
    
    for (i,id) in enumerate(paths.keys()):
        id2ind[id] = i
        ind2id[i] = id
    
    edge_dists, flow_edge_dists,edge_length  = path_neighbors(sp_label, n_paths, id2ind, ind2id, edges, flow_edges,paths)
    
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
    
    PE[:,0] = np.array(target)+1
    PE[:,1] = np.array(source)+1
    PE[:,3] = np.array(aff)* potts_weight
    PE[:,4] = np.array(aff)* potts_weight

    if vis:            
        aff_vis = plot_affinity(aff,source, target, frames, sp_label, paths, id2ind, ind2id)
        return PE, affinity, aff_vis
    else:
        return PE, affinity

                       
def path_unary(frames, segs, loc_unary, label_mappings, paths,initial_forest,refined_forest):
    # Compute unary pott of each path by averaging superpixel unary
    # The shape of loc_unary is  '# of all superpixels' x 2
    # loc_unary[:,0] is fg unary, loc_unary[:,1] is bg unary
    
    n_paths = len(paths)

    n_frames = len(frames)

    id2ind = {}
    for (i,id) in enumerate(paths.keys()):
        id2ind[id] = i

    mapping = {}
    count = 0
    ims = []
    for i in range(len(segs)):
        im = img_as_ubyte(imread(frames[i]))[:,:,:3]
        ims.append(im)
        uni = np.unique(segs[i])
        for j in uni:
            mapping[(i,j)] = count
            count += 1
        
    count = 0
    rgb_data = np.zeros((loc_unary.shape[0],3))

    for (i,id) in enumerate(paths.keys()):
        frame = paths[id].frame
        rows = paths[id].rows
        cols = paths[id].cols
    
        unique_frame = np.unique(frame)
    
        values = []
        for f in unique_frame:
            index = mapping[(f,segs[f][rows[frame == f][0], cols[frame == f][0]])]

            rgb_data[index] = np.mean(ims[f][rows[frame == f],cols[frame == f]], axis=0)
            
    cost = -np.log(initial_forest.predict_proba(rgb_data) + 1e-7)
    cost2 = -np.log(refined_forest.predict_proba(rgb_data) + 1e-7)
    
    count = 0
    unary_loc = np.zeros((n_paths, 2))    
    unary_forest = np.zeros((n_paths, 2))    
    unary_forest_refined = np.zeros((n_paths, 2))
        
    for i in range(n_frames):
        uni = np.unique(segs[i])

        for u in uni:
            orig_id = label_mappings[i][:u]

            if not id2ind.has_key(orig_id):
                count += 1
                continue
            
            p_id = id2ind[orig_id]
            u_fg = loc_unary[count][0]
            u_bg = loc_unary[count][1]

            unary_loc[p_id][0] += u_fg
            unary_loc[p_id][1] += u_bg
            unary_forest[p_id][0] += cost[count][0]
            unary_forest[p_id][1] += cost[count][1]
            unary_forest_refined[p_id][0] += cost2[count][0]
            unary_forest_refined[p_id][1] += cost2[count][1]

            count += 1
            
    for (i,id) in enumerate(paths.keys()):
        unary_loc[i] /= paths[id].n_frames
        unary_forest[i] /= paths[id].n_frames
        unary_forest_refined[i] /= paths[id].n_frames

    return unary_loc, unary_forest,unary_forest_refined


def load_traj(name):
    import cPickle
    import os

    data_dir = 'data'
    if os.path.exists("%s/trajs/%s.pickle" % (data_dir,name)):
        with open("%s/trajs/%s.pickle" % (data_dir,name) ) as f:
            paths = cPickle.load(f) # see path.py
        
            return paths
    else:
        from path import get_paths
        print "path not found. Computing ..."
        sys.stdout.flush()
        paths = get_paths(name)
         
        from cPickle import dump

        with open('data/trajs/%s.pickle' % name, 'w') as f:
             dump(paths,f)

        return paths

if len(sys.argv) == 1:
    sys.exit("No video given.")
else:    
    name = sys.argv[1]

if len(sys.argv) == 3:
    potts_weight = float(sys.argv[2])
else:
    potts_weight = 0.5
        
print "Video name: ", name
print "Pairwise weight: ", potts_weight


### load required precomputed data ###

data_dir = "data"
imdir = "%s/rgb/%s/" % (data_dir,name)

frames = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)) if f.endswith(".png")] 

imgs = [img_as_ubyte(imread(f))[:,:,:3] for f in frames]
        
sp_file = "%s/tsp/%s.mat" % (data_dir,name)

#load precomputed temporal superpixels (tsp)
sp_label = loadmat(sp_file)['sp_labels']

# relabel segment labels to 0,1,2, ...
# mappings is a mapping from original superpixel label to relabeled ones and vice versa
print 'relabel segment labels...'
segs,label_mappings = relabel(sp_label)
sys.stdout.flush()
# path here refers to each tsp trajectory
print 'load precomputed TSP trajectories (or compute if nessesary)...'
paths = load_traj(name)

### Compute color and flow edges ###     
edges = struct_edge_detect(name) # structured forest edge detector (Dollar et al. ICCV2013)

flow_edges = compute_flow_edge(name) # flow edge

######## Diffusion ##########

#from diffusion import diffuse_inprob
print 'Diffusion...'

inprobs = compute_inprob(name, segs)

diffused_prob = diffuse_inprob(inprobs, paths, segs,imgs)

inprob_image = prob_to_image(inprobs, paths, segs) 
diffused_image = prob_to_image(diffused_prob, paths,segs ) 

for i in range(diffused_image.shape[2]):
    figure(figsize(12,9))
    subplot(1,2,1)
    imshow(inprob_image[:,:,i])
    subplot(1,2,2)
    imshow(diffused_image[:,:,i])
    show()
    
locprior = compute_locprior(name, segs, diffused_prob)
loc_unary = -np.log(locprior+1e-7)

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
forest = RandomForestClassifier(20,n_jobs=-1)
forest.fit(mean_rgbs,labels)

######### Unary ###########

####### Segment long trajs #######
# Gather longer paths (more than 5 frame) and segment them first.

long_paths = {}
len_thres = 5
loc_long = {}
color_long = {}
for (i,id) in enumerate(paths.keys()):
    if paths[id].n_frames >= len_thres:
        long_paths[id] = paths[id]

# Compute unary potetential of paths by averaging superpixel unary

unary_loc, unary_forest,_ = path_unary(frames, segs,loc_unary, label_mappings, long_paths,forest,forest) #second forest is dummy

loc_weight = 0.8

unary = loc_weight * unary_loc + unary_forest

######### Pairwise #######
# Compute color edge distance and flow edge distance between neighboring trajectories.

#param = {"bmx":0.5, "girl":0.1, "hummingbird":1, "soldier":1}
# potts_weight = param[name]
PE, affinity = get_pairwise(sp_label, edges, flow_edges, long_paths, potts_weight)
    
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
forest_refined = RandomForestClassifier(20,n_jobs=-1)
forest_refined.fit(data,labels)

unary_loc, unary_forest, unary_forest_refined = path_unary(frames, segs,loc_unary, label_mappings, paths,forest, forest_refined)
# w1 = {"bmx":0.5, "girl":0.5, "hummingbird":0.5, "soldier":0.5}
# w2 = {"bmx":2, "girl":0.5, "hummingbird":2, "soldier":3}
# w3 = {"bmx":2, "girl":1.5, "hummingbird":0.5, "soldier":0.5}
#u = w1[name] * unary_loc + w2[name] * unary_forest + w3[name] * unary_forest_refined
u = 0.5 * unary_loc + 2 * unary_forest + 1 * unary_forest_refined

plot_unary(paths, sp_label, u)
#plot_unary(paths, sp_label, unary_forest)
#plot_unary(paths, sp_label, unary_forest_refined)

# param = {"bmx":0.5, "girl":1, "hummingbird":1, "soldier":0.01}
# potts_weight = param[name]

print "Compute and plot pairwise affinity"
PE, affinity, aff_vis = get_pairwise(sp_label, edges, flow_edges, paths, potts_weight, True)

for i in range(aff_vis.shape[2]):
    imshow(aff_vis[:,:,i])
    show()
    
new_mask,labeling =  optimize_lsa(u, PE,segs, paths)

#######################################################

# for i in range(len(new_mask)):
#     m[:,:,i] = new_mask[i]
#     figure(figsize(21,18))

#     im = img_as_ubyte(imread(frames[i]))            
#     subplot(1,4,1)
#     imshow(im)
#     axis("off")
    
#     subplot(1,4,2)
#     imshow(alpha_composite(im, mask_to_rgb(mask[i], (0,255,0))),cmap=gray())        
#     axis("off")

#     subplot(1,4,3)
#     imshow(mask[i],gray())
#     axis("off")

#     subplot(1,4,4)
#     imshow(alpha_composite(im, mask_to_rgb(new_mask[i], (0,255,0))),cmap=gray())        
#     axis("off")
        
#     show() 
    
#################################################################################

gt = get_segtrack_gt(name)
g = gt[0]
if len(gt) > 1:
    for i in range(1,len(gt)):
        for j in range(len(gt[i])):
            g[j] += gt[i][j]

res = []
r,c = new_mask[0].shape

for i in range(len(g)):
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

#os.system('rm *.mat')

#bmx 0.76, girl 0.63, hummingbird 0.65, soldier 0.67, parachute 0.94, drift 0.73, birdfall 0.05, monkeydog 0.13
