from pylab import *
import numpy as np
from sys import argv
from time import time
import os
from scipy.io import loadmat
from sklearn.preprocessing import scale
from skimage.color import rgb2gray,rgb2lab
from skimage.feature import hog
from skimage import img_as_ubyte
from joblib import Parallel, delayed
from skimage.segmentation import find_boundaries
from video_graph import *
from video_util import *
from IPython.core.pylabtools import figsize
from scipy.sparse import csr_matrix
from sklearn.mixture import GMM
from sklearn.ensemble import RandomForestClassifier
from krahenbuhl2013 import DenseCRF
import ipdb

def segment(frames, saliency_fg, saliency_bg, pair_features ,segs,potts_weight):
    n_nodes = pair_features.shape[0]
    prob_fg = np.zeros(n_nodes)
    prob_bg = np.zeros(n_nodes)

    count = 0
    for j in range(len(segs)-1):
        uni = np.unique(segs[j])
        for u in uni:
            rows, cols = np.nonzero(segs[j] == u)
            prob_fg[count] = np.mean(saliency_fg[rows, cols, j])
            prob_bg[count] = np.mean(saliency_bg[rows, cols, j])

            count += 1
    
    prob = np.hstack((prob_fg[:,np.newaxis], prob_bg[:, np.newaxis]))
    
    eps = 1e-7
    unary = -np.log(prob+eps).astype(np.float32)

#    potts = potts_weight * np.array([[0.5,1],
                                 #    [1,0.5]], np.float32) 
    potts = potts_weight * np.array([[0,1],
                                     [1,0]], np.float32)

    crf = DenseCRF(n_nodes, 2)

    print 'Mean field inference ...'        
    crf.set_unary_energy(unary)
    crf.add_pairwise_energy(potts, np.ascontiguousarray(pair_features.astype(np.float32)))
    
    iteration = 10

    labels = crf.map(iteration)
    print ' done.'

    count = 0
    mask = []
    for j in range(len(segs)-1):
        uni = np.unique(segs[j])

        new_mask = np.zeros(segs[j].shape)
        for u in uni:
            rows, cols = np.nonzero(segs[j] == u)
            if labels[count] == 0:
                new_mask[rows, cols] = 1
            else:
                new_mask[rows, cols] = 0
                
            count += 1
            
        mask.append(new_mask)
                
    return mask


def compare(mask1,mask2):

    figure(figsize(21,18))
    for i in range(len(mask1)):
        subplot(1,2,1)
        imshow(mask1[i])
        subplot(1,2,2)
        imshow(mask2[i])

        show()
        
def plot_prob(prob,prob2, frames,segs):
    count = 0
    for j in range(len(segs)-1):
        uni = np.unique(segs[j])

        figure(figsize = (20,18))
        prob_image1 = np.zeros(segs[j].shape)
        prob_image2 = np.zeros(segs[j].shape)
        prob_image3 = np.zeros(segs[j].shape)
        prob_image4 = np.zeros(segs[j].shape)
        
        im = img_as_ubyte(imread(frames[j]))        
        for u in uni:
            rows, cols = np.nonzero(segs[j] == u)
            prob_image1[rows, cols] = prob[count,0]
            prob_image2[rows, cols] = prob[count,1]
            prob_image3[rows, cols] = prob2[count,0]
            prob_image4[rows, cols] = prob2[count,1]
            
            count += 1

        print j
        subplot(1,5,1)
#        imshow(imread(frames[j]))
        imshow(prob_image1 + prob_image3)
        
        subplot(1,5,2)
        imshow(prob_image1)

        subplot(1,5,3)
        imshow(prob_image2)

        subplot(1,5,4)
        imshow(prob_image3)

        subplot(1,5,5)
        imshow(prob_image4)
                                             
        
        show()

def segment2(frames, saliency_fg, saliency_bg, segs, edges, edge_cost,potts_weight,n_nodes):
    import opengm

    prob_fg = np.zeros(n_nodes)
    prob_bg = np.zeros(n_nodes)

    count = 0
    for j in range(len(segs)-1):
        uni = np.unique(segs[j])
        for u in uni:
            rows, cols = np.nonzero(segs[j] == u)
            prob_fg[count] = np.mean(saliency_fg[rows, cols, j])
            prob_bg[count] = np.mean(saliency_bg[rows, cols, j])

            count += 1
    
    prob = np.hstack((prob_fg[:,np.newaxis], prob_bg[:, np.newaxis]))
    
    eps = 1e-7
    unary = -np.log(prob+eps).astype(np.float32)

    gm = opengm.graphicalModel(np.ones(n_nodes, dtype=opengm.index_type) * 2, operator="adder")
    fids=gm.addFunctions(unary)
    vis=np.arange(0,unary.shape[0],dtype=np.uint64)
# adl unary factors at once
    gm.addFactors(fids,vis)
    
    potts = potts_weight * np.array([[0,1],
                                     [1,0]])

    import time
    t = time.time()            
    for (e_id,e) in enumerate(edges):
        fid = gm.addFunction(opengm.PottsFunction([2,2], valueEqual =0, valueNotEqual=potts_weight*edge_cost[e_id]))
        s = np.sort(e)
        gm.addFactor(fid, [s[0], s[1]])
    print time.time() - t

    from opengm.inference import GraphCut
    
    inf = GraphCut(gm)
    inf.infer()
    labels = inf.arg()

    count = 0
    mask = []
    for j in range(len(segs)-1):
        uni = np.unique(segs[j])

        new_mask = np.zeros(segs[j].shape)
        for u in uni:
            rows, cols = np.nonzero(segs[j] == u)
            if labels[count] == 0:
                new_mask[rows, cols] = 1
            else:
                new_mask[rows, cols] = 0
                
            count += 1
            
        mask.append(new_mask)
                
    return mask

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
                                  
        if features == None:
            dim = len(feature)
            features = np.zeros((n, dim))
            features[0] = feature
        else:
            features[i] = feature

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

name = 'bmx'

imdir = '/home/masa/research/code/rgb/%s/' % name
vx = loadmat('/home/masa/research/code/flow/%s/vx.mat' % name)['vx']
vy = loadmat('/home/masa/research/code/flow/%s/vy.mat' % name)['vy']

frames = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)[:-1]) if f.endswith(".png")] 
from skimage.filter import vsobel,hsobel

mag = np.sqrt(vx**2 + vy ** 2)
r,c,n_frames = mag.shape
sp_file = "../TSP/results/%s.mat" % name
sp_label = loadmat(sp_file)["sp_labels"]
segs,adjs,mappings = get_tsp(sp_label)
to_save = np.zeros((r,c,n_frames),dtype=segs[0].dtype)
for i in range(n_frames):
    to_save[:,:,i] = segs[i]
np.save('segs_%s.npy' % name, to_save)

segs = np.load('segs_%s.npy' % name)
s = []
for i in range(n_frames):
    s.append(segs[:,:,i])
segs = s    

lab_range = get_lab_range(frames)
feats = get_sp_rgb_mean_all_frames(frames,segs, lab_range)

gt = get_segtrack_gt(name)
g = gt[0][0]

if len(gt)>1: g += gt[1][0]

node_id = []
id_count = 0

uni = np.unique(segs[0])
fg_colors = []
bg_colors = []
samples = []
loc = []
im = img_as_ubyte(imread(frames[0]))

for u in uni:
    rs, cs = np.nonzero(segs[0] == u)

    mean_color = np.mean(im[rs, cs],0)
    mean_vx = np.mean(vx[rs,cs,0])
    mean_vy = np.mean(vy[rs,cs,0])
    mean_x = np.mean(cs)
    mean_y = np.mean(rs)
    loc += [mean_x + mean_vx, mean_y + mean_vy]
    # if np.mean(g[rs,cs], axis=0) > 0.7 :
    #     fg_colors += [mean_color[0], mean_color[1], mean_color[2], mean_x + mean_vx, mean_y + mean_vy]
    # else:
    #     bg_colors += [mean_color[0], mean_color[1], mean_color[2], mean_x + mean_vx, mean_y + mean_vy]


    if np.mean(g[rs,cs], axis=0) > 0.7 :
        fg_colors += [mean_color[0], mean_color[1], mean_color[2]]
    else:
        bg_colors += [mean_color[0], mean_color[1], mean_color[2]]

dim = 3            
fg_samples = np.array(fg_colors).reshape(-1,dim)        
bg_samples = np.array(bg_colors).reshape(-1,dim)
loc = np.array(loc).reshape(-1,2)

from sklearn.mixture import GMM
from sklearn.ensemble import RandomForestClassifier
# fg_gmm = GMM(5)
# bg_gmm = GMM(5)

# fg_gmm.fit(fg_samples)
# bg_gmm.fit(bg_samples)
uni = np.unique(segs[1])

forest = RandomForestClassifier(20)
labels = np.zeros(fg_samples.shape[0] + bg_samples.shape[0])
labels[fg_samples.shape[0]:-1] = 1
forest.fit(np.vstack((fg_samples, bg_samples)), labels)

prob_im = np.zeros((r,c))
prob_im2 = np.zeros((r,c))
colors = []

for i in range(2):
 im = img_as_ubyte(imread(frames[i]))
 uni = np.unique(segs[i])
 for u in uni:
    rs, cs = np.nonzero(segs[i] == u)
    mean_color = np.mean(im[rs,cs], axis=0)
    mean_x = np.mean(cs)
    mean_y = np.mean(rs)
#    colors += [mean_color[0], mean_color[1], mean_color[2], mean_x, mean_y]    
    colors += [mean_color[0], mean_color[1], mean_color[2]]    

colors = np.array(colors).reshape(-1,dim)
forest_proba = forest.predict_proba(colors)
#prob_fg = np.dot(fg_gmm.predict_proba(colors), fg_gmm.weights_)
#prob_bg = np.dot(bg_gmm.predict_proba(colors), bg_gmm.weights_)

loc = loadmat('/home/masa/research/FastVideoSegment/%s_loc.mat' % name)['loc']
for u in uni:
    rs, cs = np.nonzero(segs[1] == u)
    forest_proba[u,0] += np.mean(loc[rs,cs,1])
    forest_proba[u,1] += np.mean(1 - loc[rs,cs,1])
    prob_im[rs,cs] = forest_proba[len(np.unique(segs[0])) + u,0]
    prob_im2[rs,cs] =  forest_proba[len(np.unique(segs[0])) + u,1]

figure(figsize(21,18))              
subplot(1,2,1)
imshow(prob_im,gray())
subplot(1,2,2)
imshow(prob_im2)
show()

pair_feature = get_feature_for_pairwise(frames, segs, None, lab_range).astype(np.float32)
pair_features = pair_feature[:len(np.unique(segs[0])) + len(np.unique(segs[1]))]
                                                                       

n_nodes = colors.shape[0]

prob = forest_proba

eps = 1e-7

for u in np.unique(segs[0]):
    rs, cs = np.nonzero(segs[0] == u)
    if np.mean(g[rs,cs]) > 0.7:
        prob[u,:] = [1,0]
    else:
        prob[u,:] = [0,1]

unary = -np.log(prob+eps).astype(np.float32)

# potts = potts_weight * np.array([[0.5,1],
                             #    [1,0.5]], np.float32)
potts_weight = 1
potts = potts_weight * np.array([[0,1],
                                 [1,0]], np.float32)

crf = DenseCRF(n_nodes, 2)

print 'Mean field inference ...'        
crf.set_unary_energy(unary)
crf.add_pairwise_energy(potts, np.ascontiguousarray(pair_features.astype(np.float32)))

iteration = 10

labels = crf.map(iteration)
print ' done.'

count = len(np.unique(segs[0]))

uni = np.unique(segs[1])

mask = np.zeros(segs[1].shape)
for u in uni:
        rows, cols = np.nonzero(segs[1] == u)
        if labels[count] == 0:
            mask[rows, cols] = 1
        else:
            mask[rows, cols] = 0
            
        count += 1

im = img_as_ubyte(imread(frames[1]))
from skimage.morphology import *
boundary = mask - binary_erosion(mask, square(3))
imshow(alpha_composite(im, mask_to_rgb(boundary, (0,255,0))),cmap=gray())                
show()                

# edges = []
# edge_cost = []
# n_temp = 0
# n_node = 0
# rows = []
# cols = []
# values = []
# for i in range(n_frames):
#     uni = np.unique(segs[i])
#     n_node += len(uni)
#     print i
#     for u in uni:
#         rs,cs = np.nonzero(segs[i] == u)

#         for (n_id,adj) in enumerate(adjs[i][u]):
#             if adj == False: continue
#             if node_id[i][u] == node_id[i][n_id]: continue
#             rows.append(node_id[i][u])
#             cols.append(node_id[i][n_id])
# #            values.append(np.exp(-np.linalg.norm(feats[i][u] - feats[i][n_id]) ** 2 / (sigma2)))
#             values.append(np.linalg.norm(feats[i][u] - feats[i][n_id]) ** 2)
#             values.append(values[-1])
#             cols.append(node_id[i][u])
#             rows.append(node_id[i][n_id])

#             edges.append((node_id[i][u], node_id[i][n_id]))
#             edge_cost.append(values[-1])

#         if i < n_frames -1:
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
# edge_cost = np.exp(-np.array(edge_cost) / sigma2)
                
# saliency_fg[:,:,0] = g    
# saliency_bg[:,:,0] = 1 - g    
# lab_range = get_lab_range(frames)
# pair_feature = get_feature_for_pairwise(frames, segs, adjs, lab_range).astype(np.float32) 
# final_mask = segment(frames, saliency_fg*3, saliency_bg, pair_feature, segs, 3)
# final_mask2 = segment2(frames, saliency_fg*3, saliency_bg, segs, np.array(edges), edge_cost, 3, pair_feature.shape[0])

# from video_util import *

# for i in range(n_frames):
#     figure(figsize(20,18))

#     print i
#     im = img_as_ubyte(imread(frames[i]))            
#     subplot(1,3,1)
#     imshow(im)
#     axis("off")

#     subplot(1,3,2)
#     imshow(alpha_composite(im, mask_to_rgb(final_mask[i], (0,255,0))),cmap=gray())        
#     axis("off")    

#     subplot(1,3,3)
#     imshow(alpha_composite(im, mask_to_rgb(final_mask2[i], (0,255,0))),cmap=gray())        
#     axis("off")    
    
#     show() 

#    figure(figsize(20,15))
    
#    imshow(im)
#    im[final_mask[i].astype(np.bool) == 0] = (0,0,0)
 #   imsave('seg/%05d.png' % i, im)

