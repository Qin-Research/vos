from video_util import *
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




def segment(frames, unary, pair_features ,segs,potts_weights):
    n_nodes = unary.shape[0]
    # prob_fg = np.zeros(n_nodes)
    # prob_bg = np.zeros(n_nodes)

    # count = 0
    # for j in range(len(segs)-1):
    #     uni = np.unique(segs[j])
    #     for u in uni:
    #         rows, cols = np.nonzero(segs[j] == u)
    #         prob_fg[count] = np.mean(saliency_fg[rows, cols, j])
    #         prob_bg[count] = np.mean(saliency_bg[rows, cols, j])

    #         count += 1
    
    # prob = np.hstack((prob_fg[:,np.newaxis], prob_bg[:, np.newaxis]))
    
    # eps = 1e-7
    # unary = -np.log(prob+eps).astype(np.float32)

#    potts = potts_weight * np.array([[0.5,1],
                                 #    [1,0.5]], np.float32) 
    potts =                np.array([[0,1],
                                     [1,0]], np.float32)

    crf = DenseCRF(n_nodes, 2)

    print 'Mean field inference ...'        
    crf.set_unary_energy(unary.astype(np.float32))
    
    for (i,f) in enumerate(pair_features):
      crf.add_pairwise_energy((potts_weights[i] * potts).astype(np.float32), np.ascontiguousarray(pair_features[i]).astype(np.float32))
    
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
        
def segment2(frames, unary,source, target, value, segs, potts_weight):
    import opengm

    n_nodes = unary.shape[0]
    # prob_fg = np.zeros(n_nodes)
    # prob_bg = np.zeros(n_nodes)

    # count = 0
    # for j in range(len(segs)-1):
    #     uni = np.unique(segs[j])
    #     for u in uni:
    #         rows, cols = np.nonzero(segs[j] == u)
    #         prob_fg[count] = np.mean(saliency_fg[rows, cols, j])
    #         prob_bg[count] = np.mean(saliency_bg[rows, cols, j])

    #         count += 1
    
    # prob = np.hstack((prob_fg[:,np.newaxis], prob_bg[:, np.newaxis]))
    
    # eps = 1e-7
    # unary = -np.log(prob+eps).astype(np.float32)

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

def superpixel_feature(image,seg,ii, n_frames):
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
        feature = np.mean(rgbs, axis=0)
        center_y = np.mean(rows) 
        center_x = np.mean(cols) 
        feature = np.concatenate((feature,np.array([center_y,center_x, ii])))

        patch = gray[round(center_y):round(center_y+15), round(center_x):round(center_x)+15]
        hog_feat = hog(patch,orientations=6,pixels_per_cell=(5,5), cells_per_block=(3,3))
        feature = np.concatenate((feature, hog_feat))
                                  
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
        features = superpixel_feature(img_as_ubyte(imread(im)), segs[ii], ii, len(frames))
        feats.append(features)
        
    return feats

def feats2mat(feats):
    ret = feats[0]
    for feat in feats[1:]:
        ret = np.vstack((ret, feat))
    return ret

def get_feature_for_pairwise(frames, segs):
    print 'foo'
    features = feats2mat(get_sp_feature_all_frames(frames, segs, None))
    new_features = np.zeros(features.shape)

    return features

name = 'bmx'


imdir = '/home/masa/research/code/rgb/%s/' % name
vx = loadmat('/home/masa/research/code/flow/%s/vx.mat' % name)['vx']
vy = loadmat('/home/masa/research/code/flow/%s/vy.mat' % name)['vy']

frames = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)) if f.endswith(".png")] 
from skimage.filter import vsobel,hsobel

mag = np.sqrt(vx**2 + vy ** 2)
r,c,n_frames = mag.shape
n_frames+=1
sp_file = "../TSP/results2/%s.mat" % name
sp_label = loadmat(sp_file)["sp_labels"]
segs,mappings = get_tsp(sp_label)
#segs = loadmat('sp_%s2.mat' % name)['superpixels'].astype(int)
# s = []
# for i in range(n_frames):
#      s.append(segs[:,:,i])
# segs = s    
    
lab_range = get_lab_range(frames)
#feats = get_sp_rgb_mean_all_frames(frames,segs, lab_range)

source = loadmat('/home/masa/research/FastVideoSegment/source_%s.mat' % name)['source'].flatten().astype(np.int)
target = loadmat('/home/masa/research/FastVideoSegment/target_%s.mat'% name)['target'].flatten().astype(np.int)
value = loadmat('/home/masa/research/FastVideoSegment/value_%s.mat'% name)['value'].flatten()
unary = loadmat('/home/masa/research/FastVideoSegment/unary_%s.mat'% name)['unaryPotentials']


gt = get_segtrack_gt(name)
g = gt[0][0]

if len(gt)>1: g += gt[1][0]

# node_id = []
# id_count = 0
# for i in range(n_frames):
#     uni = np.unique(segs[i])
#     id_dict = {}
#     for u in uni:
#         rs, cs = np.nonzero(segs[i] == u)
#         id_dict[u] = id_count
#         id_count += 1
#     node_id.append(id_dict)
    
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
                
lab_range = get_lab_range(frames)
pair_feature = get_feature_for_pairwise(frames, segs).astype(np.float32)
color = pair_feature[:,:3] / 30
loc = pair_feature[:,3:5] / 80
t = pair_feature[:,5][:,np.newaxis] * 10
hg = pair_feature[:,6:] 
#potts_weight = [10,10]


#pair_features = [pair_feature[:,:3], pair_feature[:,3:5]]
# pair_features = [np.hstack((color, loc,t))]
# potts_weight = [10]
pair_features = [np.hstack((color,loc,t))]

#potts_weight = [1000] bmx
potts_weight = [1000] 
final_mask = segment(frames, 10 * np.ascontiguousarray(unary), pair_features, segs, potts_weight)
final_mask2 = segment2(frames, 10 * np.ascontiguousarray(unary),source, target, value, segs, 10)
#final_mask2 = final_mask

for i in range(n_frames-1):
    figure(figsize(20,18))

    print i
    im = img_as_ubyte(imread(frames[i]))            
    subplot(1,3,1)
    imshow(im)
    axis("off")

    subplot(1,3,2)
    imshow(alpha_composite(im, mask_to_rgb(final_mask[i], (0,255,0))),cmap=gray())        
    axis("off")    

    subplot(1,3,3)
    imshow(alpha_composite(im, mask_to_rgb(final_mask2[i], (0,255,0))),cmap=gray())        
    axis("off")    
    
    show() 
    
#    figure(figsize(20,15))
    
#    imshow(im)
#    im[final_mask[i].astype(np.bool) == 0] = (0,0,0)
 #   imsave('seg/%05d.png' % i, im)

