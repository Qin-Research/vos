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
from IPython.core.pylabtools import figsize
from scipy.sparse import csr_matrix

#from segmentation import segment

import numpy as np
from sklearn.mixture import GMM
from sklearn.ensemble import RandomForestClassifier
from krahenbuhl2013 import DenseCRF
from pylab import *
import time
import ipdb

def segment(frames, sp_features, pair_features ,segs, mask, max_iter, potts_weight, n_forest=10):
    labels = np.zeros(sp_features.shape[0], dtype=np.bool)
    count = 0
    for j in range(len(mask)):
        uni = np.unique(segs[j])
        for u in uni:
            rows, cols = np.nonzero(segs[j] == u)
            if mask[j][rows[0], cols[0]] > 0:
                labels[count] = 0
            else:
                labels[count] = 1
            count += 1
    

#    sp_feature = get_feature_for_pairwise(frames, segs, adjs, lab_range).astype(np.float32)
#    sp_feature = np.load("feature.npy")[:,:3]
    from sklearn.decomposition import PCA

 #   pca = PCA(50)
#    sp_feature = pca.fit_transform(feats2mat(sp_feature))
 #   sp_feature = feats2mat(sp_feature).astype(np.float32)

    for i in range(max_iter):
        print i

        from sklearn.ensemble import RandomForestClassifier                

        print "Training forest."
        rf = RandomForestClassifier(n_forest)
        sample_weight = np.array(sal)
        sample_weight[labels == 1] = 1
        rf.fit(sp_features, labels,sample_weight)
        print "Done."

        from sklearn.mixture import GMM

        fg_gmm = GMM(5)
        bg_gmm = GMM(7)

        fg_gmm.fit(sp_features[labels == 0])
        bg_gmm.fit(sp_features[labels == 1])
        
        prob = rf.predict_proba(sp_features)
        np.save('prob%d.npy' % i, prob)

#        prob[:, 0] += sal
        prob_fg = np.dot(fg_gmm.predict_proba(sp_features), fg_gmm.weights_).reshape(-1,1)
        prob_bg = np.dot(bg_gmm.predict_proba(sp_features), bg_gmm.weights_).reshape(-1,1)
        prob_gmm = np.hstack((prob_fg, prob_bg))
        np.save('prob_gmm%d.npy' % i, prob_gmm)
        
        eps = 1e-7
        unary = -np.log(prob+eps).astype(np.float32)
#        unary = -np.log(prob_gmm+eps).astype(np.float32)
        unary[np.array(labels) == 1, 0] = np.inf
#        unary = get_unary(frames, segs, saliency, sal_thres)


        potts = potts_weight * np.array([[0.5,1],
                                         [1,0.5]], np.float32) 
        # potts = potts_weight * np.array([[0,1],
        #                                  [1,0]], np.float32)
   
        n_nodes = sp_features.shape[0]
        crf = DenseCRF(n_nodes, 2)

        print 'Mean field inference ...'        
        crf.set_unary_energy(unary)
        crf.add_pairwise_energy(potts, np.ascontiguousarray(pair_features.astype(np.float32)))
    
        iteration = 10

        labels = crf.map(iteration)
        print ' done.'

    count = 0
    for j in range(len(mask)):
        uni = np.unique(segs[j])

        new_mask = np.zeros(mask[j].shape)
        for u in uni:
            rows, cols = np.nonzero(segs[j] == u)
            if labels[count] == 0:
                new_mask[rows, cols] = 1
            else:
                new_mask[rows, cols] = 0
                
            count += 1
            
        mask[j] = new_mask
                
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

def plot_rgbmean(rgb_mean, frames,segs):
    count = 0
    for j in range(len(segs)-1):
        uni = np.unique(segs[j])
        im = img_as_ubyte(imread(frames[j]))
        figure(figsize = (20,18))

        for u in uni:
            rows, cols = np.nonzero(segs[j] == u)
            im[rows, cols] = rgb_mean[count]
            count += 1

        print j            
        imshow(im)
        show()
            

def segment2(frames, sp_features, segs, edges, edge_cost, mask, max_iter, potts_weight, adjs,lab_range):
    import opengm

    labels = np.zeros(sp_features.shape[0], dtype=np.bool)
    count = 0
    for j in range(len(mask)):
        uni = np.unique(segs[j])
        for u in uni:
            rows, cols = np.nonzero(segs[j] == u)
            if mask[j][rows[0], cols[0]] > 0:
                labels[count] = 0
            else:
                labels[count] = 1
            count += 1
    

    for i in range(max_iter):
        print i

        from sklearn.ensemble import RandomForestClassifier                
#        data = np.array(all_samples).reshape(-1,3)
        rf = RandomForestClassifier(10)
        rf.fit(sp_features, labels)
        
        prob = rf.predict_proba(sp_features)
        np.save('prob%d.npy' % i, prob)
        eps = 1e-7
        unary = (-np.log(prob+eps))
        
        unary[np.array(labels) == 1, 0] = np.inf
#        unary = get_unary(frames, segs, saliency, sal_thres)

        n_nodes = sp_features.shape[0]
        gm = opengm.graphicalModel(np.ones(n_nodes, dtype=opengm.index_type) * 2, operator="adder")
        fids=gm.addFunctions(unary)
        vis=np.arange(0,unary.shape[0],dtype=np.uint64)
# add all unary factors at once
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
    for j in range(len(mask)):
        uni = np.unique(segs[j])

        new_mask = np.zeros(mask[j].shape)
        for u in uni:
            rows, cols = np.nonzero(segs[j] == u)
            if labels[count] == 0:
                new_mask[rows, cols] = 1
            else:
                new_mask[rows, cols] = 0
                
            count += 1
            
        mask[j] = new_mask
                
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
                                  
#        feature = np.concatenate((feature,np.min(rgbs, axis=0)))
#       feature = np.concatenate((feature,np.max(rgbs, axis=0)))
        # for c in range(3):
        #     hist, bin_edges = np.histogram(rgbs[:,c], bins=n_bins, range=(0,256),normed=True )
        #     feature = np.concatenate((feature, hist))
        # for c in range(3):
        #      hist, bin_edges = np.histogram(labs[:,c], bins=n_bins, range=(lab_range[c,0], lab_range[c,1]))
        #      feature = np.concatenate((feature, hist))
        #center_y = round(np.mean(rows))
        #center_x = round(np.mean(cols))
        # patch = gray[center_y:center_y+15, center_x:center_x+15]
        # hog_feat = hog(patch,orientations=6,pixels_per_cell=(5,5), cells_per_block=(3,3))
        # feature = np.concatenate((feature, hog_feat))
        # feature = np.concatenate((feature, np.array([np.mean(rows)/image.shape[0], np.mean(cols)/image.shape[1]])))
 #       feature = np.concatenate((feature, np.mean(rgbs, axis=0)))
 #       feature = np.concatenate((feature, np.mean(labs, axis=0)))

        if features == None:
            dim = len(feature)
            features = np.zeros((n, dim))
            features[0] = feature
        else:
            features[i] = feature

 #   return scale(features)

    
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
#name = 'cheetah'

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
lab_range = get_lab_range(frames)
feats = get_sp_rgb_mean_all_frames(frames,segs, lab_range)

node_id = []

id_count = 0
loc_unary = loadmat('../FastVideoSegment/%s_loc.mat' % name)['loc']
saliency = loadmat('/home/masa/research/saliency/PCA_Saliency_CVPR2013/%s.mat' % name)['out']

init_sal = loc_unary * saliency[:,:,:loc_unary.shape[2]]
sal = []
for i in range(n_frames):

    uni = np.unique(segs[i])
    for (j,u) in enumerate(uni):
        rs, cs = np.nonzero(segs[i] == u)
        sal.append(np.mean(init_sal[:,:,i][rs,cs]))


        
count = 0
masks = []
ims = []
sal_images = []
for i in range(n_frames):
    sal_image = np.zeros((r,c))

    im = img_as_ubyte(imread(frames[i]))    
    uni = np.unique(segs[i])
    s = np.copy(sal[count:count+len(uni)])
    s = (s - np.min(s)) / (np.max(s) - np.min(s))
    thres = mean(s)
    for (j,u) in enumerate(uni):
        rs, cs = np.nonzero(segs[i] == u)
        sal_image[rs,cs] = s[j]
        # if s[j] < thres:
        #     im[rs,cs] = (0,0,0)
        count += 1

    hst, bin_edges = np.histogram(sal_image.flatten(), bins=20)
    thres = mean(sal_image[sal_image > bin_edges[1]])
    im[sal_image < thres] = (0,0,0)
    ims.append(im)
    sal_images.append(sal_image)
    masks.append(sal_image>thres)

from copy import deepcopy
#sp_feature = get_sp_feature_all_frames(frames,segs, lab_range)
sp_feature = feats2mat(feats).astype(np.float32)
#sp_feature = feats2mat(sp_feature).astype(np.float32)
    

#sp_feature2  = get_feature_for_pairwise(frames, segs, adjs, lab_range).astype(np.float32)
pair_feature = get_feature_for_pairwise(frames, segs, adjs, lab_range).astype(np.float32) 
final_mask = segment(frames, sp_feature, pair_feature, segs, deepcopy(masks), 3, 3, 20)

#final_mask2 = segment2(frames, sp_feature, segs, np.array(edges), edge_cost, deepcopy(masks), 5, 3, adjs,lab_range)

final_mask2 = final_mask
from video_util import *

for i in range(n_frames):
    figure(figsize(20,18))

    print i
#    im[final_mask[i].astype(np.bool) == 0] = (0,0,0)
#    im2[final_mask2[i].astype(np.bool) == 0] = (0,0,0)
    im = img_as_ubyte(imread(frames[i]))            
    subplot(1,5,1)
    imshow(init_sal[:,:,i],cmap=gray())
    axis("off")

    subplot(1,5,2)
    imshow(sal_images[i],cmap=gray())
    axis("off")    

    subplot(1,5,3)
    imshow(ims[i],cmap=gray())
    axis("off")    

    subplot(1,5,4)
    imshow(alpha_composite(im, mask_to_rgb(final_mask[i], (0,255,0))),cmap=gray())
    axis("off")    

    subplot(1,5,5)
    imshow(alpha_composite(im, mask_to_rgb(final_mask2[i], (0,255,0))),cmap=gray())
    axis("off")    
    

    show() 

#    figure(figsize(20,15))
    
#    imshow(im)
#    im[final_mask[i].astype(np.bool) == 0] = (0,0,0)
 #   imsave('seg/%05d.png' % i, im)

