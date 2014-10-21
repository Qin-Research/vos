from pylab import *
import numpy as np
from sys import argv
from time import time
import os
from scipy.io import loadmat
from sklearn.preprocessing import scale
from skimage.color import rgb2gray,rgb2lab
from skimage.feature import hog
from joblib import Parallel, delayed
from skimage.segmentation import find_boundaries
from video_graph import *
from IPython.core.pylabtools import figsize
from scipy.sparse import csr_matrix    
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
init_sal = np.load('sal_%s.npy' % name)
rhs = []
for i in range(n_frames):
    uni = np.unique(segs[i])
    id_dict = {}
    for u in uni:
        rs, cs = np.nonzero(segs[i] == u)
        rhs.append(np.mean(init_sal[:,:,i][rs,cs]))
        id_dict[u] = id_count
        id_count += 1
    node_id.append(id_dict)
    
rows = []
cols = []
values = []
n_node = 0

sigma2 = 10000
edges = []
edge_cost = []
n_temp = 0
for i in range(n_frames):
    uni = np.unique(segs[i])
    n_node += len(uni)
    print i
    for u in uni:
        rs,cs = np.nonzero(segs[i] == u)

        for (n_id,adj) in enumerate(adjs[i][u]):
            if adj == False: continue
            if node_id[i][u] == node_id[i][n_id]: continue
            rows.append(node_id[i][u])
            cols.append(node_id[i][n_id])
            values.append(np.exp(-np.linalg.norm(feats[i][u] - feats[i][n_id]) ** 2 / (sigma2)))
            values.append(values[-1])
            cols.append(node_id[i][u])
            rows.append(node_id[i][n_id])

            edges.append((node_id[i][u], node_id[i][n_id]))
            edge_cost.append(values[-1])

        if i < n_frames -1:
            if np.sum(sp_label[:,:,i+1] == mappings[i][:u]) > 0:

                id = node_id[i+1][mappings[i+1][mappings[i][:u]]]
                if node_id[i][u] == id: continue
                rows.append(node_id[i][u])
                cols.append(id)
                values.append(np.exp(-np.linalg.norm(feats[i][:u] - feats[i+1][mappings[i+1][mappings[i][:u]]]) ** 2 / sigma2))
                values.append(values[-1])
                cols.append(node_id[i][u])
                rows.append(id)

                edges.append((node_id[i][u], id))                
                edge_cost.append(values[-1])
                n_temp += 1
                
from scipy.sparse import csr_matrix, spdiags                                   
W = csr_matrix((values, (rows, cols)), shape=(n_node, n_node))

inv_D =spdiags(1.0/((W.sum(axis=1)).flatten()), 0, W.shape[0], W.shape[1])
D =spdiags(W.sum(axis=1).flatten(), 0, W.shape[0], W.shape[1])
lam = 100
lhs = D + lam * (D - W)
from scipy.sparse import eye

#lhs = eye(n_node) - (inv_D.dot(W))

from scipy.sparse.linalg import spsolve,lsmr
sal = spsolve(lhs, D.dot(np.array(rhs)))


# sal = np.array(rhs)
# A = inv_D.dot(W)

# for i in range(10000):
#     sal = A.dot(sal)

#sal = (sal - np.min(sal)) / (np.max(sal) - np.min(sal))        

from skimage import img_as_ubyte

count = 0
masks = []
ims = []
sal_images = []
for i in range(n_frames):
    sal_image = np.zeros((r,c))

    im = img_as_ubyte(imread(frames[i]))    
    uni = np.unique(segs[i])
    s = sal[count:count+len(uni)]
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
    
#from segmentation import segment

import numpy as np
from video_graph import *
from sklearn.mixture import GMM
from sklearn.ensemble import RandomForestClassifier
from krahenbuhl2013 import DenseCRF
from pylab import *
import time

def segment(frames, sp_features, segs, mask, max_iter, potts_weight, adjs, lab_range):
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
    

    sp_feature = get_feature_for_pairwise(frames, segs, adjs, lab_range).astype(np.float32)
    from sklearn.decomposition import PCA

    pca = PCA(50)
    sp_feature = pca.fit_transform(feats2mat(sp_feature))                 
    for i in range(max_iter):
        print i

        from sklearn.ensemble import RandomForestClassifier                
#        data = np.array(all_samples).reshape(-1,3)
        print "Training forest."
        rf = RandomForestClassifier()
        rf.fit(sp_features, labels)
        print "Done."
        
        prob = rf.predict_proba(sp_features)

        eps = 1e-7
        unary = -np.log(prob+eps).astype(np.float32)
        unary[np.array(labels) == 1, 0] = np.inf
#        unary = get_unary(frames, segs, saliency, sal_thres)


        potts = potts_weight * np.array([[0,1],
                                         [1,0]], np.float32)
    
        n_nodes = sp_features.shape[0]
        crf = DenseCRF(n_nodes, 2)

        print 'Mean field inference ...'        
        crf.set_unary_energy(unary)
        crf.add_pairwise_energy(potts, np.ascontiguousarray(sp_feature))
    
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

final_mask = segment(frames, sp_feature, segs, deepcopy(masks), 3, 1, adjs,lab_range)


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
        rf = RandomForestClassifier()
        rf.fit(sp_features, labels)
        
        prob = rf.predict_proba(sp_features)

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
    

final_mask2 = segment2(frames, sp_feature, segs, np.array(edges), edge_cost, deepcopy(masks), 5, 10, adjs,lab_range)

from video_util import *
count = 0
for i in range(n_frames):
    figure(figsize(27,21))

#    im[final_mask[i].astype(np.bool) == 0] = (0,0,0)
#    im2[final_mask2[i].astype(np.bool) == 0] = (0,0,0)
        
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
    # for i in range(n_frames):
# #    figure(figsize(20,15))
    
#     imshow(im)
#     imsave('seg/%05d.png' % i, im)
#     show()
