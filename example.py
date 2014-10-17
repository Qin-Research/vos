from pylab import *
import numpy as np
from util import *
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

# from krahenbuhl2013 import DenseCRF
# #prop = proposals.Proposal( setupBaseline( 130, 5, 0.8 ) )
# #prop = proposals.Proposal( setupBaseline( 150, 7, 0.85 ) )
# prop = proposals.Proposal( setupLearned( 150, 5, 0.8 ) )
# #prop = proposals.Proposal( setupLearned( 160, 6, 0.85 ) )

# detector = contour.MultiScaleStructuredForest()
# detector.load( "sf.dat" )
name = 'bmx'
#name = 'bmx'
#name = 'cheetah'

imdir = '/home/masa/research/code/rgb/%s/' % name
vx = loadmat('/home/masa/research/code/flow/%s/vx.mat' % name)['vx']
vy = loadmat('/home/masa/research/code/flow/%s/vy.mat' % name)['vy']

frames = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)[:-1]) if f.endswith(".png")]
from skimage.filter import vsobel,hsobel

mag = np.sqrt(vx**2 + vy ** 2)
angle = np.arctan2(vy,vx) / np.pi * 180

for i in range(len(frames)):
    u = vx[:,:,i]
    v = vy[:,:,i]

    u_x = hsobel(u)
    u_y = vsobel(u)
    v_x = hsobel(v)
    v_y = vsobel(v)

    # a_u = hsobel(angle[:,:,i])
    # a_v = vsobel(angle[:,:,i])

    figure(figsize=(12,9))

#    subplot(1,2,1)
    grad_mag = np.sqrt(u_x**2 + u_y** 2 + v_x**2 + v_y**2)
    imshow(grad_mag,cmap=jet())
    imsave('%05d.png' % i, grad_mag)
    colorbar()

    # subplot(1,2,2)
    # imshow(np.sqrt(a_u**2 + a_v **2))
    # colorbar()
    show()
    
gt = get_segtrack_gt(name)

# #gt = get_segtrack_gt(name)
#lab_range = get_lab_range(frames)

#segs,adjs = video_superpixel(frames,detector)
sp_file = "../TSP/results/%s.mat" % name
sp_label = loadmat(sp_file)["sp_labels"]
segs,adjs,mappings = get_tsp(sp_label)
r,c,_ = sp_label.shape

from collections import defaultdict
from skimage.segmentation import mark_boundaries
frame_num = 21
#imshow(mark_boundaries(imread(frames[frame_num]),segs[frame_num])) 
# count = defaultdict(int)
# for i in range(len(frames)):
#     g = gt[0][i] + gt[1][i]
#     rows, cols = np.nonzero(g)
#     uni = np.unique(sp_label[:,:,i][rows, cols])
#     for u in uni:
#         count[u] += 1

# bg_label = np.setdiff1d(np.unique(sp_label),array(count.keys()))
# bg_count = defaultdict(int)
# for i in range(len(frames)):
#     for l in bg_label:
#         if np.sum(sp_label[:,:,i] == l) > 0:
#             bg_count[l] += 1

labels = []

for i in range(len(frames)):
    label = np.zeros((r,c), np.bool)
    labels.append(label)


angle_thres = 20        
for i in range(len(frames)-1):
    
    dead = setdiff1d(np.unique(sp_label[:,:,i]), np.unique(sp_label[:,:,i+1]))        
    new = setdiff1d(np.unique(sp_label[:,:,i+1]), np.unique(sp_label[:,:,i]))
    
    dominant_angle = get_dominant_angle(angle[:,:,i])
    
    for d in dead:
        mask = segs[i] == mappings[i][d]
        if abs(np.median(angle[:,:,i][mask]) - dominant_angle) > 20:
           labels[i][mask] = 1
           
    for n in new:
        mask = segs[i+1] == mappings[i+1][n]
        if abs(np.median(angle[:,:,i][mask]) - dominant_angle) > 20:
           labels[i+1][mask] = 1

    # for d in dead:
    #     rows,cols = np.nonzero(segs[i] == mappings[i][d])
    #     for j in range(len(rows)):
    #         if abs(angle[:,:,i][rows[j],cols[j]] - dominant_angle) > 20:
    #                    labels[i][rows[j],cols[j]] = 1
           
    # for n in new:
    #     rows,cols = np.nonzero(segs[i+1] == mappings[i+1][n])
    #     for j in range(len(rows)):
    #         if abs(angle[:,:,i+1][rows[j],cols[j]] - dominant_angle) > 20:
    #                    labels[i+1][rows[j],cols[j]] = 1
                   
        
for (i,l) in enumerate(labels):
    figure(figsize(12,9))
#    print i,get_dominant_angle(angle[:,:,i])
 #   imsave("%05d.png" % i, l)
  #  subplot(1,2,1)
    imshow(angle[:,:,i], cmap=jet())
    colorbar()
    axis('off')

    # subplot(1,2,2)
    # imshow(l, cmap=gray())
    # axis("off")
    show()
                
    #     seg.append(mappings[i][d])
    # for n in new:
    #     seg.append(mappings[i+1][n])
        
#feats = get_sp_feature_all_frames(frames,segs, lab_range)
#unary,prob,rescaled,unnom = flow_unary(frames, segs, vx,vy)
# potts_cost = 3
# label_compat = np.array([[0,1],[1,0]], dtype=np.float32) * potts_cost
# crf = DenseCRF(np.vstack(feats).shape[0], 2)
# crf.set_unary_energy(unary.astype(np.float32))
# crf.add_pairwise_energy(label_compat, np.vstack(unary).astype(np.float32))
# label = crf.map()
#G, source_id, terminal_id, node_ids, id2region = build_csgraph(segs, feats, adjs, 8)


# plot_seg(frames, segs, label, id2region)
  

# local_maxima = get_flow_local_maxima(frames,segs, node_ids, vx, vy,adjs,20,50)
# # plot_local_maxima(id2region,segs, vx,vy, local_maxima)
# paths,paths2 = shortest_path(G, local_maxima,source_id, terminal_id)
# paths,paths2 = shortest_path(G, [node_ids[25][694],node_ids[25][610],node_ids[25][523]], source_id, terminal_id)
# plot_paths(2, paths, frames,segs, id2region, vx, vy)
# plot_all_paths(frames,segs, paths+paths2,id2region)

    
    