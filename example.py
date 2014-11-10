# from pylab import *
# import numpy as np
# #from util import *
# from sys import argv
# from time import time
# import os
# from skimage import img_as_ubyte
# from scipy.io import loadmat
# from sklearn.preprocessing import scale
# from skimage.color import rgb2gray,rgb2lab
# from skimage.feature import hog
# from joblib import Parallel, delayed
# from skimage.segmentation import find_boundaries
# from video_graph import *
# from IPython.core.pylabtools import figsize
# from video_util import *
# # from krahenbuhl2013 import DenseCRF
# # #prop = proposals.Proposal( setupBaseline( 130, 5, 0.8 ) )
# # #prop = proposals.Proposal( setupBaseline( 150, 7, 0.85 ) )
# # prop = proposals.Proposal( setupLearned( 150, 5, 0.8 ) )
# # #prop = proposals.Proposal( setupLearned( 160, 6, 0.85 ) )

# # detector = contour.MultiScaleStructuredForest()
# # detector.load( "sf.dat" )
# name = 'bmx'
# #name = 'cheetah'
# def get_dominant_motion(motion):
#     hist,bins = np.histogram(motion.flatten(), bins=500)
#     return bins[np.argmax(hist)]

# imdir = '/home/masa/research/code/rgb/%s/' % name
# vx = loadmat('/home/masa/research/code/flow/%s/vx.mat' % name)['vx']
# vy = loadmat('/home/masa/research/code/flow/%s/vy.mat' % name)['vy']

# frames = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)[:-1]) if f.endswith(".png")]
# from skimage.filter import vsobel,hsobel
# sp_file = "../TSP/results/%s.mat" % name
# sp_label = loadmat(sp_file)["sp_labels"]
# r,c,n = sp_label.shape
# segs,adjs,mappings = get_tsp(sp_label)

# mag = np.sqrt(vx**2 + vy ** 2)
# angle = np.arctan2(vy,vx) / np.pi * 180
# edges = loadmat('/home/masa/research/release/%s.mat' % name)['edges']

# for i in range(len(frames)):
#     print i
#     mag_contrast = np.zeros(mag.shape[:2])
#     angle_contrast = np.zeros(mag.shape[:2])
#     uni = np.unique(segs[i])
#     dominant_angle = get_dominant_angle(angle[:,:,i])
#     dominant_motion = get_dominant_motion(mag[:,:,i])
    
#     for u in uni:
#         rs, cs = np.nonzero(segs[i] == u)
#         mag_contrast[rs,cs] = np.mean(np.abs(mag[rs,cs,i] - dominant_motion))
#         angle_contrast[rs,cs] = np.mean(np.abs(angle[rs,cs,i]  - dominant_angle))     
    
#     u = vx[:,:,i]
#     v = vy[:,:,i]

#     u_x = hsobel(u)
#     u_y = vsobel(u)
#     v_x = hsobel(v)
#     v_y = vsobel(v)


#     # a_u = hsobel(angle[:,:,i])
#     # a_v = vsobel(angle[:,:,i])

#     # figure(figsize(21,18))
#     # subplot(1,5,1)
#     # imshow(angle[:,:,i])

#     # subplot(1,5,2)
#     # imshow(mag[:,:,i])

#     # subplot(1,5,3)
#     # imshow(angle_contrast)

#     # subplot(1,5,4)
#     # imshow(mag_contrast)

#     # gamma = 0.5
#     # coeff = (1-np.exp(-gamma*dominant_motion)) / (1+np.exp(-gamma*dominant_motion))
#     # print coeff, dominant_motion
#     # subplot(1,5,5)
#     # imshow(mag_contrast/np.max(mag_contrast) + coeff * angle_contrast / np.max(angle_contrast))
            
#     # show()
#     # figure(figsize=(21,18))

#     grad_mag = np.sqrt(u_x**2 + u_y** 2 + v_x**2 + v_y**2)

#     imshow(grad_mag,cmap=jet())
        
#     show()

#    #  subplot(1,3,2)
#    #  hst, bin_edges = np.histogram(grad_mag.flatten(), bins=20)    
#    #  imshow(grad_mag >= bin_edges[1],cmap=gray())

#    #  subplot(1,3,3)
#    #  from scipy.ndimage.morphology import distance_transform_edt
    
#    # imshow(distance_transform_edt(1 - (grad_mag >= bin_edges[1] )))

#    # imsave('%05d.png' % i, grad_mag)


#    #  subplot(1,2,2)
#    #  imshow(np.sqrt(a_u**2 + a_v **2))
#    #  colorbar()
    
# gt = get_segtrack_gt(name)

# # # #gt = get_segtrack_gt(name)
# # #lab_range = get_lab_range(frames)

# # #segs,adjs = video_superpixel(frames,detector)
# # sp_file = "../TSP/results/%s.mat" % name
# # sp_label = loadmat(sp_file)["sp_labels"]
# # segs,adjs,mappings = get_tsp(sp_label)
# # r,c,_ = sp_label.shape

# # from collections import defaultdict
# # from skimage.segmentation import mark_boundaries
# # frame_num = 21
# #imshow(mark_boundaries(imread(frames[frame_num]),segs[frame_num])) 
# # count = defaultdict(int)
# # for i in range(len(frames)):
# #     g = gt[0][i] + gt[1][i]
# #     rows, cols = np.nonzero(g)
# #     uni = np.unique(sp_label[:,:,i][rows, cols])
# #     for u in uni:
# #         count[u] += 1

# # bg_label = np.setdiff1d(np.unique(sp_label),array(count.keys()))
# # bg_count = defaultdict(int)
# # for i in range(len(frames)):
# #     for l in bg_label:
# #         if np.sum(sp_label[:,:,i] == l) > 0:
# #             bg_count[l] += 1

# # labels = []

# # for i in range(len(frames)):
# #     label = np.zeros((r,c), np.bool)
# #     labels.append(label)

# # g = gt[0][0]
# # if len(gt) > 1:  g+= gt[1][0]
# # uni = np.unique(sp_label[:,:,0])

# # gt_label = []
# # for u in uni:
# #     rows, cols = np.nonzero(sp_label[:,:,0] == u)
# #     gt_ratio = np.mean(g[rows, cols])
# #     if gt_ratio > 0.8:
# #         gt_label.append(u)

                     
# # # angle_thres = 20        
# # for i in range(len(frames)):

# #     for l in gt_label:
# #        labels[i][sp_label[:,:,i] == l] = True
    
# #     dead = setdiff1d(np.unique(sp_label[:,:,i]), np.unique(sp_label[:,:,i+1]))        
# #     new = setdiff1d(np.unique(sp_label[:,:,i+1]), np.unique(sp_label[:,:,i]))
    
# # #    dominant_angle = get_dominant_angle(angle[:,:,i])
    
# #     for d in dead:
# #         mask = segs[i] == mappings[i][d]
# # #        if abs(np.median(angle[:,:,i][mask]) - dominant_angle) > 20:
# #         labels[i][mask] = 1
           
# #     for n in new:
# #         mask = segs[i+1] == mappings[i+1][n]
# # #        if abs(np.median(angle[:,:,i][mask]) - dominant_angle) > 20:
# #         labels[i+1][mask] = 1

#     # for d in dead:
#     #     rows,cols = np.nonzero(segs[i] == mappings[i][d])
#     #     for j in range(len(rows)):
#     #         if abs(angle[:,:,i][rows[j],cols[j]] - dominant_angle) > 20:
#     #                    labels[i][rows[j],cols[j]] = 1
           
#     # for n in new:
#     #     rows,cols = np.nonzero(segs[i+1] == mappings[i+1][n])
#     #     for j in range(len(rows)):
#     #         if abs(angle[:,:,i+1][rows[j],cols[j]] - dominant_angle) > 20:
#     #                    labels[i+1][rows[j],cols[j]] = 1
                   
        
# for (i,l) in enumerate(labels):
# #    print i,get_dominant_angle(angle[:,:,i])
#  #   imsave("%05d.png" % i, l)
#   #  subplot(1,2,1)
#     # subplot(1,2,2)
#     im = img_as_ubyte(imread(frames[i]))
#     imshow(alpha_composite(im, mask_to_rgb(l, (0,255,0))), cmap=gray())
#     # axis("off")
#     show()
                
# #     #     seg.append(mappings[i][d])
# #     # for n in new:
# #     #     seg.append(mappings[i+1][n])
        
# #feats = get_sp_feature_all_frames(frames,segs, lab_range)
# #unary,prob,rescaled,unnom = flow_unary(frames, segs, vx,vy)
# # potts_cost = 3
# # label_compat = np.array([[0,1],[1,0]], dtype=np.float32) * potts_cost
# # crf = DenseCRF(np.vstack(feats).shape[0], 2)
# # crf.set_unary_energy(unary.astype(np.float32))
# # crf.add_pairwise_energy(label_compat, np.vstack(unary).astype(np.float32))
# # label = crf.map()
# #G, source_id, terminal_id, node_ids, id2region = build_csgraph(segs, feats, adjs, 8)


# # plot_seg(frames, segs, label, id2region)
  

# # local_maxima = get_flow_local_maxima(frames,segs, node_ids, vx, vy,adjs,20,50)
# # # plot_local_maxima(id2region,segs, vx,vy, local_maxima)
# # paths,paths2 = shortest_path(G, local_maxima,source_id, terminal_id)
# # paths,paths2 = shortest_path(G, [node_ids[25][694],node_ids[25][610],node_ids[25][523]], source_id, terminal_id)
# # plot_paths(2, paths, frames,segs, id2region, vx, vy)
# # plot_all_paths(frames,segs, paths+paths2,id2region)

    
    
from pylab import *
import numpy as np
#from util import *
from sys import argv
from time import time
import os
from skimage import img_as_ubyte
from scipy.io import loadmat
from sklearn.preprocessing import scale
from skimage.color import rgb2gray,rgb2lab
from skimage.feature import hog
from joblib import Parallel, delayed
from skimage.segmentation import find_boundaries
from video_graph import *
from IPython.core.pylabtools import figsize
from video_util import *
# from krahenbuhl2013 import DenseCRF
# #prop = proposals.Proposal( setupBaseline( 130, 5, 0.8 ) )
# #prop = proposals.Proposal( setupBaseline( 150, 7, 0.85 ) )
# prop = proposals.Proposal( setupLearned( 150, 5, 0.8 ) )
# #prop = proposals.Proposal( setupLearned( 160, 6, 0.85 ) )

# detector = contour.MultiScaleStructuredForest()
# detector.load( "sf.dat" )
#name = 'soldier'
name = 'hummingbird'
def get_dominant_motion(motion):
    hist,bins = np.histogram(motion.flatten(), bins=500)
    return bins[np.argmax(hist)]

imdir = '/home/masa/research/code/rgb/%s/' % name
vx = loadmat('/home/masa/research/code/flow/%s/vx.mat' % name)['vx']
vy = loadmat('/home/masa/research/code/flow/%s/vy.mat' % name)['vy']

frames = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)[:-1]) if f.endswith(".png")]
from skimage.filter import vsobel,hsobel
sp_file = "../TSP/results2/%s.mat" % name
sp_label = loadmat(sp_file)["sp_labels"]
r,c,n = sp_label.shape
segs,mappings = get_tsp(sp_label)
#segs = loadmat('sp_%s2.mat' % name)['superpixels']



# edges = loadmat('/home/masa/research/release/%s.mat' % name)['edges']
mag = np.sqrt(vx**2 + vy ** 2)
# img = np.zeros((r,c,3))
# gray = rgb2gray(imread(frames[0]))
# for i in range(r):
#     for j in range(c):
#         img[i,j,0] = vx[i,j,0]
#         img[i,j,1] = vy[i,j,0]
#         img[i,j,2] = gray[i,j]

# imshow(img)
# show()        
# for i in range(len(frames)):
#     print i
#     mag_contrast = np.zeros(mag.shape[:2])
#     angle_contrast = np.zeros(mag.shape[:2])
#     uni = np.unique(segs[i])
#     # dominant_angle = get_dominant_angle(angle[:,:,i])
#     # dominant_motion = get_dominant_motion(mag[:,:,i])
    
#     # for u in uni:
#     #     rs, cs = np.nonzero(segs[i] == u)
#     #     mag_contrast[rs,cs] = np.mean(np.abs(mag[rs,cs,i] - dominant_motion))
#     #     angle_contrast[rs,cs] = np.mean(np.abs(angle[rs,cs,i]  - dominant_angle))     
    
#     u = vx[:,:,i]
#     v = vy[:,:,i]

#     u_x = hsobel(u)
#     u_y = vsobel(u)
#     v_x = hsobel(v)
#     v_y = vsobel(v)

#     grad_mag = np.sqrt(u_x**2 + u_y** 2 + v_x**2 + v_y**2)
#     grad_mag = (-np.min(grad_mag) + grad_mag) / (np.max(grad_mag) - np.min(grad_mag))
# #    color_edge = (-np.min(edges[:,:,i]) + edges[:,:,i]) / (np.max(edges[:,:,i]) - np.min(edges[:,:,i]))
#     color_edge = edges[:,:,i]

#     gamma = 10    
#     coeff = (1-np.exp(-gamma*grad_mag)) / (1+np.exp(-gamma*grad_mag))

#     combined = color_edge * coeff

#     figure(figsize(18,15))
#     subplot(1,2,1)
#     imshow(coeff)
#     subplot(1,2,2)
#     imshow(combined)
#     show()


    
    
    # a_u = hsobel(angle[:,:,i])
    # a_v = vsobel(angle[:,:,i])

    # figure(figsize(21,18))
    # subplot(1,5,1)
    # imshow(angle[:,:,i])

    # subplot(1,5,2)
    # imshow(mag[:,:,i])

    # subplot(1,5,3)
    # imshow(angle_contrast)

    # subplot(1,5,4)
    # imshow(mag_contrast)

    # gamma = 0.5
    # coeff = (1-np.exp(-gamma*dominant_motion)) / (1+np.exp(-gamma*dominant_motion))
    # print coeff, dominant_motion
    # subplot(1,5,5)
    # imshow(mag_contrast/np.max(mag_contrast) + coeff * angle_contrast / np.max(angle_contrast))
            
    # show()
#     # figure(figsize=(21,18))



#     imshow(grad_mag,cmap=jet())
        
#     show()

#    #  subplot(1,3,2)
#    #  hst, bin_edges = np.histogram(grad_mag.flatten(), bins=20)    
#    #  imshow(grad_mag >= bin_edges[1],cmap=gray())

#    #  subplot(1,3,3)
#    #  from scipy.ndimage.morphology import distance_transform_edt
    
#    # imshow(distance_transform_edt(1 - (grad_mag >= bin_edges[1] )))

#    # imsave('%05d.png' % i, grad_mag)


#    #  subplot(1,2,2)
#    #  imshow(np.sqrt(a_u**2 + a_v **2))
#    #  colorbar()
    
# gt = get_segtrack_gt(name)

# # #gt = get_segtrack_gt(name)
# #lab_range = get_lab_range(frames)

# #segs,adjs = video_superpixel(frames,detector)
# sp_file = "../TSP/results/%s.mat" % name
# sp_label = loadmat(sp_file)["sp_labels"]
# segs,adjs,mappings = get_tsp(sp_label)
# r,c,_ = sp_label.shape

# from collections import defaultdict
# from skimage.segmentation import mark_boundaries
# frame_num = 21
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
labels2 = []
for i in range(len(frames)):
    label = np.zeros((r,c), np.bool)
    label2 = np.zeros((r,c), np.bool)
    labels.append(label)
    labels2.append(label2)
angle = np.arctan2(vx,vy) / np.pi * 180
for i in range(len(frames)-1):

    # for l in gt_label:
    #    labels[i][sp_label[:,:,i] == l] = True
    
    dead = setdiff1d(np.unique(sp_label[:,:,i]), np.unique(sp_label[:,:,i+1]))        
    new = setdiff1d(np.unique(sp_label[:,:,i+1]), np.unique(sp_label[:,:,i]))
   
    dominant_angle = get_dominant_angle(angle[:,:,i])
    
    for d in dead:
        mask = segs[i] == mappings[i][d]
        labels2[i][mask] = 1
        rows, cols = np.nonzero(mask)
        for y in rows:
            for x in cols:
                if abs(angle[y,x,i] - dominant_angle) > 20:
                    labels[i][y,x] = 1
        # if abs(np.median(angle[:,:,i][mask]) - dominant_angle) > 20:
        #     labels[i][mask] = 1

    dominant_angle = get_dominant_angle(angle[:,:,i+1])                   
    for n in new:
        mask = segs[i+1] == mappings[i+1][n]
        labels2[i+1][mask] = 1        
        rows, cols = np.nonzero(mask)
        for y in rows:
            for x in cols:
                if abs(angle[y,x,i+1] - dominant_angle) > 20:
                    labels[i+1][y,x] = 1
        
        # if abs(np.median(angle[:,:,i][mask]) - dominant_angle) > 20:
        #    labels[i+1][mask] = 1

    figure(figsize(18,15))
    subplot(1,3,1)
    imshow(angle[:,:,i])
    subplot(1,3,2)
    imshow(labels[i])
    subplot(1,3,3)
    imshow(labels2[i])
    
    show()

#     for d in dead:
#         rows,cols = np.nonzero(segs[i] == mappings[i][d])
#         for j in range(len(rows)):
# #            if abs(angle[:,:,i][rows[j],cols[j]] - dominant_angle) > 20:
#                        labels[i][rows[j],cols[j]] = 1
           
#     for n in new:
#         rows,cols = np.nonzero(segs[i+1] == mappings[i+1][n])
#         for j in range(len(rows)):
#  #           if abs(angle[:,:,i+1][rows[j],cols[j]] - dominant_angle) > 20:
#                        labels[i+1][rows[j],cols[j]] = 1
                   
        
# for (i,l) in enumerate(labels):
# #    print i,get_dominant_angle(angle[:,:,i])
#  #   imsave("%05d.png" % i, l)
#   #  subplot(1,2,1)
#     # subplot(1,2,2)
#     im = img_as_ubyte(imread(frames[i]))
#     imshow(alpha_composite(im, mask_to_rgb(l, (0,255,0))), cmap=gray())
#     # axis("off")
#     show()
                
#     #     seg.append(mappings[i][d])
#     # for n in new:
#     #     seg.append(mappings[i+1][n])
        
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

    
    
