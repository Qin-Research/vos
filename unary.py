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

name = 'hummingbird'
imdir = '/home/masa/research/code/rgb/%s/' % name
loc = loadmat('../FastVideoSegment/%s_loc2.mat' % name)['loc']


frames = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)) if f.endswith(".png")]
n_frames = len(frames)

sp_file = "../TSP/results2/%s.mat" % name
sp_label = loadmat(sp_file)["sp_labels"]
segs,mappings = get_tsp(sp_label)
vx = loadmat('/home/masa/research/code/flow/%s/vx.mat' % name)['vx']
vy = loadmat('/home/masa/research/code/flow/%s/vy.mat' % name)['vy']
mag = np.sqrt(vx**2 + vy ** 2)
angle = np.arctan2(vx,vy) / np.pi * 180
from cPickle import load
with open('paths_%s.pickle' % name) as f:
    paths = load(f)

prop_flow = np.zeros(mag.shape)    
prop_flow2 = np.zeros(mag.shape)    
for (i,id) in enumerate(paths.keys()):
    frame = paths[id].frame
    rows = paths[id].rows
    cols = paths[id].cols
    unique_frame = np.unique(frame)

    flows = paths[id].mean_flows
    m = np.sqrt(flows[:,0] ** 2 + flows[:,1] **2)
    a = np.max(np.arctan2(flows[:,0], flows[:,1]))
    med_mag = np.max(m)
    for f in unique_frame:
        rs = rows[frame == f]
        cs = cols[frame == f]
        prop_flow[rs,cs,f] = med_mag
        prop_flow2[rs,cs,f] = a

for i in range(prop_flow.shape[2]):

    print i
    subplot(1,2,1)
    imshow(prop_flow[:,:,i])

    subplot(1,2,2)
    imshow(prop_flow2[:,:,i])
    
    show()    
        
r,c,_ = vx.shape
for i in range(n_frames-1):
    mask = loc[:,:,i] > 0.05
    im = img_as_ubyte(imread(frames[i]))
    
    uni = np.unique(segs[i])
    n_sp = len(uni)

    bg_sp = []
    rrows = []
    ccols = []

    for u in range(n_sp):
        rrows.append([])
        ccols.append([])
    for rr in range(r):
        for cc in range(c):
            l = segs[i][rr,cc]
            rrows[l].append(rr)
            ccols[l].append(cc)
                 
    for u in uni:
        rows = rrows[u]
        cols = ccols[u]

        
        if mask[rows,cols][0] == 0:
            bg_sp.append(u)

    mean_flow = np.zeros((n_sp,2))
    mean_rgb = np.zeros((n_sp,3))
    mean_angle = np.zeros(n_sp)
    mean_mag = np.zeros(n_sp)
    
    
    for u in uni:
        rows = rrows[u]
        cols = ccols[u]
        mean_rgb[u] = np.mean(im[rows, cols], axis=0)
        mean_flow[u] = np.array([np.mean(vx[rows, cols]), np.mean(vy[rows, cols])])
        mean_angle[u] = np.mean(angle[rows, cols,i])
        mean_mag[u] = np.mean(mag[rows, cols,i])

    sal = []
    sal2 = []
    sal_angle = []
    sal_mag = []
    flow_weight = 0
    sal_image = np.zeros((r,c))
    sal_image2 = np.zeros((r,c))
    sal_image_angle = np.zeros((r,c))
    sal_image_mag = np.zeros((r,c))

    for u in uni:
        if u in bg_sp:
             sal.append(0)
             sal2.append(0)
             sal_mag.append(0)
             sal_angle.append(0)
        else:
            sum_sal = 0
            sum_sal2 = 0
            sum_sal_angle = 0
            sum_sal_mag = 0
            for uu in bg_sp:
                sum_sal += np.linalg.norm(mean_rgb[u] - mean_rgb[uu])
                sum_sal2 += np.linalg.norm(mean_flow[u] - mean_flow[uu])
                sum_sal_mag += abs(mean_mag[u] - mean_mag[uu])
                sum_sal_angle += abs(mean_angle[u] - mean_angle[uu])
                
            sal.append(sum_sal)
            sal2.append(sum_sal2)
            sal_mag.append(sum_sal_mag)
            sal_angle.append(sum_sal_angle)

        rows = rrows[u]
        cols = ccols[u]
                    
        sal_image[rows, cols] = sal[-1]
        sal_image2[rows, cols] = sal2[-1]
        sal_image_angle[rows,cols] = sal_angle[-1]
        sal_image_mag[rows,cols] = sal_mag[-1]
        
        
    figure(figsize(21,18))


    print i

    subplot(1,4,1)

    imshow(sal_image,gray())
#    imshow(alpha_composite(im, mask_to_rgb(mask, (0,255,0))))
    axis("off")

    subplot(1,4,2)
    imshow(sal_image2)

    subplot(1,4,3)
    imshow(sal_image_angle)

    subplot(1,4,4)
    imshow(sal_image_mag)    
#    imshow(sal_image / np.max(sal_image) * sal_image2  / np.max(sal_image2))    
    show()

