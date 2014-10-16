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

name = 'cheetah'
name = 'bmx'
#name = 'hummingbird'

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

saliency = loadmat('/home/masa/research/saliency/PCA_Saliency_CVPR2013/%s.mat' % name)['out']

sal = np.zeros((r,c,vx.shape[2]))

out_dir = 'sal_%s' % name
# if not os.path.exists(out_dir): os.mkdir(out_dir)
# for d in ['/1','/2', '/3', '/4']:
#     os.mkdir(out_dir + d)
    
for j in range(vx.shape[2]):
    uni = np.unique(segs[j])
    flow_mean = np.zeros(len(uni))

    for (i,u) in enumerate(uni):
        rows, cols = np.nonzero(segs[j] == u)
        fm = np.mean(mag[rows, cols,j])
        flow_mean[i] = fm
        
    from IPython.core.pylabtools import figsize
    figure(figsize=(20,15))

    flow_sal = np.zeros(len(uni))

    sal_image = np.zeros(segs[j].shape)
    flow_image = np.zeros(segs[j].shape)    
    for (i,u) in enumerate(uni):
        fm = flow_mean[i]
        sm = 0

        for (ii,uu) in enumerate(uni):
            if ii == i: continue
            sm += np.abs(flow_mean[ii] - fm)
            
        rows, cols = np.nonzero(segs[j] == u)
        sal_image[rows, cols] = sm
        flow_image[rows, cols] = fm


    final_sal = sal_image / np.max(sal_image) * saliency[:,:,j]
    final_sal2 = np.exp(sal_image / np.max(sal_image)) * saliency[:,:,j]
    
    sal[:,:,j] = final_sal
        
    subplot(1,4,1)
    imshow(flow_image,cmap=gray())    
    axis('off')
#    imsave(out_dir + '/1/%05d.png' % j, flow_image)
    
    subplot(1,4,2)
    imshow(saliency[:,:,j],cmap=gray())    
    axis('off')
#    imsave(out_dir + '/2/%05d.png' % j, saliency[:,:,j])
    
    subplot(1,4,3)
    imshow(final_sal,cmap=gray())
    axis('off')
#    imsave(out_dir + '/3/%05d.png' % j, final_sal)    
    
    subplot(1,4,4)
    imshow(final_sal2,cmap=gray())
    axis('off')
 #   imsave(out_dir + '/4/%05d.png' % j, final_sal2)    
    
    show()

count = 0
from skimage import img_as_ubyte

for i in range(n_frames):
    sal_image = np.zeros((r,c))
    im = img_as_ubyte(imread(frames[i]))    
    uni = np.unique(segs[i])

    hist, bin_edges = np.histogram(sal[:,:,i].flatten(), bins=20) 
    thres = mean(sal[:,:,i][sal[:,:,i] > bin_edges[1]])
    print i,thres

    im[sal[:,:,i] < thres] = (0,0,0)

    figure(figsize(20,15))
    imshow(im,cmap=gray())
    show()
                        
np.save('sal_%s.npy' % name, sal)
