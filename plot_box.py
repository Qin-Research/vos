from scipy.io import loadmat
import os
from pylab import *
from IPython.core.pylabtools import figsize
name = 'hummingbird'

bbox = loadmat('/home/masa/research/release/%s_box.mat' % name)['bb']
imdir = '/home/masa/research/code/rgb/%s/' % name

frames = [os.path.join(imdir, f) for f in sorted(os.listdir(imdir)) if f.endswith(".png")]

for i in range(len(frames)):
    print i
    im = imread(frames[i])
    r,c,_ = im.shape
    figure(figsize(12,9))
    score = np.zeros((r,c))
    for n in range(bbox[i][0].shape[0]):
        bb = bbox[i][0]

        box_x = [bb[n][0], bb[n][0], bb[n][0] + bb[n][2],bb[n][0] + bb[n][2]]
        box_y= [bb[n][1], bb[n][1]+bb[n][3],bb[n][1]+bb[n][3], bb[n][1]]
        mask = np.zeros((r,c),bool)
        mask[int(box_y[0]):int(box_y[1]), int(box_x[0]):int(box_x[2])] = 1
        score[mask] += bb[n][4]

    subplot(1,2,1)
    imshow(im)
    subplot(1,2,2)    
    imshow(score)
    show()
        

#        subplot(1,5,n+1 )    
 #       imshow(im)
  #      plot([bb[n][0], bb[n][0], bb[n][0] + bb[n][2],bb[n][0] + bb[n][2]], [bb[n][1], bb[n][1]+bb[n][3],bb[n][1]+bb[n][3], bb[n][1]])
   #     axis('off')
#    show()
