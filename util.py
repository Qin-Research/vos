import numpy as np
import os
from pylab import *
import shutil
from PIL import Image
from skimage.color import rgb2gray
from skimage.io import imread
from skimage import color
from skimage.segmentation import relabel_sequential
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from bidict import bidict
from joblib import Parallel, delayed

ldof_cpu = "external/pami2010LibLinux64/demo_ldof"
color_flow = 'external/flow_util/color_flow'
segtrackv2_dir = "data/SegTrackv2/"

def alpha_composite(img, mask, alpha=0.5):
    comp = Image.blend(Image.fromarray(img), Image.fromarray(mask), alpha)
    return np.asarray(comp)

def mask_to_rgb(mask,color):
    mask = rgb2gray(mask)
    mask = mask > 0.5
    r,c = mask.shape
    mask_rgb = np.zeros((r,c,3))
    rows,cols = np.nonzero(mask)
    mask_rgb[rows, cols,:] = color
    mask_rgb = mask_rgb.astype(np.ubyte)
    return mask_rgb

def make_overlay_mask(img, mask):
    rows, cols, _ = np.nonzero(mask)
    overlay_mask = img.copy()
    overlay_mask[rows, cols] = mask[rows, cols]
    return overlay_mask

def get_segtrack_gt(name):
    with open(os.path.join(segtrackv2_dir, "ImageSets", name+".txt")) as f:
        file_list = f.readlines()[1:]
        
    gt_dir = os.path.join(segtrackv2_dir,"GroundTruth", name)
    entries = os.listdir(gt_dir)
    gts = []
    if len(entries) < 3 :
       for e in entries:
            gt_path = os.path.join(gt_dir, e)
            gt = [os.path.join(gt_path, f.rstrip()+ ".png") for f in file_list]
            if not os.path.exists(gt[0]):
                gt = [os.path.join(gt_path, f.rstrip()+".bmp") for f in file_list]
            if not os.path.exists(gt[0]):
                gt = [os.path.join(gt_path, f.rstrip()+".jpg") for f in file_list]
                    
            gts.append(sorted(gt))
    else:
       gt = [os.path.join(gt_dir, f.rstrip()+ ".png") for f in file_list]
       if not os.path.exists(gt[0]):
           gt = [os.path.join(gt_dir, f.rstrip()+".bmp") for f in file_list]
       if not os.path.exists(gt[0]):
           gt = [os.path.join(gt_dir, f.rstrip()+".jpg") for f in file_list]
       
       gts.append(sorted(gt))

    ret = []
    for gt in gts:
        frames = []
        for f in gt:
            frames.append(rgb2gray(imread(f)))
        ret.append(frames)
    return ret
    
def get_segtrack_frames(name, with_gt=True, gt_only=False):
    with open(os.path.join(segtrackv2_dir, "ImageSets", name+".txt")) as f:
        file_list = f.readlines()[1:]
    video_dir = os.path.join(segtrackv2_dir,"JPEGImages", name)
    frames = [os.path.join(video_dir, f.rstrip()+".png") for f in file_list]
    if not os.path.exists(frames[0]):
        frames = [os.path.join(video_dir, f.rstrip()+".bmp") for f in file_list]
    if not os.path.exists(frames[0]):
        frames = [os.path.join(video_dir, f.rstrip()+".jpg") for f in file_list]
        
    frames = sorted(frames)

    if not with_gt:
        imgs = []
        for f_name in frames:
            frame = imread(f_name)
            imgs.append(frame)
            
        return imgs
    
    gt_dir = os.path.join(segtrackv2_dir,"GroundTruth", name)
    entries = os.listdir(gt_dir)
    gts = []
    if len(entries) < 3 :
       for e in entries:
            gt_path = os.path.join(gt_dir, e)
            gt = [os.path.join(gt_path, f.rstrip()+ ".png") for f in file_list]
            if not os.path.exists(gt[0]):
                gt = [os.path.join(gt_path, f.rstrip()+".bmp") for f in file_list]
            if not os.path.exists(gt[0]):
                gt = [os.path.join(gt_path, f.rstrip()+".jpg") for f in file_list]
                    
            gts.append(sorted(gt))
    else:
       gt = [os.path.join(gt_dir, f.rstrip()+ ".png") for f in file_list]
       if not os.path.exists(gt[0]):
           gt = [os.path.join(gt_dir, f.rstrip()+".bmp") for f in file_list]
       if not os.path.exists(gt[0]):
           gt = [os.path.join(gt_dir, f.rstrip()+".jpg") for f in file_list]
       
       gts.append(sorted(gt))
    
    overlayed = []
    n_gts = len(gts)
    n_frames = len(file_list)
    colors = [(255, 0, 0), (0, 255, 0)]

    for n in range(n_frames):
        img = imread(frames[n])
        r,c,_ = img.shape
        mask_all = np.zeros((r,c,3))
        for (i,gt) in enumerate(gts):
            mask = imread(gts[i][n])
            rgb = mask_to_rgb(mask, colors[i])
            mask_all += rgb
        overlay_mask = make_overlay_mask(img, mask_all)
        overlayed.append(alpha_composite(img, overlay_mask))

    return overlayed

import string
import random

alphabets = string.digits + string.letters

def randstr(n):
    return ''.join(random.choice(alphabets) for i in xrange(n))

def get_flow(im1, im2):
    from skimage.io import imsave
    tmp1 = randstr(10)+'.ppm'
    tmp2 = randstr(10)+'.ppm'
    tmp3 = randstr(10)+'.flo'
    tmp4 = randstr(10)+'.npy'
    tmp5 = randstr(10)+'.png'
    imsave(tmp1, im1)
    imsave(tmp2, im2)
    os.system('%s %s %s %s %s' % (ldof_cpu, tmp1, tmp2, tmp3, tmp4))
    os.system('%s %s %s' % (color_flow, tmp3, tmp5))
    flow = np.load(tmp4)
    flow_img = imread(tmp5)
    for f in [tmp1, tmp2, tmp3, tmp4, tmp5]:
        os.remove(f)
    return flow, flow_img

def o    
def flow_dir(name):
    import os
    
    dir_name = 'data/rgb/' + name
    f_names = [os.path.join(dir_name, f) for f in os.listdir(dir_name)]
    f_names = sorted(f_names)

    cur = f_names[0]

    r,c,_ = imread(f_names[0]).shape
    vx = np.zeros((r,c,len(f_names)-1))
    vy = np.zeros((r,c,len(f_names)-1))
    for (i,nxt) in enumerate(f_names[1:]):
        im1 = imread(cur)
        im2 = imread(nxt)

        flow,img = get_flow(im1,im2)
        imsave("%05d.flo.color.png" % i, img)
        vx[:,:,i] = flow[:,:,0]
        vy[:,:,i] = flow[:,:,1]
        #print cur,nxt
        # command = "%s %s %s %05d.flo %05d.npy" % (ldof_cpu, cur, nxt,  i,i)
        # os.system(command)
        # command = "%s %05d.flo %05d.flo.color.png" % (color_flow,i,i)
        # os.system(command)
        
        cur = nxt

    from scipy.io import savemat
    import os
    
    if not os.paths.exists('data/flow/' + name): os.mkdir('data/flow/' + name)
        
    savemat("data/flow/%s/vx.mat" % name, {'vx':vx})        
    savemat("data/flow/%s/vy.mat" % name, {'vy':vy})        
        
def make_video(dir_name, save_file=None):
    imgs = get_frames(dir_name)
    show_video(imgs, save_file)

def get_frames(dir_name):
    f_names = [os.path.join(dir_name, f) for f in os.listdir(dir_name)]
    f_names = sorted(f_names)
    imgs = [imread(f) for f in f_names]
    return imgs
    
def show_video(frames,save_file=None):
    fig = plt.figure()
    ims = []
    for f in frames:
        ims.append([plt.imshow(f)])
        
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,repeat_delay=1000)
    
    if save_file:
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
#        ani.save(save_file,writer=writer)
        ani.save(save_file,fps=30)
    else:
        plt.show()        

def save_frames(frames):
    from scipy.misc import imsave
    from skimage.transform import resize    
    for (i,f) in enumerate(frames):
        n = i+1
#        f = resize(f, (160,240,3))
        imsave("%05d.png" % n, f)

def get_segtrack_gt(name):
    with open(os.path.join(segtrackv2_dir, "ImageSets", name+".txt")) as f:
        file_list = f.readlines()[1:]
        
    gt_dir = os.path.join(segtrackv2_dir,"GroundTruth", name)
    entries = os.listdir(gt_dir)
    gts = []
    if len(entries) < 3 :
       for e in entries:
            gt_path = os.path.join(gt_dir, e)
            gt = [os.path.join(gt_path, f.rstrip()+ ".png") for f in file_list]
            if not os.path.exists(gt[0]):
                gt = [os.path.join(gt_path, f.rstrip()+".bmp") for f in file_list]
            if not os.path.exists(gt[0]):
                gt = [os.path.join(gt_path, f.rstrip()+".jpg") for f in file_list]
                    
            gts.append(sorted(gt))
    else:
       gt = [os.path.join(gt_dir, f.rstrip()+ ".png") for f in file_list]
       if not os.path.exists(gt[0]):
           gt = [os.path.join(gt_dir, f.rstrip()+".bmp") for f in file_list]
       if not os.path.exists(gt[0]):
           gt = [os.path.join(gt_dir, f.rstrip()+".jpg") for f in file_list]
       
       gts.append(sorted(gt))

    ret = []
    for gt in gts:
        frames = []
        for f in gt:
            frames.append(rgb2gray(imread(f)))
        ret.append(frames)
    return ret

def get_sp_adj(seg):
    uni = np.unique(seg)
    r,c = seg.shape

    adj = np.zeros((len(uni), len(uni)), dtype=np.bool)
    from collections import defaultdict
#    adj = defaultdict(set)
    for j in range(r):
        for i in range(c):
            if i and seg[j,i-1] != seg[j,i]:
                adj[seg[j,i-1],seg[j,i]] = 1
                adj[seg[j,i], seg[j,i-1]] = 1
            if j and seg[j-1,i] != seg[j,i]:
                adj[seg[j-1,i],seg[j,i]] = 1
                adj[seg[j,i],seg[j-1,i]] = 1
                
    return adj

def relabel_job(sp_label):
    count = 0
    r,c = sp_label.shape
    seg = np.zeros((r,c),dtype=np.int)
    mapping = bidict()

    for y in range(r):
        for x in range(c):
            l = sp_label[y,x]
            if l not in mapping.keys():
                seg[y,x] = count
                mapping[l] = count
                count += 1
            else:
                seg[y,x] = mapping[l]
                
    return seg, mapping
    
def relabel(sp_label):
    r,c,n = sp_label.shape

    r = Parallel(n_jobs=-1)(delayed(relabel_job)(sp_label[:,:,i]) for i in range(n))
    segs,mappings = zip(*r)
    
    return segs, mappings
    
def compute_ap(gt, pred):
    score = 0
    for i in range(len(pred)):
        m1 = np.zeros(gt[0].shape)
        m2 = np.zeros(gt[0].shape)
        m1[gt[i].astype(int) == 1] = 1
        m2[pred[i].astype(int) == 1] = 1
        
        score += float(np.sum(np.logical_and(m1, m2))) / np.sum(np.logical_or(m1, m2))

    return score / len(pred)
