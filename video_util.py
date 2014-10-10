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

segtrackv2_dir = "/home/masa/research/video_code/SegTrackv2/"
video_segment_dir = "/home/masa/research/video_code/video_segment/"
converter = video_segment_dir+"converter/segment_converter"
seg_tree = video_segment_dir + "build/seg_tree_sample"
ldof_cpu = "/home/masa/research/flow/pami2010LibLinux64/demo_ldof"
color_flow = '/home/masa/research/code/flow_util/color_flow'

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

    
def flow_dir(dir_name):
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
    savemat("vx.mat", {'vx':vx})        
    savemat("vy.mat", {'vy':vy})        
        
def job(i,cur,out_base, f_names):
    n_frame = len(f_names)
    from os.path import join            
    out_dir = join(out_base, str(i))
    if not os.path.exists(out_dir): os.mkdir(out_dir)

    for d in range(-5, 6):
        if i + d < 0 or d == 0 or i + d >= n_frame:continue
        if d < 0:
            im1 = imread(f_names[i+d])
            im2 = imread(cur)
        else:
            im1 = imread(cur)
            im2 = imread(f_names[i+d])

        flow, color = get_flow(im1, im2)
        out_path1 = join(out_dir, str(d)+".flo.npy")
        out_path2 = join(out_dir, str(d)+".flo.png")
        np.save(out_path1, flow)
        imsave(out_path2, color)
        
def flow_dir2(dir_name, out_base):
    f_names = [os.path.join(dir_name, f) for f in os.listdir(dir_name)]
    f_names = sorted(f_names)

    n_frame = len(f_names)
    from os.path import join
    if not os.path.exists(out_base):os.mkdir(out_base)


    from joblib import delayed, Parallel
    import time
    t = time.time()
    Parallel(n_jobs = 8)(delayed(job)(i, cur, out_base, f_names) for (i,cur) in enumerate(f_names[:-1]))
    print time.time() - t
        
    # for (i,cur) in enumerate(f_names[:-1]):
    #     out_dir = join(out_base, str(i))
    #     if not os.path.exists(out_dir): os.mkdir(out_dir)
            
    #     for d in range(-5, 6):
    #         if i + d < 0 or d == 0 or i + d >= n_frame:continue
    #         if d < 0:
    #             im1 = imread(f_names[i+d])
    #             im2 = imread(cur)
    #         else:
    #             im1 = imread(cur)
    #             im2 = imread(f_names[i+d])

    #         flow, color = get_flow(im1, im2)
    #         out_path1 = join(out_dir, str(d)+".flo.npy")
    #         out_path2 = join(out_dir, str(d)+".flo.png")
    #         np.save(out_path1, flow)
    #         imsave(out_path2, color)
            
            

        # print cur,nxt
        # command = "%s %s %s %05d.flo %05d.npy" % (ldof_cpu, cur, nxt,  i,i)
        # os.system(command)
        # command = "%s %05d.flo %05d.flo.color.png" % (color_flow,i,i)
        # os.system(command)                
        # cur = nxt
    
    
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

        
def graph_segment(video_file,
                  output_dir,
                  out_base="output/",
                  seg_tree = seg_tree,
                  converter = converter):

    if not os.path.exists(out_base): os.mkdir(out_base)
    os.system("%s --input_file=%s" % (seg_tree, video_file))
    output = video_file + ".pb"
    out = out_base + output_dir
    if not os.path.exists(out): os.mkdir(out)
    os.system("%s --input=%s --output_dir=%s" % (converter, output, out))

    for d in os.listdir(out):
        rgb_dir = os.path.join(out, d, "rgb")
        id_dir = os.path.join(out, d, "id")
        id_rgb_dir = os.path.join(out, d, "id_rgb")
        if not os.path.exists(rgb_dir): os.mkdir(rgb_dir)
        if not os.path.exists(id_dir): os.mkdir(id_dir)
        if not os.path.exists(id_rgb_dir): os.mkdir(id_rgb_dir)

        path = os.path.join(out, d)
        for f in os.listdir(path):
            if f.endswith(".txt"):
                id_file = os.path.join(path, f)
                img = np.loadtxt(id_file)
                imsave(os.path.join(id_rgb_dir, f+".png"),img)
                shutil.move(id_file, id_dir)
            elif f.endswith(".png"):
                rgb_file = os.path.join(path, f)                
                shutil.move(rgb_file, rgb_dir)

                
def load_supervoxel(dir_name):
    files = sorted(os.listdir(dir_name))
    files = [os.path.join(dir_name, f) for f in files]
    tmp = np.loadtxt(files[0])
    n_frames = len(files)
    rows, cols = tmp.shape
    supervoxel = np.empty((rows, cols, n_frames), dtype=np.int)                  
    for (i,f) in enumerate(files):
        supervoxel[:,:, i] = np.loadtxt(f)

    unique = np.unique(supervoxel)
    if np.max(unique) != len(unique):
        supervoxel = my_relabel_sequential(supervoxel)    
    return supervoxel

def supervoxel_to_superpixel(sv):

    frames = sv.shape[2]
    superpixels = np.empty(sv.shape)

    offset = 0
    for i in range(frames):
        frame = sv[:,:,i]
        superpixels[:,:,i] = frame + offset
        offset = np.max(superpixels[:,:,i])+1
        
    return superpixels
        
        
def my_relabel_sequential(sv):
    uni = np.unique(sv)
    n = len(uni)

    for i in range(n):
        if uni[i] == i: continue 
        sv[sv == uni[i]] = i
        
    return sv

def supervoxel_neighbor(sv):
    labels = np.unique(sv)
    n_labels = labels.shape[0]
    adj_mat = np.zeros((n_labels, n_labels))

    w,h,d = sv.shape

    for i in range(w):
        for j in range(h):
            for k in range(d):
                l = sv[i,j,k]

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        for dz in [-1, 0,1]:
                            ii = i + dx
                            jj = j + dy
                            kk = k + dz

                            if ii < 0 or ii >= w: continue
                            if jj < 0 or jj >= h: continue
                            if kk < 0 or kk >= d: continue

                            ll = sv[ii,jj,kk]
                            if l != ll:
                                adj_mat[l, ll] = 1 
                                adj_mat[ll, l] = 1
    return adj_mat


def flow_unary(sv, flow_mag):
    n = len(np.unique(sv))
    unary = np.empty((n, 2))

    for i in range(n):
        row, col, depth = nonzero(sv[:,:,:-1] == i)
        flow_values = flow_mag[row, col, depth]
        # hist(flow_values)
        # show()
        if len(flow_values) == 0:
            unary[i, 0] = 0
            continue
        mx = np.max(flow_values)
        unary[i, 0] = mx

    tmp = unary[:, 0].copy()
    max_median = np.max(unary[:,0])
    print 
    unary[:,0] /= max_median
    unary[:,1] = 1 - unary[:,0]    
    # epsilon = 1e-5
    # unary[:,0] = -np.log(epsilon + unary[:,0])
    
    return tmp, unary

def foo(sv, flow_mag):
    height, width, n_frames = sv.shape    
    seg = np.empty((height, width, n_frames-1))
    for i in range(n_frames-1):
        uni = np.unique(sv[:,:,i])
        n = len(np.unique(uni))

        unary = np.empty((height,width,2))

        for (c,j) in enumerate(uni):
            row, col = nonzero(sv[:,:,i] == j)
            flow_values = flow_mag[row, col, i]
            # hist(flow_values)
            # show()
            mx = np.median(flow_values)
            unary[row,col, 0] = mx
    
        max_median = np.max(unary[:,:,0])

        unary[:,:,0] /= max_median
        seg[:,:,i] = unary[:,:,0] > 0.5

    # epsilon = 1e-5
    # unary[:,0] = -np.log(epsilon + unary[:,0])
    
    return seg


def supervoxel_pairwise(frames, sv, adj):
    n = adj.shape[0]

    n_edges = np.sum(adj) / 2

    pair_cost = np.empty(n_edges)
    pair_index = np.empty((n_edges,2), dtype=np.int)

    c = 0
    for i in range(n):
        for j in range(n):
            if i > j: continue
            pair_cost[c] = 0
            pair_index[c] = [i,j]
            c += 1

    return pair_cost, pair_index
    
# def plot_frames(frames):
#     n = len(frames)
#     n_frames = len(frames[0])
#     from pylab import *
#     for i in range(n_frames):
#         for j in range(n):
#           subplot(1,n,j)
#           imshow(l[:,:, i], cmap=gray())
#      subplot(1,3,2)
#      imshow(rgb[i])
#      subplot(1,3,3)
#      if i == l.shape[2]-1:
#        imshow(flow_rgb[-1])
#      else:
#        imshow(flow_rgb[i])
#      show()
    
