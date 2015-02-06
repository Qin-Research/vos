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
deepflow_dir = "/home/masa/research/flow/DeepFlow_release1.0.1"
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

def deep_flow(im1, im2):
    from skimage.io import imsave
    tmp1 = randstr(10)+'.png'
    tmp2 = randstr(10)+'.png'
    tmp3 = randstr(10)+'.flo'
    tmp4 = randstr(10)+'.npy'
    tmp5 = randstr(10)+'.png'
    
    imsave(tmp1, im1)
    imsave(tmp2, im2)

    # os.system('%s %s %s %s %s' % (ldof_cpu, tmp1, tmp2, tmp3, tmp4))
    # os.system('%s %s %s' % (color_flow, tmp3, tmp5))
    import subprocess

    p1 = subprocess.Popen(["%s/deepmatching-static" % deepflow_dir, tmp1, tmp2, "-iccv_settings"], stdout=subprocess.PIPE)
    p2 = subprocess.Popen(["python", "%s/rescore.py" % deepflow_dir, tmp1,tmp2 ], stdin=p1.stdout, stdout=subprocess.PIPE)
    command = "%s/deepflow-static %s %s %s" % (deepflow_dir, tmp1,tmp2,tmp3) +  " -matchf -sintel"
    import shlex    
    p3 = subprocess.Popen(shlex.split(command), stdin=p2.stdout)
    
    p1.stdout.close()  # Allow p1 to receive a SIGPIPE if p2 exits.
    p3.wait()
#    output,err = p3.communicate()

    command = "%s/color %s %s" % (deepflow_dir, tmp3, tmp4)
    print command
    os.system(command)
    os.system('%s %s %s' % (color_flow, tmp3, tmp5))
    flow = np.load(tmp4)
    flow_img = imread(tmp5)
    for f in [tmp1, tmp2, tmp3, tmp4, tmp5]:
         os.remove(f)
    return flow,flow_img

    
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

        flow,img = deep_flow(im1,im2)
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

from pylab import *    
#from gop import *
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
from bidict import bidict

def flow_weight(mag, seg):
    unique = np.unique(seg)
    weight = np.zeros(len(unique))
    for i in unique:
        rows, cols = np.nonzero(seg == i)
        weight[i] = np.sum(mag[rows, cols])

    return weight / weight.sum()

segtrackv2_dir = "/home/masa/research/video_code/SegTrackv2/"

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

def maxima_job(seg,i, r1, sum_mag,adj):
    ids = []

    for j in np.nonzero(adj[i])[0]:
        ids.append(j)
    

    neighbor_ids = np.unique(ids)
    local_maxima = []
    sums = []
    if sum_mag[i] > np.max(sum_mag[neighbor_ids]):
        local_maxima.append(i)
        sums.append(sum_mag[i])
        
    return local_maxima, sums
    
def flow_maxima(seg, mag, num, adj,n_jobs=1):
    uni = np.unique(seg)
    n_region = len(uni)
    sum_mag = np.zeros(n_region)

    
    for (i,r1) in enumerate(uni):
        rows1,cols1 = np.nonzero(seg == r1)
        sum_mag[i] = np.sum(mag[rows1, cols1])

    r = Parallel(n_jobs = n_jobs)(delayed(maxima_job)(seg,i,r1, sum_mag,adj)for (i,r1) in enumerate(uni))
    local_maxima,sums = zip(*r)

    lm = []
    sms = []
    for i in range(1,len(sums)):
        lm += local_maxima[i]
        sms += sums[i]

    local_maxima = lm
    sums = sms
        
#        if local_maxima[i] == False: continue
#            for id in neighbor_ids:
#                local_maxima[id] = False

    index = np.argsort(sums)[-num:]
    return np.array(local_maxima)[index], np.array(sums)[index]

# def superpixel_feature(image,seg,lab_range):
#     uni = np.unique(seg)
#     n = len(uni)
#     dim = 0
#     features = None
#     lab_image = rgb2lab(image)
#     gray = np.pad(rgb2gray(image), (7,7), 'symmetric')

#     n_bins = 20    
#     for (i,region) in enumerate(uni):
#         rows, cols = np.nonzero(seg == region)
#         rgbs = image[rows, cols, :]
#         labs = lab_image[rows, cols,:]
        
#         #feature = np.empty(0)
#         feature = np.mean(rgbs, axis=0)
#         feature = np.concatenate((feature,np.min(rgbs, axis=0)))
#         feature = np.concatenate((feature,np.max(rgbs, axis=0)))
#         # for c in range(3):
#         #     hist, bin_edges = np.histogram(rgbs[:,c], bins=n_bins, range=(0,256),normed=True )
#         #     feature = np.concatenate((feature, hist))
#         # for c in range(3):
#         #      hist, bin_edges = np.histogram(labs[:,c], bins=n_bins, range=(lab_range[c,0], lab_range[c,1]))
#         #      feature = np.concatenate((feature, hist))
#         # center_y = round(np.mean(rows))
#         # center_x = round(np.mean(cols))
#         # patch = gray[center_y:center_y+15, center_x:center_x+15]
#         # hog_feat = hog(patch,orientations=6,pixels_per_cell=(5,5), cells_per_block=(3,3))
#         # feature = np.concatenate((feature, hog_feat))
#         # feature = np.concatenate((feature, np.array([np.mean(rows)/image.shape[0], np.mean(cols)/image.shape[1]])))
#  #       feature = np.concatenate((feature, np.mean(rgbs, axis=0)))
#  #       feature = np.concatenate((feature, np.mean(labs, axis=0)))

#         if features == None:
#             dim = len(feature)
#             features = np.zeros((n, dim))
#             features[0] = feature
#         else:
#             features[i] = feature

#  #   return scale(features)

    
#     return (features)

def superpixel_rgb_mean(image,seg,lab_range):
    uni = np.unique(seg)
    n = len(uni)
    dim = 0
    features = None
    lab_image = rgb2lab(image)
    gray = np.pad(rgb2gray(image), (7,7), 'symmetric')

    n_bins = 20
    mean_rgb = np.zeros((len(uni),3))
    count = np.zeros(len(uni))
    
    for (i,region) in enumerate(uni):
        rows, cols = np.nonzero(seg == region)
        rgbs = image[rows, cols, :]
        
        feature = np.mean(rgbs, axis=0)

        if features == None:
            dim = len(feature)
            features = np.zeros((n, dim))
            features[0] = feature
        else:
            features[i] = feature

 #   return scale(features)
    return (features)

def change_edge_weight(G, feats, id2region):
    rows, cols = G.nonzero()
    values = np.zeros(len(rows))
    for i in range(len(rows)):
        f,n = id2region[rows[i]]
        f2,n2 = id2region[cols[i]]
        values[i] = np.linalg.norm(feats[f][n] - feats[f2][n2])

    from scipy.sparse import csr_matrix
    G_new = csr_matrix((values, (rows, cols)), shape=G.shape)
    return G_new
    

def build_graph(segs, features):
    n_frames = len(segs)

    G = nx.Graph()
    count = 0
    node_ids = []
    id2region = {}
    for n in range(n_frames):
        unique = np.unique(segs[n])
        node_ids.append([])
        for (i,region) in enumerate(unique):
            G.add_node(count)
            node_ids[n].append(count)
            id2region[count] = (n,i)
            count += 1

    G.add_node(count)
    source_id = count
    count += 1
    G.add_node(count)
    terminal_id = count    

    for id in node_ids[0]:
        G.add_edge(source_id, id, weight = 0)
    for id in node_ids[-1]:
        G.add_edge(terminal_id, id, weight = 0)

    masks = []
    masks2 = []
    for (i,region1) in enumerate(np.unique(segs[0])):
            num_edge = 0
            masks.append(segs[0] == i)
    
    for n in range(n_frames-1):
        print "%dth frame" % n
        for (i,region1) in enumerate(np.unique(segs[n])):
            num_edge = 0
            seg1 = masks[i]
            for (j,region2) in enumerate(np.unique(segs[n+1])):
                if i == 0:
                    seg2 = segs[n+1] == j
                else:
                    seg2 = masks2[j]
                if add_edge(i,j,seg1,seg2):
                    distance = np.linalg.norm(features[n][i] - features[n+1][j])
                    G.add_edge(node_ids[n][i], node_ids[n+1][j], weight = distance)
                    num_edge += 1
                if i == 0: masks2.append(seg2)
            print i,num_edge
            
        masks = masks2
        masks2 = []

    return G, source_id, terminal_id, node_ids, id2region

def add_edge(seg1,seg2):
    if np.sum(np.logical_and(seg1,seg2)) > 0: return True
    else: return False

def job(n,segs1, segs2,features1, features2,node_ids1,node_ids2,adjs):
    new_rows = []
    new_cols = []
    new_weights = []

    print n
    for (i,region1) in enumerate(np.unique(segs1)):
        seg1 = segs1 == i
        rows, cols = np.nonzero(seg1)
        tmp = set()
        for (j,region2) in enumerate(np.unique(segs2[rows, cols])):
            tmp.add(region2)
            for neighbor in np.nonzero(adjs[region2])[0]:
               tmp.add(neighbor)

        for j in tmp:
            distance = np.linalg.norm(features1[i] - features2[j])
            new_rows.append(node_ids2[j])                                
            new_cols.append(node_ids1[i])
            new_weights.append(distance)
                
    return new_rows, new_cols, new_weights
            
def build_csgraph(segs, features, adjs, rad, n_jobs=1):
    n_frames = len(segs)

    count = 0
    node_ids = []
    id2region = {}
    for n in range(n_frames):
        unique = np.unique(segs[n])
        node_ids.append([])
        for (i,region) in enumerate(unique):
            node_ids[n].append(count)
            id2region[count] = (n,i)
            count += 1

    source_id = count
    count += 1
    terminal_id = count
    count += 1

    height, width = segs[0].shape
    rows = []
    cols = []
    weights = []
    
    for id in node_ids[0]:
        rows.append(id)
        cols.append(source_id)        
        weights.append(0)
        
    for id in node_ids[-1]:
        rows.append(terminal_id)
        cols.append(id)
        weights.append(0)

    y,x = np.ogrid[0:height, 0:width]
    weighting = [1,1.5,2] 
    for n in range(n_frames-1,0,-1):        
        for (i,region1) in enumerate(np.unique(segs[n])):
            seg1 = segs[n] == i
            rs, cs = np.nonzero(seg1)

            center_y = np.mean(rs)
            center_x = np.mean(cs)

            lhs = (x - center_x) ** 2 + (y - center_y) ** 2
            for ii in range(1,4):
                if n-ii < 0: break

                j = n - ii
                r = rad * ii

                mask = lhs < r**2
                in_mask = np.unique(segs[j][mask])

                for id in in_mask:
                    distance = np.linalg.norm(features[n][i] - features[j][id])
                    cols.append(node_ids[j][id])                                
                    rows.append(node_ids[n][i])
                    weights.append(weighting[ii-1] * distance)
                
    # r = Parallel(n_jobs=n_jobs)(delayed(job)(n, segs[n], segs[n+1],features[n], features[n+1],node_ids[n], node_ids[n+1],adjs[n+1] ) for n in range(n_frames-1))
    # r,c,w = zip(*r)
    # for i in range(len(r)):
    #     rows += r[i]
    #     cols += c[i]
    #     weights += w[i]    
                    
    from scipy.sparse import csr_matrix
    G = csr_matrix((weights, (rows, cols)), shape=(count, count))
    return G,source_id, terminal_id, node_ids, id2region

def shortest_path(G,start_index,source_id, terminal_id):
    from scipy.sparse.csgraph import dijkstra
    distances, predecessors = dijkstra(G, directed=True, indices=start_index,return_predecessors=True)
    paths = []
    for (p, pred) in enumerate(predecessors):
        path = []
        i = pred[source_id]
        while i != start_index[p]:
            if i == terminal_id:
                print 'Passed terminal node. Something Wrong.'
            path.append(i)
            i = pred[i]
        path.append(i)
        paths.append(path)

    distances, predecessors = dijkstra(G.T, directed=True, indices=start_index,return_predecessors=True)
    paths2 = []
    for (p, pred) in enumerate(predecessors):
        path = []
        i = pred[terminal_id]
        while i != start_index[p]:
            if i == terminal_id:
                print 'Passed terminal node. Something Wrong.'
            path.append(i)
            i = pred[i]
        path.append(i)
        paths2.append(path)
                
    return paths,paths2

def get_lab_range(frames):
    l_min = np.inf
    a_min = np.inf
    b_min = np.inf
    l_max = -np.inf
    a_max = -np.inf
    b_max = -np.inf
    
    for (ii,im) in enumerate(frames):
        lab = rgb2lab(imread(im))
        l_min = min(l_min, np.min(lab[:,:,0]))
        a_min = min(a_min, np.min(lab[:,:,1]))
        b_min = min(b_min, np.min(lab[:,:,2]))
        l_max = max(l_max,np.max(lab[:,:,0]))
        a_max = max(a_max, np.max(lab[:,:,1]))
        b_max = max(b_max, np.max(lab[:,:,2]))
    
    lab_range = np.array([[l_min,l_max],
                          [a_min,a_max],
                          [b_min,b_max]])

    return lab_range
            

def edges2adj(edges,n_sp):
    n_edge = len(edges)
    adj = np.zeros((n_sp, n_sp), dtype=np.bool)
    for i in range(n_edge):
        adj[edges[i].a, edges[i].b] = 1
        adj[edges[i].b, edges[i].a] = 1
    return adj

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


def relabel(sp_label):
    segs = []
    mappings = []
    r,c,n = sp_label.shape
    
    for i in range(n):
        count = 0
        seg = np.zeros((r,c),dtype=np.int)
        mapping = bidict()
        for y in range(r):
            for x in range(c):
                l = sp_label[y,x,i]
                if l not in mapping.keys():
                    seg[y,x] = count
                    mapping[l] = count
                    count += 1
                else:
                    seg[y,x] = mapping[l]
                    
        mappings.append(mapping)
        segs.append(seg)

    return segs, mappings
        
                
                
def get_tsp(sp_label):
    sps,mappings = relabel(sp_label)
    return sps,mappings

def get_tsp2(sp_label):
    sps,mappings = relabel(sp_label)
    adjs = []
#    from skimage.segmentation import relabel_sequential
    for n in range(sp_label.shape[2]):
         adjs.append(get_sp_adj(sps[n]))

    return sps, adjs,mappings

    
        
def video_superpixel(frames,detector):
    segs = []
    sp_adjs = []
    for (ii,im) in enumerate(frames):
        s = segmentation.geodesicKMeans( imgproc.imread( im ), detector, 1000 )
        segs.append(np.array(s.s))
        sp_adjs.append(edges2adj(s.edges, len(np.unique(segs[-1]))))

    return segs,sp_adjs
        
# def get_sp_feature_all_frames(frames, segs, lab_range):
#     feats = []
#     from skimage import img_as_ubyte
#     for (ii,im) in enumerate(frames):
# #        features = superpixel_feature((imread(im)), segs[ii], lab_range)
#         features = superpixel_feature(img_as_ubyte(imread(im)), segs[ii], lab_range)
#         feats.append(features)
        
#     return feats

def get_sp_rgb_mean_all_frames(frames, segs, lab_range):
    feats = []
    from skimage import img_as_ubyte
    for (ii,im) in enumerate(frames):
        features = superpixel_rgb_mean(img_as_ubyte(imread(im)), segs[ii], lab_range)
        feats.append(features)
        
    return feats

def get_flow_local_maxima(frames, segs, node_ids,vx, vy,adjs,num_per_frame, total):
    sms = []
    ids = []

    from collections import defaultdict
    
    for (ii,im) in enumerate(frames):
        flow_mag = np.sqrt(vx[:,:,ii] ** 2 + vy[:,:,ii] ** 2)
        r,c, = flow_mag.shape    

        if type(adjs[0]) == defaultdict:
            sum_mag = dict()
            labels = []

            loc_mx = []
            loc_mx_values = []
            for u in np.unique(segs[ii]):
                rows, cols = np.nonzero(segs[ii] == u)
                sum_mag[u] = np.sum(flow_mag[rows, cols])
                labels.append(u)

            for (i,l) in enumerate(labels):
                value = sum_mag[l]
                ok = True
                for neighbor in adjs[ii][u]:
                    if sum_mag[neighbor] > value:
                        ok = False
                        break
                if ok:
                    loc_mx.append(l)
                    loc_mx_values.append(value)
                
        else:
            maxima,sums = flow_maxima(segs[ii], flow_mag, num_per_frame,adjs[ii],8)
            for m in maxima:
                ids.append(node_ids[ii][m])
            sms = np.concatenate((sms, sums))
            
        # index = np.argsort(sms)
        # local_maxima = np.array(ids)[index[-20:]]
        local_maxima = np.random.choice(ids, size=total, replace=False,p=sms/np.sum(sms))
        return local_maxima
    
def plot_flow_mag(frames,segs,vx,vy):
    for (ii,im) in enumerate(frames):
        r,c = segs[ii].shape
        mag_image = np.zeros((r,c))
        flow_mag = np.sqrt(vx[:,:,ii] ** 2 + vy[:,:,ii] ** 2)
        unique = np.unique(segs[ii])
        for i in unique:
           rows, cols = np.nonzero(segs[ii] == i)
           mag_image[rows, cols] = np.sum(flow_mag[rows, cols]/ len(rows))
        print ii
        imshow(flow_mag)
        show()

def flow_unary(frames, segs, vx,vy):
    fg_prob = []
    rescaled = []
    for (ii,im) in enumerate(frames):
        r,c = segs[ii].shape
        mag_image = np.zeros((r,c))
        flow_mag = np.sqrt(vx[:,:,ii] ** 2 + vy[:,:,ii] ** 2)
        unique = np.unique(segs[ii])
        p = []
        for i in unique:
           rows, cols = np.nonzero(segs[ii] == i)
           p.append(np.sum(flow_mag[rows, cols])/ len(rows))
           rescaled.append(np.sum(flow_mag[rows, cols])/ len(rows))

        mx = np.max(p)
        for pp in p:
           fg_prob.append(pp/mx)

    fg_prob = np.asarray(fg_prob)[:,np.newaxis]
    unnom = np.asarray(rescaled).copy()
    rescaled = np.asarray(rescaled) / np.max(rescaled)
    unary = -np.log(np.hstack((fg_prob, 1 - fg_prob)) + 0.00001)
    return unary,fg_prob,rescaled,unnom
            
def plot_sp(im, seg, n):
    rgb = np.array(imgproc.imread(im))
    rgb[seg == n] = (255,0,0)
    imshow(rgb)

    
def plot_paths(p, paths,frames, segs,id2region, vx, vy):
    for (iter, id) in enumerate(paths[p]):
        f,sp = id2region[id]
        print f
        img = np.zeros(segs[f].shape)
        unique = np.unique(segs[f])
        r,c = img.shape
        mag_image = np.zeros((r,c))
        flow_mag = np.sqrt(vx[:,:,f] ** 2 + vy[:,:,f] ** 2)        
        for i in unique:
           rows, cols = np.nonzero(segs[f] == i)
           mag_image[rows, cols] = np.sum(flow_mag[rows, cols])
        
    
        img[segs[f] == sp] = True
 
        rgb = np.array(imgproc.imread(frames[f]))

        figure(figsize=(20,15))
        rgb[segs[f] == sp] = (255,0,0)
        subplot(1,2,1)
        imshow(np.array(imgproc.imread(frames[f])))
        axis('off')
        # subplot(1,3,2)
        # imshow(flow_mag)
        # axis('off')
        subplot(1,2,2)
        imshow(rgb)
        axis('off')
        savefig('%05d.png' % iter)
        show()

def plot_local_maxima(id2region, segs, vx,vy,local_maxima):
    r,c = segs[0].shape

    path_image = []
    for i in range(len(segs)):
        path_image.append(np.zeros((r,c),dtype=np.bool))

    for p in local_maxima:
        f,region = id2region[p]
        path_image[f][segs[f] == region] = True

    for i in range(len(segs)):
        
        mag_image = np.zeros((r,c))
        flow_mag = np.sqrt(vx[:,:,i] ** 2 + vy[:,:,i] ** 2)
        unique = np.unique(segs[i])
        
        for j in unique:
           rows, cols = np.nonzero(segs[i] == j)
           mag_image[rows, cols] = np.sum(flow_mag[rows, cols])
        print i
        subplot(1,2,1)
        imshow(mag_image)
        axis('off')
        # subplot(1,3,2)
        # imshow(flow_mag)
        # axis('off')
        subplot(1,2,2)
        imshow(path_image[i],cmap=gray())
        axis('off')
        
        show()

        
           
            
def plot_all_paths(frames,segs,paths,id2region):
    im = imread(frames[0])
    r,c,_ = im.shape

    path_image = []
    for i in range(len(frames)):
        path_image.append(np.zeros((r,c),dtype=np.bool))

    for p in paths:
        for id in p:
            f,region = id2region[id]
            path_image[f][segs[f] == region] = True

    for i in range(len(frames)):
        print i
        subplot(1,2,1)
        imshow(np.array(imgproc.imread(frames[i])))
        axis('off')
        # subplot(1,3,2)
        # imshow(flow_mag)
        # axis('off')
        subplot(1,2,2)
        imshow(path_image[i],cmap=gray())
        axis('off')
        
        show()
        
def feats2mat(feats):
    ret = feats[0]
    for feat in feats[1:]:
        ret = np.vstack((ret, feat))
    return ret


def get_gop(frames,prop, detector,vx,vy):

    for f in frames:
        im = imgproc.imread(f)
        flow_mag = np.sqrt(vx[:,:,f] ** 2 + vy[:,:,f] ** 2)
        s = segmentation.geodesicKMeans( im, detector, 1000 )

        
        # for i in unique:
        #     rows, cols = np.nonzero(s.s == i)
        #     mag_image[rows, cols] = np.sum(flow_mag[rows, cols])
    
        mx = np.zeros(s.s.shape,dtype = np.bool)
        for m in maxima:
            mx[s.s == m] = True
        subplot(1,2,1)
        imshow(mx)
        subplot(1,2,2)
        imshow(mag_image)
        show()
        continue
        
        weight = flow_weight(flow_mag, s.s)
        seeds = np.random.choice(np.arange(s.Ns), replace=False, size=50, p=weight).astype(np.int32)
        ret = prop.propose2(s,seeds)
    #    b = prop.propose2(s,seeds)
        b = ret[:, 1:].astype(np.bool)
        seed_index = ret[:,0]
    #    b = prop.propose(s)
        t2 = time()
    	# If you just want the boxes use
        print( "Generated %d proposals in %0.2fs (OverSeg: %0.2fs, Prop: %0.2fs)"%(b.shape[0],t2-t0,t1-t0,t2-t1) )
        figure()
        print ii
    
        if not os.path.exists(str(ii)): os.mkdir(str(ii))
        sum_err = 0
    
        if ii == 0:
            masks = np.empty((r,c,b.shape[0]),dtype=np.ubyte)
    
        seed_mask = np.zeros((r,c))
        for seed in seeds:
            seed_mask[s.s == seed] = 1
            
        count = np.zeros((r,c))
        
        
        for i in range(b.shape[0]):
            if seed_index[i] == -1: continue
            im = np.array( s.image )
            mask = np.zeros((im.shape[:-1]), dtype=np.ubyte)
            mask[b[i,s.s]] = 1
            if np.sum(mask) > r*c/2: continue
            count += mask
            err1 = np.sum(np.logical_xor(gt[0][ii],mask))
            err2 = np.sum(np.logical_xor(gt[1][ii],mask))
            err = err1
            if err2 < err1: err = err2
            sum_err += err
    
            if ii == 0:
                masks[:,:, i] = mask
            im[ b[i,s.s] ] = (255,0,0)
            name = os.path.join(str(ii), str(i)+'.png')
            imsave(name, im)
    
        # for ind in np.unique(seed_index):
        #     if ind != -1: continue
        #     m = b[seed_index == ind]
        #     print ind
        #     for mm in m:
        #        mask = np.zeros((im.shape[:-1]), dtype=np.ubyte)
        #        mask[mm[s.s]] = 1
        #        imshow(mask, cmap=gray())
        #        show()
            
            
        avg_xor.append(float(sum_err) / b.shape[0])
        # subplot(1,3,1)
        # imshow(count,cmap=jet())
        # colorbar()
        # subplot(1,3,2)
        # imshow(seed_mask,cmap=gray())
        # subplot(1,3,3)
        imshow(mag_image,cmap=jet())
        colorbar()
        show()
        imshow(im)
        show()
        
def pdist_helper(condensed,n,index):
    from itertools import combinations

    distance = np.zeros(n)
    distance[index] = 0
    count = 1
    for (i,pair) in enumerate(combinations(range(n),2)):
        if index in pair:
            if pair[0] == index: distance[pair[1]] = condensed[i]
            else: distance[pair[0]] = condensed[i]
            count += 1
        if count == n: return distance


def plot_nearest_neighbor(condensed, n, index,frames, segs,id2region,n_neighbors):
    
    distance = pdist_helper(condensed, n, index)
    order = np.argsort(distance)[1:]

 #   f,region = id2region[index]
#    plot_sp(frames[f], segs[f], region)
#    show()
    for neighbor in range(n_neighbors):
        f,region = id2region[order[neighbor]]
        print f
        figure(figsize=(20, 15))        
        subplot(1,2,1)
        plot_sp(frames[f], segs[f], region)
        axis('off')
        subplot(1,2,2)
        f,region = id2region[index]        
        plot_sp(frames[f], segs[f], region)        
        axis('off')
        savefig(str(neighbor) + '.png')
        show()    

def plot_seg(frames, segs, label, id2region):
    im = imread(frames[0])
    r,c,_ = im.shape

    path_image = []
    for i in range(len(frames)):
        path_image.append(np.zeros((r,c),dtype=np.bool))

    for (i,l) in enumerate(label):
        if l == 0:
            (f,region) = id2region[i]
            path_image[f][segs[f] == region] = 1
            
    for i in range(len(frames)):
        figure(figsize=(20,15))
        subplot(1,2,1)
        imshow(imread(frames[i]))
        axis('off')

        subplot(1,2,2)
        imshow(path_image[i],cmap=gray())
        axis('off')
        show()

def plot_seg2(frames, sp_label, label):
    im = imread(frames[0])
    r,c,_ = im.shape

    path_image = []
    for i in range(len(frames)):
        path_image.append(np.zeros((r,c),dtype=np.bool))

            
    for i in range(len(frames)):
        for l in label:
            path_image[i][sp_label[:,:,i] == l] = 1
            
        figure(figsize=(20,15))
        subplot(1,2,1)
        imshow(imread(frames[i]))
        axis('off')

        subplot(1,2,2)
        imshow(path_image[i],cmap=gray())
        axis('off')
        show()
        
def plot_tsp_path(frames, tsp,label):
    tsp == label
    for f in range(tsp.shape[2]-1):
        figure(figsize=(12,9))
        plot_sp(frames[f], tsp[:,:,f], label)
        axis("off")
        show()

def propagate_flow(frames, tsp, vx,vy):                
#        imshow(tsp[:,:,f] == label)
    from collections import defaultdict
    max_flow = defaultdict(float)
    
    for f in range(len(frames)):
        flow_mag = np.sqrt(vx[:,:,f] ** 2 + vy[:,:,f] ** 2)
        for l in np.unique(tsp[:,:,f]):
            rows, cols = np.nonzero(tsp[:,:,f] == l)
            sum_flow = np.sum(flow_mag[rows, cols])
            max_flow[l] = max(max_flow[l], sum_flow)
            
    for f in range(len(frames)):
        img = np.zeros(tsp[:,:,0].shape)
        print f
        for l in np.unique(tsp[:,:,f]):
            img[tsp[:,:,f] == l] = max_flow[l]
        imshow(img)
        axis("off")
        show()
                             
def fgbg_mining(frames, sp_label, vx,vy, fg_num, bg_num,vis=False):

    frame_id = []
    sp_id = []
    flow_sum = []
    mag_images = []
    r,c = sp_label[:,:,0].shape    
    for (ii,im) in enumerate(frames):

        mag_image = np.zeros((r,c))
        flow_mag = np.sqrt(vx[:,:,ii] ** 2 + vy[:,:,ii] ** 2)
        unique = np.unique(sp_label[:,:,ii])
        for i in unique:
           rows, cols = np.nonzero(sp_label[:,:,ii] == i)
           flow_sum.append(np.sum(flow_mag[rows, cols]))
           frame_id.append(ii)
           sp_id.append(i)
           mag_image[rows, cols] = np.sum(flow_mag[rows, cols])
        mag_images.append(mag_image)           

    order = np.argsort(flow_sum)
    fg_id = order[-fg_num:]
    bg_id = order[:bg_num]

    fg = (np.array(frame_id)[fg_id], np.array(sp_id)[fg_id])
    bg = (np.array(frame_id)[bg_id], np.array(sp_id)[bg_id])

    if vis:
        for i in range(len(frames)):
            fg_mask = np.zeros((r,c),dtype=np.bool)
            bg_mask = np.zeros((r,c),dtype=np.bool)
            for j in range(len(fg_id)):
                if fg[0][j] == i:
                    fg_mask[sp_label[:,:, i] == fg[1][j]] = True

            for j in range(len(bg_id)):
                if bg[0][j] == i:
                    bg_mask[sp_label[:,:, i] == bg[1][j]] = True
                                        
            figure(figsize=(20,15))

            subplot(1,3,1)
            imshow(mag_images[i])
            axis('off')
                
            subplot(1,3,2)
            imshow(fg_mask,cmap=gray())
            axis('off')

            subplot(1,3,3)
            imshow(bg_mask,cmap=gray())
            axis('off')
            show()
                
                     
    return fg,bg,flow_sum
    

def get_dominant_angle(angle):
    hist,bins = np.histogram(angle.flatten(), bins=20)
    return bins[np.argmax(hist)]

def get_dominant_motion(motion):
    hist,bins = np.histogram(motion.flatten(), bins=20)
    return bins[np.argmax(hist)]


def get_unary(frames, segs, saliency, sal_thres):

    n_frames = len(frames)
    fg_unary = []
    bg_unary = []
    
    for i in range(n_frames):
        uni = np.unique(segs[i])
        for u in uni:
            mean_sal = np.mean(saliency[:,:,i][segs[i] == u])
            fg_unary.append(-mean_sal)
            bg_unary.append(-(sal_thres - mean_sal))

    unary = np.hstack((np.array(fg_unary, np.float32)[:,np.newaxis], 
                       np.array(bg_unary, np.float32)[:,np.newaxis]))

    return unary


