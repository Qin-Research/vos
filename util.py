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

