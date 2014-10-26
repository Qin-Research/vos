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

# def get_feature_for_pairwise(frames, segs, adjs,lab_range):
#     features = feats2mat(get_sp_feature_all_frames(frames, segs, lab_range))
#     new_features = np.zeros(features.shape)

#     return features
    # for i in range(n_frames):
    #     uni = np.unique(segs[i])
    #     for u in uni:
    #         for (id,a) in enumerate(adjs[i][u]):
    #             if a == False:continue
