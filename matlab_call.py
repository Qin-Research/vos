from scipy.io import loadmat,savemat
import os
import numpy as np

def struct_edge_detect(name):
    savemat('name.mat', {'name':name})
    os.system("matlab -nodisplay -nojvm -nosplash < matlab_func/edge_dir.m");
    
    edges = loadmat('edges_%s.mat' % name)['edges']
    return edges

def compute_flow_edge(name):
    savemat('name.mat', {'name':name})
    os.system("matlab -nodisplay -nojvm -nosplash < matlab_func/flow_edge.m");
    
    edges = loadmat('flow_edges_%s.mat' % name)['boundaryMaps']
    return edges

def compute_inprob(name,segs):
    savemat('name.mat', {'name':name})
    n = len(segs)
    r,c = segs[0].shape
    sp = np.zeros((r,c,n),dtype=np.int)
    for i in range(n): sp[:,:,i] = segs[i]

    savemat('sp_%s.mat' % name, {'superpixels':sp})
    os.system("matlab -nodisplay -nojvm -nosplash < matlab_func/inprobs.m");
    
    p = loadmat('inprobs_%s.mat' % name)['inRatios']
    inprobs = []

    for i in range(n-1):
        inprobs.append(p[i][0].flatten())

    inprobs.append(np.zeros(len(np.unique(segs[-1])))) #last frame has no optical flow avaliable, so inprob is zero 
    
    return inprobs

def compute_locprior(name, segs, diffused_prob):
    savemat('name.mat', {'name':name})
    n = len(segs)
    r,c = segs[0].shape
    sp = np.zeros((r,c,n),dtype=np.int)
    for i in range(n): sp[:,:,i] = segs[i]

    savemat('sp_%s.mat' % name, {'superpixels':sp})
    savemat('diffused_%s.mat' % name, {'diffused_inprobs':diffused_prob})
    os.system("matlab -nodisplay -nojvm -nosplash < matlab_func/compute_locprior.m");
    
    locprior = loadmat('locprior_%s.mat' % name)['locationUnaries']
    return locprior

def optimize_lsa(unary,pairwise, segs,paths):

    savemat('energy.mat', {'UE': unary.transpose(), 'PE':pairwise})    
    os.system("matlab -nodisplay -nojvm -nosplash < matlab_func/optimize.m");
    labels = loadmat('labeling.mat')['labels']
    count = 0
    mask = []
    r,c = segs[0].shape
    mask_label = np.ones((r,c,len(segs))) * 0.5

    for (i,path) in enumerate(paths.values()):
        if labels[i][0] == 0:
            mask_label[path.rows, path.cols, path.frame] = 1
        else:
            mask_label[path.rows, path.cols, path.frame] = 0            
            
    for j in range(len(segs)):
        mask.append(mask_label[:,:,j])
    
    return mask,labels
