from pylab import *
import numpy as np

def prob_to_image(prob,paths,segs):
    id2index = []
    index = 0

    r,c = segs[0].shape

    image = np.zeros((r,c,len(segs)))
    
    for i in range(len(prob)):
        id2index.append({})
        for (jj,j) in enumerate(prob[i]):
            id2index[i][jj] = index
            index += 1
    
    for (i,id) in enumerate(paths.keys()):
        frame = paths[id].frame
        rows = paths[id].rows
        cols = paths[id].cols
    
        unique_frame = np.unique(frame)
    
        for f in unique_frame:
            r = rows[frame == f]
            c = cols[frame == f]
            image[r,c,f] = prob[f][segs[f][r[0],c[0]]]

    return image

def plot_paths_value(paths, sp_label, values,cm):

    val = np.zeros(sp_label.shape)
    for (i,id) in enumerate(paths.keys()):
        val[paths[id].rows, paths[id].cols, paths[id].frame] = values[i]

    for i in range(sp_label.shape[2]):
        imshow(val[:,:,i], cm)
        show()

    return val
    
def plot_affinity2(affinity, frames, sp_label, paths, id_mapping, id_mapping2):

    r,c,n_frame = sp_label.shape
    aff = np.ones((r,c,n_frame)) * inf

    for k in range(n_frame):
        for j in range(c):
            for i in range(r):
               l = sp_label[i,j,k]
               index = id_mapping[l]
               
               if i > 0:
                   ll = sp_label[i-1,j,k]
                   if l != ll:
                       aff[i,j,k] = affinity[index][id_mapping[ll]]

               if i < sp_label.shape[0]-1:
                   ll = sp_label[i+1,j,k]

                   if l != ll:
                       aff[i,j,k] = affinity[index][id_mapping[ll]]                       
                       
               if j > 0:
                   ll = sp_label[i,j-1,k]

                   if l != ll:
                       aff[i,j,k] = affinity[index][id_mapping[ll]]                       
                                              
               if j < sp_label.shape[1] -1:
                   ll = sp_label[i,j+1,k]

                   if l != ll:
                       aff[i,j,k] = affinity[index][id_mapping[ll]]                       



    for i in range(sp_label.shape[2]):

        figure(figsize(21,18))
        im = imread(frames[i])

        subplot(1,2,1)
        imshow(im)
        axis("off")

        subplot(1,2,2)
        imshow(aff[:,:,i],jet())
        axis("off")
        colorbar()

        show()
                                       
    return aff
