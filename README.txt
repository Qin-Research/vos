[![bmx](http://img.youtube.com/vi/5oVHtXf6JXc/0.jpg)](https://www.youtube.com/watch?v=5oVHtXf6JXc  "bmx")




Please read my thesis, in particular chap 3 first.
And you also want to read Papazoglou et al., ICCV2013, which my method depends on (But I want to remove this dependency).

Read HOWTO.txt to how to run my code.

The directory "data" contains input images, precomputed temporal superpixel and so on.
The directory "external" contains external matlab codes which will be called from my python code.

There are many things to improve on.

1. Video representation
 - The idea of labeling temporal superpixel is not bad.
 - But even if a single tsp trajectory contains both fg and bg pixels, they are given the same label.
 - Some post processing may be needed.

2. Unary potential
 - The idea of diffusion over trajectories is good.
 - To compute fg likelihood, we currently depend on the inside-outside map of Papazoglou et al., ICCV2013. I want to improve it, or replace it with other methods.
   -- Because the inside-outside map is just not robust, esp when there are multiple objects (see my thesis, sec 3.2).
   -- It depends on thresholding optical flow edge, which might ignore weak but important edges.
   -- Each frame is processed completely independently.
 - How to improve the inside-outside map?
   -- Shoot rays in 3d, using TSP.
   -- Other ideas?
 - Final appearance model and location model are not good.
   -- Training random forest requires thresholding diffused inside probability. That sounds just bad.
   -- Location model is borrowed again from Papazoglou et al., ICCV2013. I want to replace this with other, new method. 
   -- Unary potential for a trajectory is computed by averaging unary potential of superpixels on the trajectory. Any better idea? 

3. Pairwise potential
 - First of all, we need to examine if attraction-repulsion potential is doing any good.
   -- Comparison with the standard potts potential may be needed.
 - My original motivations for using attraction-repulsion are:
   -- Unsupervised segmentation -> Unary potential alone is not reliable.
      --- But after I added random forest to unary potential, unary potential became very strong (you can visualise it)
      --- So this motivation is not justified anymore.
   -- Trajectory of superpixels contain much more information than a single pixel or superpixel.
      --- So more informative pairwise energy can be computed, i.e. not just RGB color difference, for example.
 - Currently, affinity between two neighboring trajectories is a weighted combination of color and optical flow distance.
   -- Distance is maximized over frames where two trajectories are adjacent. This may not be good.
      --- I didn't put much thought on this, but maximazition is motivated by 'Segmentation of Moving Objects by Long Term Video Analysis', PAMI 2014, Sec 4.
   -- The length of two trajectories can be incorporated into the affinity.
      --- To make the weight of optical flow edge distance smaller (Color edge is more distinct).
      --- Strong color edge, long trajectory -> this color edge is likely to be inside background.
      --- Strong color edge, short trajectory -> this color edge is likely to be foreground boundary. Affinity should be very small.
      
4. Optimization method
 - If we continue to use attraction-repulsion pair potential, we need to use LSA.
 - If not, graph cut can be used.
 - Is two step optimization really effective?
   -- Grabcut-like effect might be working. Need to examine further.
   -- Two random forests are used, each of which is weighted.
   -- Refined random forest, trained after the first segmentation, should have a larger weight. But it turned out the forest used in the first segmentation is more accurate (Visualise unary potential to see this).
      --- Hence, the first forest has a larger weight (see segmentation.py)
      --- This contradicts my motivation for two step segmentation.

If you have any question, just ask me. My email address is masahi129@gmail.com. 
