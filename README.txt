Please read my thesis, in particular chap 3 first.
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
 - Currently we depend on the 

3. Pairwise potential

4. Optimization method
