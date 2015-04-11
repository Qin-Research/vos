addpath(genpath('external/release/'))

load models/forest/modelBsds.mat
load name.mat

images = dir([['data/rgb/',name, '/'], '*.png']);
im = imread(['data/rgb/' name '/' images(1).name]);
[r,c,ch] = size(im);

edges = zeros(r,c,length(images)-1);
for zz = 1 : length(images) -1

    I = imread(['data/rgb/' name '/' images(zz).name]);
    [E,O,inds,segs] = edgesDetect( I, model );
    edges(:,:,zz) = E;
%    imwrite(E, ['results/' images(zz).name(1:end-4) '.png'], 'png');

end

save('edges.mat', 'edges') 
