load name.mat
addpath(genpath('external/FastVideoSegment/'))
flow_file = ['flow_' name '.mat'];
load(flow_file)
load sp.mat
%addpath('modified')
addpath('matlab_func/modified')
frames = size(superpixels,3) - 1;
sp = cell(frames, 1);

for i =1:frames
    sp{i} = uint16(superpixels(:,:,i))+1;
end

    [inMaps, inmaps2] = getInOutMaps2( flow );
    frames = length( flow );
    [rows, cols, c] = size(flow{1});
    inmaps = zeros(rows, cols, frames);
    for i=1:frames
        inmaps(:,:,i) = inmaps2{i};
    end

    inRatios = sp_inratio(sp, inMaps );

    save inprobs.mat inRatios 
    
