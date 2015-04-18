load name.mat
addpath(genpath('external/FastVideoSegment/'))
flow_file = ['flow_' name '.mat'];

if exist(flow_file,'file') == 2
 load(flow_file)
else
 options.infolder = fullfile('data','rgb', name);
 options.outfolder = fullfile('data','flow');
 options.flowmethod = 'broxPAMI2011';
options.vocal = false; 
 n_frames = numel(dir([options.infolder '/*.png']));
 options.ranges = [1, n_frames+1];
 flow = computeOpticalFlow( options,1);
 save(flow_file, 'flow');
end    

load(['sp_' name '.mat'])
%addpath('modified')
addpath('matlab_func/modified')
frames = size(superpixels,3);
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

    save(['inprobs_' name '.mat'], 'inRatios') 
    
