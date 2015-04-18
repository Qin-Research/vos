load name.mat
addpath(genpath('external/FastVideoSegment/'))
%addpath('modified')
addpath('matlab_func/modified')
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

 movefile(flow_file, 'external/FastVideoSegment');
end    

% copy pasted from getInOutMaps.m

    frames = length( flow );

    [ height, width, ~ ] = size( flow{ 1 } );
    
    boundaryMaps = zeros(height,width,frames);

    for( frame = 1: frames )
        boundaryMap = prob_edge( flow{ frame }, 3 );
	boundaryMaps(:,:,frame) = boundaryMap;
    end

save(['flow_edges_' name '.mat'], 'boundaryMaps');    
