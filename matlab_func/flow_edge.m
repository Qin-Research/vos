load name.mat
addpath(genpath('external/FastVideoSegment/'))
%addpath('modified')
addpath('matlab_func/modified')
flow_file = ['flow_' name '.mat'];
load(flow_file)

% copy pasted from getInOutMaps.m

    frames = length( flow );

    [ height, width, ~ ] = size( flow{ 1 } );
    
    boundaryMaps = zeros(height,width,frames);

    for( frame = 1: frames )
        boundaryMap = prob_edge( flow{ frame }, 3 );
	boundaryMaps(:,:,frame) = boundaryMap;
    end

save 'flow_edges.mat' boundaryMaps;    
