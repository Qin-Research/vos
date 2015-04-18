load name.mat
addpath(genpath('external/FastVideoSegment/'))
flow_file = ['flow_' name '.mat'];
load(flow_file)
load(['sp_' name '.mat'])
load(['diffused_' name '.mat'])


addpath('matlab_func/modified')
frames = size(superpixels,3);
sp = cell(frames, 1);

nodes = 0;
for i =1:frames
    sp{i} = uint16(superpixels(:,:,i))+1;
    nodes = nodes + length(unique(sp{i}));
end

nodes

diffused = cell(frames,1);
for i =1:frames
    diffused{i} = diffused_inprobs{i}';
end    

% copy pasted from videoRapideSegment.m
   [ ~, accumulatedInRatios] = accum_inprob(flow, sp, diffused );
    frames = length( flow );
   
    locationMasks = cell2mat( accumulatedInRatios );
    
    locationUnaries = 0.5 * ones( nodes, 2, 'single' );

    locationNorm = 0.75; % from videoRapidSegment.m, line 67
    locationUnaries( 1: length( locationMasks ), 1 ) = ...
        locationMasks / ( locationNorm * max( locationMasks ) );
    locationUnaries( locationUnaries > 0.95 ) = 0.999;
    
%    for( frame = 1: frames )
%        start = bounds( frame );
%        stop = bounds( frame + 1 ) - 1;
%        
%        frameMasks = locationUnaries( start: stop, 1 );
%        overThres = sum( frameMasks > 0.6 ) / single( ( stop - start ) );
%
%        if( overThres < 0.05 )
%            E = 0.005;
%        else
%            E = 0.000;
%        end
%        locationUnaries( start: stop, 1 ) = ...
%            max( locationUnaries( start: stop, 1 ), E );
%        
%    end
%    locationUnaries( :, 2 ) = 1 - locationUnaries( :, 1 );
   
    save(['locprior_' name '.mat'], 'locationUnaries');
