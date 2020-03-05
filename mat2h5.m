% Load the NYUv2 dataset variables in matlab format
load('data/nyu_depth_v2_labeled.mat')

% Save images
h5create('data/nyu_depth_v2.h5', '/images', size(images))
h5write('data/nyu_depth_v2.h5', '/images', images)

% Save depth maps
h5create('data/nyu_depth_v2.h5', '/depths', size(depths))
h5write('data/nyu_depth_v2.h5', '/depths', depths)
