
addpath(fullfile(pwd,'lib'));

imgFile = 'data/101087_small.jpg';
outFile = 'data/101087_small_gPb.mat';

gPb_orient = globalPb(imgFile, outFile);

ucm = contours2ucm(gPb_orient, 'imageSize');
figure;imshow(ucm);
