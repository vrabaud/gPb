function compile(gccVer)
% Compiles all the private routines
%
% This function can recompile all that you need for the toolbox to run.
% That includes:
%  - the SBA static library that is needed to build the ...
%  - ... SBA mex file
%  - 3 normal mex files
% By default, the toolbox comes with precompiled binaries for windows 32
% It is easy to compile on Linux and I am developing on Linux so it works
% for sure there.
%
% USAGE
%  toolboxCompile()
%
% INPUTS
%  doSba      - [false] if true, it will compile the dll projection files
%               for SBA and that requires matlab to be launched from the
%               visual c++ terminal on Windows
%  gccVer     - on Linux, specify your gcc version here (e.g. 4.2 if
%               gcc-4.2 is a supported compiler on Matlab), which can be
%               useful if matlab thinks your gcc is too old/new
%
% OUTPUTS
%
% EXAMPLE
%
% See also
%
% Vincent's Structure From Motion Toolbox      Version 3.1
% Copyright (C) 2008-2011 Vincent Rabaud.  [vrabaud-at-cs.ucsd.edu]
% Please email me if you find bugs, or have suggestions or questions!
% Licensed under the GPL [see external/gpl.txt]

if nargin==0; gccVer=num2str(4.7); else gccVer=num2str(gccVer); end

if strcmp(computer,'GLNX86') || strcmp(computer,'GLNXA64')
  gcc_extra = [ 'CXX=g++-' gccVer ' CC=g++-' gccVer ' LD=g++-' gccVer ];
end

disp('Compiling.......................................');
savepwd=pwd;

% build the SBA mex file
cd('source/gpb_src/');

isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

if isOctave
    % Octave on Linux
    mkoctfile --mex ./matlab/segmentation/mex_contour_sides.cc  -I./include
    mkoctfile --mex ./matlab/segmentation/mex_nonmax_oriented.cc  -I./include -L./ -L../../lib -lopencv_gpb
    mkoctfile --mex ./matlab/segmentation/mex_pb_parts_final_selected.cc -I./include -L../../lib -lopencv_gpb
else
    mex ./matlab/segmentation/mex_contour_sides.cc  -I./include
    mex ./matlab/segmentation/mex_nonmax_oriented.cc  -I./include -L./ -L../../lib -lopencv_gpb
    mex ./matlab/segmentation/mex_pb_parts_final_selected.cc -I./include -L../../lib -lopencv_gpb
end

system('mv *mex ../../lib')
cd(savepwd);

% process savgol
cd('source/savgol/');

if isOctave
    mkoctfile --mex ./savgol_border.cpp
else
    mex ./savgol_border.cpp
end

system('mv *mex ../../lib')
cd(savepwd);

% process buildW
cd('source/buildW/');

if isOctave
    mkoctfile --mex ./buildW.cpp -I./util -L../../lib -lopencv_gpb
else
    mex ./buildW.cpp -I./util -L../../lib -lopencv_gpb
end

system('mv *mex ../../lib')
cd(savepwd);


% process custom mex files
cd('source/opencv_gpb/');

if isOctave
    mkoctfile --mex ./mex/watershed.cpp ./mex/MxArray.cpp -I./src -I../buildW/util/ -I../buildW -I/opt/ros/fuerte/include -L../../lib -L/opt/ros/fuerte/lib -lopencv_gpb -lopencv_core -lopencv_imgproc -L../build -lbuildW -lopencv_gpb
else
    mex ./mex/watershed.cpp ./mex/MxArray.cpp -I./src -I../buildW/util/ -I../buildW -I/opt/ros/fuerte/include -L../../lib -L/opt/ros/fuerte/lib -lopencv_gpb -lopencv_core -lopencv_imgproc -L../build -lbuildW -lopencv_gpb
end

system('mv *.mex ../../lib')
cd(savepwd);


% process ucm
cd('source/ucm/');

if isOctave
    mkoctfile --mex ./ucm_mean_pb.cpp -L../../lib -lopencv_gpb
else
    mex ./ucm_mean_pb.cpp -L../../lib -lopencv_gpb
end

system('mv *mex ../../lib')
cd(savepwd);





disp('..................................Done Compiling');

end
