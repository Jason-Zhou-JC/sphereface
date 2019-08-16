% --------------------------------------------------------
% Copyright (c) Weiyang Liu, Yandong Wen
% Licensed under The MIT License [see LICENSE for details]
%
% Intro:
% This script is used to detect the faces in training & testing datasets (CASIA & LFW).
% Face and facial landmark detection are performed by MTCNN 
% (paper: http://kpzhang93.github.io/papers/spl.pdf, 
%  code: https://github.com/kpzhang93/MTCNN_face_detection_alignment).
%
% Note:
% If you want to use this script for other dataset, please make sure
% (a) the dataset is structured as `dataset/idnetity/image`, e.g. `casia/id/001.jpg`
% (b) the folder name and image format (bmp, png, etc.) are correctly specified.
% 
% Usage:
% cd $SPHEREFACE_ROOT/preprocess
% run code/face_detect_demo.m
% --------------------------------------------------------

function face_detect()

clear;clc;close all;
cd('../');

% collect a image list of dataset
try
    % load('result/dataList_lumi_gallery.mat');
    % resume = load('result/i_lumi_gallery.mat');
    load('result/dataList_lumi_probe.mat');
    resume = load('result/i_lumi_probe.mat');
    start = resume.i;
    fprintf('--- Resume from %dth image (total %d) ...\n', start, length(dataList));
catch
    % dataList = collectData('/home/jason/Datasets/LUMI/face_recognition/gallery_images', 'gallery_images');
    dataList = collectData('/home/jason/Datasets/LUMI/face_recognition/probe_images', 'probe_images');
    start = 1;
end

%% mtcnn settings
minSize   = 20;
factor    = 0.85;
threshold = [0.6 0.7 0.9];

%% add toolbox paths
matCaffe       = fullfile(pwd, '../tools/caffe-sphereface/matlab');
pdollarToolbox = fullfile(pwd, '../tools/toolbox');
MTCNN          = fullfile(pwd, '../tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1');
addpath(genpath(matCaffe));
addpath(genpath(pdollarToolbox));
addpath(genpath(MTCNN));

%% caffe settings
gpu = 1;
if gpu
   gpu_id = 0;
   caffe.set_mode_gpu();
   caffe.set_device(gpu_id);
else
   caffe.set_mode_cpu();
end
caffe.reset_all();
modelPath = fullfile(pwd, '../tools/MTCNN_face_detection_alignment/code/codes/MTCNNv1/model');
PNet = caffe.Net(fullfile(modelPath, 'det1.prototxt'), ...
                 fullfile(modelPath, 'det1.caffemodel'), 'test');
RNet = caffe.Net(fullfile(modelPath, 'det2.prototxt'), ...
                 fullfile(modelPath, 'det2.caffemodel'), 'test');
ONet = caffe.Net(fullfile(modelPath, 'det3.prototxt'), ...
                 fullfile(modelPath, 'det3.caffemodel'), 'test');

%% face and facial landmark detection
fprintf('There are %d images:\n', length(dataList));
for i = start:length(dataList)
    % fprintf('detecting the %dth image %s...\n', i, dataList(i).file);
    if i == start || mod(i, 1000) == 0
        fprintf('detecting the %dth image...\n', i);
    end
    if i == 1 || mod(i, 50000) == 0
        % save result/i_lumi_gallery.mat i
        % save('result/dataList_lumi_gallery.mat', 'dataList', '-v7.3');
        save result/i_lumi_probe.mat i
        save('result/dataList_lumi_probe.mat', 'dataList', '-v7.3');
    end
    % load image
    try
        img = imread(dataList(i).file);
    catch
        continue;
    end
    if size(img, 3)==1
       img = repmat(img, [1,1,3]);
    end
    % detection, we recommend you to set minsize as x * short side
    minl=min([size(img,1) size(img,2)]);
	minSize=fix(minl*0.1);
    [bboxes, landmarks] = detect_face(img, minSize, PNet, RNet, ONet, threshold, false, factor);

    if size(bboxes, 1)>1
       % pick the face closed to the center
       center   = size(img) / 2;
       if strcmp(dataList(i).dataset, 'gallery_images')
           center(1) = center(1) / 2;
       end
       distance = sum(bsxfun(@minus, [mean(bboxes(:, [2, 4]), 2), ...
                                      mean(bboxes(:, [1, 3]), 2)], center(1:2)).^2, 2);
       [~, Ix]  = min(distance);
       dataList(i).facial5point = reshape(landmarks(:, Ix), [5, 2]);
       dataList(i).bboxes = reshape(bboxes(Ix, [1:4]), [2, 2]);
    elseif size(bboxes, 1)==1
       dataList(i).facial5point = reshape(landmarks, [5, 2]);
       dataList(i).bboxes = reshape(bboxes([1:4]), [2, 2]);
    else
       dataList(i).facial5point = [];
       dataList(i).bboxes = [];
    end
end

if start <= length(dataList)
    i = length(dataList) + 1;
    % save result/i_lumi_gallery.mat i
    % save('result/dataList_lumi_gallery.mat', 'dataList', '-v7.3');
    save result/i_lumi_probe.mat i
    save('result/dataList_lumi_probe.mat', 'dataList', '-v7.3');
end

end


function list = collectData(folder, name)
    % subFolders = struct2cell(dir(folder))';
    % subFolders = subFolders(3:end, 1);
    % files      = cell(size(subFolders));
    % for i = 1:length(subFolders)
        % fprintf('%s --- Collecting the %dth folder (total %d) ...\n', name, i, length(subFolders));
        % subList  = struct2cell(dir(fullfile(folder, subFolders{i}, '*.jpg')))';
        % files{i} = fullfile(folder, subFolders{i}, subList(:, 1));
    % end
    fprintf('%s --- Collecting images ...\n', name);
    subList  = struct2cell(dir(fullfile(folder, '*.jpg')))';
    files = fullfile(folder, subList(:, 1));
    files      = vertcat(files);
    dataset    = cell(size(files));
    dataset(:) = {name};
    list       = cell2struct([files dataset], {'file', 'dataset'}, 2);
end