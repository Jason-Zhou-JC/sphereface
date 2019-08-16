% --------------------------------------------------------
% Copyright (c) Weiyang Liu, Yandong Wen
% Licensed under The MIT License [see LICENSE for details]
%
% Intro:
% Get a image list `$SPHEREFACE_ROOT/train/data/CASIA-WebFace-112X96.txt`,
% which is needed by caffe-sphereface
%
% Usage:
% cd $SPHEREFACE_ROOT/train
% run code/get_list.m
% --------------------------------------------------------

clear;clc;close all;
cd('../');

root      = '/home/jason/Datasets/InsightFace';
% folder    = fullfile(root, 'lfw_112x112');
% folder    = fullfile(root, 'calfw_112x112');
% folder    = fullfile(root, 'cplfw_112x112');
folder    = fullfile(root, 'faces_webface_112x112');
% folder    = fullfile(root, 'faces_emore_112x112');
% folder    = fullfile(root, 'faces_glintasia_112x112');
subFolder = struct2cell(dir(folder))';
subFolder = subFolder(3:end, 1);

% % exclude the identities appearing in LFW dataset
% indx = ismember(subFolder, [{'0166921'}, {'1056413'}, {'1193098'}]);
% subFolder(indx) = [];
% 
% % exclude the identities appearing in LFW dataset by AM-Softmax
% indx = ismember(subFolder, [{'0000513'}, {'0004763'}, {'0005009'}, {'0005082'}, {'0005172'}, {'0166921'},...
%                             {'0208962'}, {'0454042'}, {'0662519'}, {'0706787'}, {'0955471'}, {'1056413'},...
%                             {'1091782'}, {'1193098'}, {'1303492'}, {'3478560'}]);
% subFolder(indx) = [];
% 
% % exclude the identities appearing in FaceScrub(MegaFace) dataset by AM-Softmax
% indx = ismember(subFolder, [{'0111013'}, {'0000105'}, {'0100792'}, {'0121605'}, {'0247143'}, {'0113158'},...
%                             {'0262635'}, {'0003115'}, {'0001002'}, {'0000529'}, {'0386472'}, {'0221043'},...
%                             {'1423270'}, {'0000396'}, {'0413168'}, {'0799777'}, {'0005188'}, {'0001664'},...
%                             {'0000460'}, {'0100866'}, {'0881631'}, {'0000662'}, {'3229685'}, {'0005102'},...
%                             {'0000447'}, {'0376540'}, {'0484678'}, {'1072555'}, {'0004294'}, {'0005342'},...
%                             {'0001624'}, {'0000664'}, {'0911320'}, {'0119003'}, {'0534635'}, {'0460694'},...
%                             {'0000202'}, {'0405103'}, {'1065664'}, {'1132359'}, {'0088144'}, {'1842439'}]);
% subFolder(indx) = [];

% create the list for trianing
% fid = fopen(fullfile(root, 'lfw_112x112.txt'), 'w');
% fid = fopen(fullfile(root, 'calfw_112x112.txt'), 'w');
% fid = fopen(fullfile(root, 'cplfw_112x112.txt'), 'w');
fid = fopen(fullfile(root, 'faces_webface_112x112.txt'), 'w');
% fid = fopen(fullfile(root, 'faces_emore_112x112.txt'), 'w');
% fid = fopen(fullfile(root, 'faces_glintasia_112x112.txt'), 'w');
for i = 1:length(subFolder)
    fprintf('Collecting the %dth folder (total %d) ...\n', i, length(subFolder));
    subList   = struct2cell(dir(fullfile(folder, subFolder{i}, '*.jpg')))';
    % fileNames = fullfile(folder, subFolder{i}, subList(:, 1));
    fileNames = fullfile(subFolder{i}, subList(:, 1));
    for j = 1:length(fileNames)
        % fprintf(fid, '%s %d\n', fileNames{j}, i-1);
        fprintf(fid, '%s %d\n', fileNames{j}, str2num(subFolder{i}));
    end
end
fclose(fid);