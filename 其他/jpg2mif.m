% jpg2mif.m
% ͼ��ת.mif�ļ�
%%
clc,clear;
close all;

%% ����
mifFilename = '../src/image/cc.mif';
imageFilename = '../src/image/cc.jpg';
rows = [];
cols = [];
dataWidth = 24;

%% ��ȡͼƬ
image = imread(imageFilename);
if isempty(rows)
    [rows,cols,~] = size(image);
else
    image = imresize(image,[rows,cols]);
end
figure;imshow(image);
title('ԭʼͼ��');

%% ͼ����
% image = rgb2gray(image);
% figure;imshow(image);
% title('�Ҷ�ͼ');

%% ����
switch(dataWidth)
    case 1 % ��ֵͼ
        data = logical(image);
    case 8 % �Ҷ�ͼ
        data = uint8(image);
    case 16 % 16λ��ɫͼ
        image = double(image);
        R = fix(image(:,:,1) / 8);
        G = fix(image(:,:,2) / 4);
        B = fix(image(:,:,3) / 8);
        data = uint16(R * 32 * 64 + G * 32 + B);
    case 24  % 24λ��ɫͼ
        image = double(image);
        R = image(:,:,1);
        G = image(:,:,2);
        B = image(:,:,3);
        data = uint32(R * 256 * 256 + G * 256 + B);      
    otherwise
        error('dataWidth��ֵ����');
end

%% д�ļ�
dataDepth = rows * cols;
fid = fopen(mifFilename,'w');
fprintf(fid,'width=%d;\n',dataWidth);
fprintf(fid,'depth=%d;\n',dataDepth);
fprintf(fid,'address_radix=uns;\n');
fprintf(fid,'data_radix=hex;\n');
fprintf(fid,'Content Begin\n');
indexes = 0:dataDepth-1;
data = data';    
fprintf(fid,'%d:%x;\n',[indexes;data(:)']);
fprintf(fid,'end;');
fclose(fid); % �ر��ļ�


