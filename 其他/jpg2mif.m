%jpg2mif.m
%ͼ��ת.mif�ļ�
%%
clc,clear;
close all;

%% ����
MifFile = 'src.mif';
ImageFile = '../src/cc.jpg';
rows = [];
cols = [];
% rows = 256;
% cols = 256;
data_width = 24;

%% ��ȡͼƬ
f = imread(ImageFile);
if isempty(rows)
    [rows,cols,~] = size(f);
else
    f = imresize(f,[rows,cols]);
end
figure;
imshow(f);
title('ԭʼͼ��');

%% ͼ����
% f = rgb2gray(f);
% figure;
% imshow(f);
% title('�Ҷ�ͼ');

%% ����
switch(data_width)
    case 1 %��ֵͼ
        data = logical(f);
    case 8 %�Ҷ�ͼ
        data = uint8(f);
    case 16 %16λ��ɫͼ
        f = double(f);
        R = fix(f(:,:,1)/8);
        G = fix(f(:,:,2)/4);
        B = fix(f(:,:,3)/8);
        data = uint16(R*32*64+G*32+B);
    case 24  %24λ��ɫͼ
        f = double(f);
        R = f(:,:,1);
        G = f(:,:,2);
        B = f(:,:,3);
        data = uint32(R*256*256+G*256+B);      
    otherwise
        error('data_width��ֵ����');
end

%% д�ļ�
data_depth = rows * cols;
fid = fopen(MifFile,'w');
fprintf(fid,'width=%d;\n',data_width);
fprintf(fid,'depth=%d;\n',data_depth);
fprintf(fid,'address_radix=uns;\n');
fprintf(fid,'data_radix=hex;\n');
fprintf(fid,'Content Begin\n');
for y=0:rows-1
    for x=0:cols-1        fprintf(fid,'%d:%x;\n',y*cols+x,data(y+1,x+1));
    end
end
fprintf(fid,'end;');
fclose(fid);%�ر��ļ�


