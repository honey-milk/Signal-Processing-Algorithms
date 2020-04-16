% txt2jpg.m
% �ı��ļ�תͼ��
%%
clc,clear;
close all;

%% ����
textFilename = '../src/image/cc.txt';
imageFilename = '../src/image/cc.jpg';
rows = 300;
cols = 300;
dataWidth = 24;

%% ���ļ�
fid = fopen(textFilename,'r');
data = fscanf(fid,'%u\n');
fclose(fid); % �ر��ļ�
data = reshape(data,cols,rows)';

%% ����
switch(dataWidth)
    case 1 %��ֵͼ
        image = logical(data);
    case 8 %�Ҷ�ͼ
        image = uint8(data);
    case 16 %16λ��ɫͼ
        R = fix(data/2048);
        G = fix(rem(data,2048)/32);
        B = rem(data,32);
        image = uint8(cat(3,R,G,B));
    case 24  %24λ��ɫͼ
        R = rem(fix(data/65536),256);
        G = fix(rem(data,65536)/256);
        B = rem(data,256);
        image = uint8(cat(3,R,G,B));
    otherwise
        error('dataWidth��ֵ����');
end

%% ����ͼƬ����ʾ
figure;imshow(image);
imwrite(image,imageFilename); 
