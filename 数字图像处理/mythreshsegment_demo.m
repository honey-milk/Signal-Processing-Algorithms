%% mythreshsegment_demo.m
%% ͼ����ֵ�ָ�ʾ��
%% 
clc,clear;
close all;

%%
addpath('../����ѧϰ');

%% ��ȡͼƬ
filename = '../src/image/Fig1019(a).tif';
f = imread(filename);
if size(f,3)==3
    f = rgb2gray(f);
end
figure;
imshow(f,[]);
title('ԭʼͼ��');

%% ȫ����ֵ�����
f1 = myglobalthresh(f);
figure;imshow(f1,[]);
title('ȫ����ֵ��');

%% K-means2������
f2 = mykmeanssegment(f,3);
figure;imshow(f2,[]);
title('K-means');

%% ���
T = myostuthresh(f);
f3 = imbinarize(f,T);
figure;imshow(f3,[]);
title('���');

%% �ֲ���ֵ��
f4 = mylocalthresh(f,ones(3),30,1);
figure;imshow(f4,[]);
title('�ֲ���ֵ��');

%% �ƶ�ƽ����
f5 = mymovingthresh(f,20,0.5);
figure;imshow(f5,[]);
title('�ƶ�ƽ����');
