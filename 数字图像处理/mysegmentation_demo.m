%% mysegmentation_demo.m
%% ����ͼ��ͼ��ָ�
%%
clc,clear;
close all;

%% ��ȡͼƬ
tic;
filename = '../src/image/segment.jpg';
f = imread(filename);
figure;imshow(f,[]);

%% ת�Ҷ�ͼ
gray = rgb2gray(f);
gray = repmat(gray,1,1,3);

%% ͼ��ָ�
sigma = 0.5;
k = 500;
min_size = 100;
[g,num_ccs] = mysegmentation(f,sigma,k,min_size);
toc

%% ��ʾ���
figure;imshow(g,[]);
figure;imshow(gray*0.25+g*0.75,[]);
