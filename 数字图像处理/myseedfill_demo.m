%myseedfill_demo.m
%�������ʾ��
clc,clear;
close all;

%% ��ȡͼƬ
filename = 'text.png';
f = imread(filename);
figure;
imshow(f,[]);

%% �������
seed = false(size(f));
seed(13,94) = 1;
g = myseedfill(f,seed);
figure;
imshow(g,[]);

