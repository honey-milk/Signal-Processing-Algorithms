%myregiongrow_demo.m
%��������ʾ��
clc,clear all;
close all;

%% ��ȡͼƬ
filename = 'F:\MATLAB\function\src\Fig1020(a).tif';
f = imread(filename);
figure;
imshow(f,[]);

%% ��������
seed = f == 255;
g = myregiongrow(f,seed,50);
figure;
imshow(g,[]);

