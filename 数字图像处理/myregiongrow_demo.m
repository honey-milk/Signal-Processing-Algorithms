%% myregiongrow_demo.m
%% ��������ʾ��
%%
clc,clear;
close all;

%% ��ȡͼƬ
filename = '../src/image/Fig1020(a).tif';
f = imread(filename);
figure;imshow(f,[]);

%% ��������
seed = f == 255;
g = myregiongrow(f,seed,50);
figure;imshow(g,[]);