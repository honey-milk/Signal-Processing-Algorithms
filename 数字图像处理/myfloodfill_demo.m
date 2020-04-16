%% myfloodfill_demo.m
%% ��ˮ���ʾ��
%%
clc,clear;
close all;

%% ��ȡͼƬ
filename = '../src/image/floodfill.jpg';
f = imread(filename);
% if size(f,3)==3
%     f = rgb2gray(f);
% end
figure;imshow(f,[]);

%% ��ˮ���
seed = false(size(f,1),size(f,2));
seed(100,100) = 1;
g = myfloodfill(f,seed,[0,255,0],10);
figure;imshow(g,[]);
