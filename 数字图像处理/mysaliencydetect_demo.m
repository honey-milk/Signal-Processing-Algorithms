%% mysaliencydetect_demo.m
%% �����Լ��ʾ��
%% 
clc,clear;
close all;

%% ����ͼƬ
img = imread('../src/image/cc.jpg');
figure;
imshow(img);

%% �����Լ��
[rects,objectMap] = mysaliencydetect(img);
for i=1:size(rects,1)
    img = insertShape(img,'rectangle',rects,'Color',[255,0,0],'LineWidth',4); 
end
figure;imshow(objectMap);
figure;imshow(img);
