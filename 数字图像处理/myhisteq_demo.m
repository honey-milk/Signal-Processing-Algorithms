%myhisteq_demo.m
%ֱ��ͼ���⻯ʾ��
%%
clc,clear;
close all;

%% ��ȡͼ��
img_src = imread('F:\MATLAB\function\src\histeq.jpg');
%��ȡͼƬ��Ⱥ͸߶�
[H,W,C] = size(img_src);
%��ԭͼΪ��ɫͼ����ת�Ҷ�ͼ
if C==3
    img_src = rgb2gray(img_src); 
end
%��ʾͼ��
figure;
imshow(img_src);
%���㲢����ֱ��ͼ
figure;
myimhist(img_src);

%% ֱ��ͼ���⻯
img_eq1 = myhisteq(img_src);
figure;
imshow(img_eq1);
%���㲢���ƾ��⻯���ֱ��ͼ
figure;
myimhist(img_eq1);

%% ʹ�øĽ���������ֱ��ͼ���⻯
img_eq2 = myimprovedhisteq(img_src);
figure;
imshow(img_eq2);
%���㲢���ƾ��⻯���ֱ��ͼ
figure;
myimhist(img_eq2);
