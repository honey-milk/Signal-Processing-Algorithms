%% myhisteq_demo.m
%% ֱ��ͼ���⻯ʾ��
%%
clc,clear;
close all;

%% ��ȡͼ��
srcImage = imread('../src/image/histeq.jpg');
if size(srcImage,3) == 3
    srcImage = rgb2gray(srcImage); 
end
%��ʾͼ��
figure;imshow(srcImage);
%���㲢����ֱ��ͼ
myimhist(srcImage);

%% ֱ��ͼ���⻯
eqImage1 = myhisteq(srcImage);
figure;imshow(eqImage1);
%���㲢���ƾ��⻯���ֱ��ͼ
myimhist(eqImage1);

%% ʹ�øĽ���������ֱ��ͼ���⻯
eqImage2 = myimprovedhisteq(srcImage);
figure;imshow(eqImage2);
%���㲢���ƾ��⻯���ֱ��ͼ
myimhist(eqImage2);
