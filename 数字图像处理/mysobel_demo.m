%% mysobel_demo.m
%% Sobel���ӱ�Ե���
%%
clc,clear;
close all;

%% ��ȡͼƬ
f = imread('../src/image/lena.jpg');
figure;imshow(f);
title('ԭʼͼ��');

%% ��Ե���
[g,gx,gy] = mysobel(f);
bw = mysobel(f,0.2); %ʹ����ֵ��ֵ��
figure;imshow(gx,[]);
title('x�����Ե');
figure;imshow(gy,[]);
title('y�����Ե');
figure;imshow(g);
title('�ܱ�Ե');
