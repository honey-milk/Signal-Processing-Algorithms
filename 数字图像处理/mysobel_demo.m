%mysobel_demo.m
%%
clc,clear all;
close all;

%% ��ȡͼƬ
f = imread('test3.bmp');
figure;
imshow(f);
title('ԭʼͼ��');

%% ��Ե���
[g,gx,gy] = mysobel(f);
bw = mysobel(f,0.2); %ʹ����ֵ��ֵ��
figure;
imshow(gx,[]);
title('x�����Ե');
figure;
imshow(gy,[]);
title('y�����Ե');
figure;
imshow(g,[]);
title('�ܱ�Ե');
figure;
imshow(bw,[]);
title('��ֵ����Ե');


