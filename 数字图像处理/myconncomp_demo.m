%% myconncomp_demo.m
%% ��ͨ����ʾ��
%%
clc,clear;
close all;

%% ��ȡͼƬ
filename = '../src/image/cc.jpg';
f = imread(filename);
figure;
imshow(f,[]);

%% ��ֵ��
f2 = rgb2gray(f);
thresh = graythresh(f2);
f2 = imbinarize(f2,thresh);
figure;
imshow(f2,[]);

%% ��ͨ����
label = mybwlabel(f2,4);
f3 = label * 255 / max(label(:));
figure;
imshow(f3,[]);

%% ��������ɫ
f4 = f;
[rows,cols,c] = size(f);
for i=0:max(label(:))
    rgb = round(rand(1,3)*255);
    index = label==i;
    indexr = false(rows,cols,c);
    indexg = false(rows,cols,c);
    indexb = false(rows,cols,c);
    indexr(:,:,1) = index;
    indexg(:,:,2) = index;
    indexb(:,:,3) = index;
    f4(indexr) = rgb(1);
    f4(indexg) = rgb(2);
    f4(indexb) = rgb(3);
end
figure;
imshow(f4,[]);

%% ��ͨ����
cc = myconncomp(f2,4);
f5 = f;
for i=1:numel(cc)
    f5 = insertShape(f5, 'rectangle', cc(i).BoundingBox, 'Color', [255,0,0],'LineWidth', 4); 
    f5 = insertMarker(f5,cc(i).Centroid,'color','g','size',10);
end
figure;
imshow(f5,[]);

%% �Ƴ�С����
f6 = mybwareaopen(f2,500);
figure;
imshow(f6,[]);
