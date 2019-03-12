function h = myimhist(I)
%MYIMHIST - Display histogram of image data.
%
%   This MATLAB function calculates the histogram for the intensity image I and
%   displays a plot of the histogram
%
%   h = myimhist(I)

h = zeros(256,1);
%��ȡͼ���С
[row,col,c] = size(I);
%��ͼ���ǲ�ɫͼ����תΪ�Ҷ�ͼ
if c==3
    I = rgb2gray(I);
end

%ͳ��ÿ���Ҷȼ��ĵ���
for i=1:row
    for j=1:col
        h(I(i,j)+1) = h(I(i,j)+1)+1;
    end
end

%���û����������������ֱ��ͼ
if nargout<1
    bar((0:255),h,0.1);
    xlim([0 255]);
end


