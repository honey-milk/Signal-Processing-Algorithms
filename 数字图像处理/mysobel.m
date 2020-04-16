function [g,gx,gy] = mysobel(f,T)
%MYSOBEL - Edge detectation by Sobel operater
%
%   g = mysobel(f)
%   g = mysobel(f,T)
%   [g,gx,gy] = mysobel(...)


%% ��������Ŀ
narginchk(1,2);
nargoutchk(0,3);

%% ȱʡ��������
if nargin<2
    T = [];
end

%% ת�Ҷ�ͼ
if size(f,3)==3
    f = rgb2gray(f);
end
f = double(f); %תdouble����

%% ��Ե���
Hx = [-1,-2,-1;0,0,0;1,2,1]; %ˮƽ����sobel����
Hy = Hx'; %��ֱ����sobel����
gx = imfilter(f,Hx,'replicate');    %ˮƽ��Ե
gy = imfilter(f,Hy,'replicate');    %��ֱ��Ե
gx = abs(gx); %ȡ����ֵ
gy = abs(gy); %ȡ����ֵ
g = sqrt(gx.^2+gy.^2);              %��Ե

%% ��һ��
g = g./max(g(:));
gx = gx./max(gx(:));
gy = gy./max(gy(:));

%% ��ֵ��
if ~isempty(T)
    g = g>=T;
end
