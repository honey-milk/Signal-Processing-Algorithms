function g = mykmeanssegment(f,k)
%MYKMEANSSEGMENT - Picture segment by K-menas
%
%   g = mykmeanssegment(f)
%   g = mykmeanssegment(f,k)

%% ��������
if nargin<2
    k = 2;
end

%% ����ͼ��f����
[rows,cols] = size(f);  %��ȡͼ��ߴ�
f = double(f); %תΪ�з�����double����
f = f(:);   %תΪ������

%% K-mean����
indf = mykmeans(f,k,struct('MaxIter',100,'Delta',1));

%% kֵ��
grayrank = round(linspace(0,255,k));  %�Ҷȵȼ�
for i=1:k
    f(indf==i) = grayrank(i);
end

%% ����תΪ����
g = reshape(f,[rows,cols]);
g = uint8(g);
