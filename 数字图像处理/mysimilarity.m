function score = mysimilarity(f1,f2)
%MYSIMILARITY - Compute the similarity of two figures
%
%   score = mysimilarity(f1,f2)

%% ת�Ҷ�ͼ
% ͼ��1
if size(f1,3) == 3
    f1 = rgb2gray(f1);
end
% ͼ��2
if size(f2,3) == 3
    f2 = rgb2gray(f2);
end

%% ��ֵ��
% ͼ��1
if ~islogical(f1)
    f1 = im2bw(f1,graythresh(f1));
end
% ͼ��2
if ~islogical(f2)
    f2 = im2bw(f2,graythresh(f2));
end

%% ��ͨ�����ҳ�ͼ������������
% ͼ��1
CC = regionprops(f1);
Area = zeros(1,numel(CC));
for i=1:numel(CC)
    Area(i) = CC(:).Area;
end
[area1,index] = max(Area);
rect = ceil(CC(index).BoundingBox);
xstart = rect(1);
ystart = rect(2);
xend = xstart+rect(3)-1;
yend = ystart+rect(4)-1;
obj1 = f1(ystart:yend,xstart:xend); %����1���ھ�������
% ͼ��2
CC = regionprops(f2);
Area = zeros(1,numel(CC));
for i=1:numel(CC)
    Area(i) = CC(:).Area;
end
[~,index] = max(Area);
rect = ceil(CC(index).BoundingBox);
xstart = rect(1);
ystart = rect(2);
xend = xstart+rect(3)-1;
yend = ystart+rect(4)-1;
obj2 = f2(ystart:yend,xstart:xend); %����2���ھ�������

%% ������������
[rows1,cols1] = size(obj1);
[rows2,cols2] = size(obj2);
rowscale = rows1/rows2;
colscale = cols1/cols2;

%% ���Ų��������ƶ�
%ȡrowscale��colscaleС����Ϊ��������
scale = min(rowscale,colscale);
%��������2
scaledrows = round(rows2*scale); 
scaledcols = round(cols2*scale);
scaledobj2 = imresize(obj2,[scaledrows,scaledcols]); %scaledobj2��obj2���ź�ͼ��
%��scaledobj2�����
area2 = sum(sum(scaledobj2==1));
%��obj1��scaledobj2�е�������
maxarea = max(area1,area2);
%���ź������2���������1�ɻ����ĳߴ�
striderows = rows1-scaledrows+1;
stridecols = cols1-scaledcols+1;
scores = zeros(striderows,stridecols);
%��������
for i=1:striderows
    for j=1:stridecols
        % ��obj1�пٳ���scaledobj2��С��ͬ������
        xstart = j;
        ystart = i;
        xend = xstart+scaledcols-1;
        yend = ystart+scaledrows-1;
        region = obj1(ystart:yend,xstart:xend);
        %��region��scaledobj2���ص����
        overlap = region & scaledobj2;
        areaoverlap = sum(sum(overlap==1));  
        %�����ص��ȣ����ƶȣ�
        scores(i,j) = areaoverlap/maxarea;        
    end
end
%��������ƶ�
score = max(scores(:));


