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
    f1 = imbinarize(f1,graythresh(f1));
end
% ͼ��2
if ~islogical(f2)
    f2 = imbinarize(f2,graythresh(f2));
end

%% ��ͨ�����ҳ�ͼ������������
% ͼ��1
CC = regionprops(f1);
Area = zeros(1,numel(CC));
for i=1:numel(CC)
    Area(i) = CC(:).Area;
end
[area,index] = max(Area);
centroid1 = CC(index).Centroid;
f1 = bwareaopen(f1,area); %�Ƴ�С����

% ͼ��2
CC = regionprops(f2);
Area = zeros(1,numel(CC));
for i=1:numel(CC)
    Area(i) = CC(:).Area;
end
[area,index] = max(Area);
centroid2 = CC(index).Centroid;
f2 = bwareaopen(f2,area); %�Ƴ�С����

%% �����Ե�����ľ���
% ͼ��1
B = bwboundaries(f1,'noholes'); %��ȡ��Ե�� 
B = B{1};
dist1 = sqrt((B(:,2)-centroid1(1)).^2+(B(:,1)-centroid1(2)).^2); %�������

% ͼ��2
B = bwboundaries(f2,'noholes'); %��ȡ��Ե�� 
B = B{1};
dist2 = sqrt((B(:,2)-centroid2(1)).^2+(B(:,1)-centroid2(2)).^2); %�������

%% ��������������ת����
% ��ֵ��ʹ�������
nd1 = length(dist1);
nd2 = length(dist2);
if nd1 > nd2
    x = linspace(0,1,nd2);
    xq = linspace(0,1,nd1);
    dist2 = interp1(x,dist2,xq,'spline')';
else
    x = linspace(0,1,nd1);
    xq = linspace(0,1,nd2);
    dist1 = interp1(x,dist1,xq,'spline')';   
end
np = length(dist1);
% ���룬�����ת����
temp1 = repmat(dist1,1,np);
temp2 = zeros(np,np);
temp2(:,1) = dist2;
for i=2:np
    temp2(:,i) = [temp2(2:end,i-1);temp2(1,i-1)]; %��λ
end
Rxyn = sum(temp1.*temp2); %ͼ��1��ͼ��2�Ļ���غ���
[~,index] = max(Rxyn);
dist2 = temp2(:,index);

%% �������ƶ�
R = corrcoef(dist1,dist2); %�������ƶȣ�ȡֵ��Χ0-1
score = R(1,2);

%% ��ͼ
% ʹ�����ֵ���
dist2 = dist2*mean(dist1)/mean(dist2);
figure;
plot(dist1,'r');
hold on;
plot(dist2,'g');
hold off;
