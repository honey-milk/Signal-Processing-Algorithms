function resbboxes = mynms(bboxes,threshold,neighbornum,type)
%MYNMS - Non maximum suppression
%
%   resbboxes = mynms(bboxes,threshold,neighbornum,type)

% bboxes��m x 5����,��ʾ��m����5�зֱ���[x y w h score]
% threshold���ص�����ֵ
% type���ص�����ֵ����

%% �������
narginchk(3,4);
nargoutchk(0,1);

%% �������ֵ���
% ����bboxes
if isempty(bboxes)
  resbboxes = [];
  return;
end
% ����type
if nargin < 4
    type = 1;
end

%% ��ȡbboxes���е�ֵ
x1 = bboxes(:,1);
y1 = bboxes(:,2);
w = bboxes(:,3);
h = bboxes(:,4);    
score = bboxes(:,5);
x2 = x1+w-1;
y2 = y1+h-1;
% ����ÿһ��������
area = w.*h;
% ���÷ְ���������
[~, I] = sort(score);

%% �㷨
% ������ʼ��
pick = score*0;
counter = 1;
% ѭ��ֱ�����п������
while ~isempty(I)
    last = length(I); %��ǰʣ��������
    i = I(last); %ѡ�����һ�������÷���ߵĿ�
    %�����ཻ���
    xx1 = max(x1(i),x1(I(1:last-1)));
    yy1 = max(y1(i),y1(I(1:last-1)));
    xx2 = min(x2(i),x2(I(1:last-1)));
    yy2 = min(y2(i),y2(I(1:last-1)));  
    w = max(0,xx2-xx1+1);
    h = max(0,yy2-yy1+1); 
    inter = w.*h;
    %��ͬ�����µ��ص���
    if type==1
        %�ص��������С������ı�ֵ
        o = inter./min(area(i),area(I(1:last-1)));
    else
        %����/����
        o = inter./(area(i)+area(I(1:last-1))-inter);
    end
    %�����ص�������ڵ�����ֵ�Ŀ�
    index = find(o>=threshold);
    if length(index)>=neighbornum
        pick(counter) = i;
        counter = counter+1;            
    end
    %���������ص����С����ֵ�Ŀ������´δ���
    I = I(o<threshold);
end
pick = pick(1:(counter-1));
resbboxes = bboxes(pick,:);

