function groupList = mygroupRectangles(rectList,winSize,threshold,neighbornum)
%MYGROUPRECTANGLES - Group Rectangles
%
%   groupList = mygroupRectangles(rectList,winSize,threshold,neighbornum)

%% groupRectangles:���ο��ں�
% rectList: �����ľ��ο�
% threshold: ���ο�������ֵ
% e:�жϾ��ο����ڵ���ֵ

%% �������
narginchk(2,4);
nargoutchk(0,1);

%% ȱʡ��������
groupList = [];
if isempty(rectList)
   return; 
end
if nargin<3  
    threshold = 0.2;
elseif nargin<2
    neighbornum = 2;
    threshold = 0.2;
end
thwidth = threshold*winSize(1);  %�����ֵ
thheight = threshold*winSize(1); %�߶���ֵ

%% ���ο����
groupcell = {};
while(~isempty(rectList)) %����rectList
    rect = rectList(1,:);
    rectList(1,:) = [];
    temp = [];
    group = [];
    group = cat(1,group,rect);
    temp = cat(1,temp,rect);
    while(~isempty(temp))
        seed = temp(end,:);
        temp(end,:) = [];
        index = [];
        for i=1:size(rectList,1) %�������ڵľ��ο�
            rect = rectList(i,:);
            x1 = seed(1)+seed(3)/2;
            y1 = seed(2)+seed(4)/2;
            x2 = rect(1)+rect(3)/2;
            y2 = rect(2)+rect(4)/2;            
            dx = abs(x1-x2);
            dy = abs(y1-y2); 
            if(dx<=thwidth && dy<=thheight)
                index = cat(1,index,i);
                temp = cat(1,temp,rect);
            end
        end
        group = cat(1,group,rectList(index,:));
        rectList(index,:) = []; %ɾ��
    end
    groupcell = cat(1,groupcell,group);
end

%% ��ÿ�������
tempList = [];
for i=1:numel(groupcell)
    group = groupcell{i};
    num = size(group,1);
    if(num>=neighbornum)
        if num==1
            tempList = cat(1,tempList,group);
        else
            tempList = cat(1,tempList,mean(group));
        end
    end
end

%% ɾ��������С��
for i=1:size(tempList,1)
    issmallrect = false;
    r1 = tempList(i,:);
    for j=1:size(tempList,1)
        if(i==j)
            continue;
        end
        r2 = tempList(j,:);
        if(r1(1) > r2(1)-thwidth &&... 
           r1(2) > r2(2)-thheight &&...
           r1(1)+r1(3) < r2(1)+r2(3)+thwidth &&...
           r1(2)+r1(4) < r2(2)+r2(4)+thheight )   
           issmallrect = true;
           break;
        end
    end
    if ~issmallrect
        groupList = cat(1,groupList,r1);
    end
end