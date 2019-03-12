function g = myfloodfill(f,point,newval,range,neighbor,fixedrange)
%MYFLOODFILL - Flood filling
%
%   g = myfloodfill(f)
%   g = myfloodfill(f,neighbor)
%   g = myfloodfill(f,neighbor,fixedrange)

%% myfloodfill:��ˮ����㷨
% f���Ҷ�ͼ���ɫͼ
% point�����ӵ�
% newval��������ɫ
% neighbor��4�����8����
% fixedrange���̶���Χ�򸡶���Χ

%% �������
narginchk(3,6);
nargoutchk(1,1);

%% ȱʡ��������
if nargin<4
    range = [0,0];
    neighbor = 4;   %Ĭ��4����
    fixedrange = true;
elseif nargin<5
    neighbor = 4;   %Ĭ��4����
    fixedrange = true;
elseif nargin<6
    fixedrange = true;
end

%% ����ֵ���
%����f
[rows,cols,ch] = size(f);
f = double(f);
%����point
x = point(1);
y = point(2);
if (x<1 || x>cols) || (y<1 || y>rows)
    error('point���곬����ͼ��f�ķ�Χ');
end
%����range
if ch==3 && size(range,2)==1
    range = repmat(range,1,3); %���Ƴ�3ά��RGB��
end

%% ��ɫ��ֵ
val = reshape(f(point(2),point(1),:),1,ch);
lowthresh = val + range(1,:);
highthresh = val + range(2,:);

%% ����
if neighbor==4  %4����
    dxy = [-1,0;0,-1;1,0;0,1];
elseif neighbor==8 %8����
    dxy = [-1,-1;0,-1;1,-1;-1,0;1,0;-1,1;0,1;1,1];
else
    error('����neighbor����Ϊ4��8');
end

%% �㷨
g = f;
label = zeros(rows,cols);
stack = []; %�����ӵ�ջ
stack = cat(1,stack,point);
%������������
while(~isempty(stack))
    point = stack(end,:);
    stack(end,:) = [];
    x = point(1);
    y = point(2);
    label(y,x) = 1;
    g(y,x,:) = newval;
    if fixedrange==false
        val = reshape(f(y,x,:),1,ch);
        lowthresh = val + range(1,:);
        highthresh = val + range(2,:);
    end
    %����4�����8����
    for i=1:neighbor
        x2 = x + dxy(i,1);
        y2 = y + dxy(i,2);
        if (y2>0 && y2<=rows) && (x2>0 && x2<=cols)
            val = reshape(f(y2,x2,:),1,ch);
            if( all(val >= lowthresh)...
                && all(val <= highthresh)...
                && label(y2,x2)==0 )
                stack = cat(1,stack,[x2,y2]);
                label(y2,x2) = 1;
            end            
        end
    end
end

%% ���ͼ��gתΪuint8����
g = uint8(g);
    