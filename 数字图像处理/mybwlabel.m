function [label,num] = mybwlabel(bw,conn)
%MYBWLABEL - Label connected components in 2-D binary image
%
%   L = mybwlabel(BW)
%   L = mybwlabel(BW,conn)
%   [L,num] = mybwlabel(...)

%% ������������㷨

%% �������
narginchk(1,2);
nargoutchk(0,2);

%% ȱʡ��������
if nargin<2
    conn = 8;   %Ĭ��8����
end

%% ����ֵ���
%����f
[rows,cols,ch] = size(bw);
if ch~=1
    error('����ͼ��BW����Ϊ��ֵͼ');
end

%% ����
if conn==4  %4����
    dxy = [-1,0;0,-1;1,0;0,1];
elseif conn==8 %8����
    dxy = [-1,-1;0,-1;1,-1;-1,0;1,0;-1,1;0,1;1,1];
else
    error('����conn����Ϊ4��8');
end

%% �㷨
label = zeros(rows,cols);
num = 0;
%����ͼƬ
for y=1:rows
    for x=1:cols
        if bw(y,x)==1 && label(y,x)==0 %Ѱ������
            num = num+1;
            stack = [];
            stack = cat(1,stack,[x,y]); %����ѹջ
            label(y,x) = num;
            while(~isempty(stack))
                seed = stack(end,:);
                stack(end,:) = [];
                %����4�����8����
                for i=1:conn
                    x2 = seed(1) + dxy(i,1);
                    y2 = seed(2) + dxy(i,2);
                    if (y2>0 && y2<=rows) && (x2>0 && x2<=cols)
                        if bw(y2,x2)==1 && label(y2,x2)==0
                            stack = cat(1,stack,[x2,y2]); %����ѹջ
                            label(y2,x2) = num;
                        end                        
                    end
                end
            end
        end
    end
end
    