function g = myseedfill(f,seed,conn)
%MYSEEDFILL - Seed filling
%
%   g = myseedfill(f,seed)
%   g = myseedfill(f,seed,conn)

%% �������
narginchk(2,3);
nargoutchk(0,1);

%% ȱʡ��������
if nargin<3
    conn = 8;   %Ĭ��8����
end

%% ����ֵ���
%����f
[rows,cols,ch] = size(f);
if ch~=1
    error('����ͼ��f����Ϊ��ֵͼ');
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
g = false(rows,cols);
[seedy,seedx] = find(seed);
nseed = length(seedy);
%��������
for n=1:nseed
    stack = [];
    stack = cat(1,stack,[seedx(n),seedy(n)]); %����ѹջ
    g(seedy(n),seedx(n)) = 1; %���Ϊ�Ѵ���
    while(~isempty(stack))
        point = stack(end,:); %����ջ��Ԫ��
        stack(end,:) = [];
        %����4�����8����
        for i=1:conn
            x2 = point(1) + dxy(i,1);
            y2 = point(2) + dxy(i,2);
            if (y2>0 && y2<=rows) && (x2>0 && x2<=cols)
                if f(y2,x2)==1>0 && g(y2,x2)==0
                    stack = cat(1,stack,[x2,y2]); %����ѹջ
                    g(y2,x2) = 1; %���Ϊ�Ѵ���
                end                        
            end
        end
    end    
end
