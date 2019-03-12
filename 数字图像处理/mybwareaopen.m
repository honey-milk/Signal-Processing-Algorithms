function g = mybwareaopen(f,area,conn)
%MYBWAREAOPEN - Remove small objects from binary image
%
%   g = mybwareaopen(f,area)
%   g = mybwareaopen(f,area,conn)

%% �������
narginchk(2,3);
nargoutchk(0,1);

%% ȱʡ��������
if nargin<3
    conn = 8;   %Ĭ��8����
end

%% ��ͨ����
label = mybwlabel(f,conn);

%% ȥ��С����
g = f;
numcc = max(label(:)); %��ͨ������
for i=1:numcc
    index = label==i;
    Area = sum(index(:));
    if(Area < area)
        g(index) = 0; %�Ҷ���0
    end
end

