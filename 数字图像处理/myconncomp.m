function cc = myconncomp(f,conn)
%MYCONNCOMP - Find connected components in binary image
%
%   cc = myconncomp(f)
%   cc = myconncomp(f,conn)

%% �������
narginchk(1,2);
nargoutchk(0,1);

%% ȱʡ��������
if nargin<2
    conn = 8;   %Ĭ��8����
end

%% ��ͨ����
label = mybwlabel(f,conn);

%% ������ͨ�����
numcc = max(label(:)); %��ͨ������
cc(numcc) = struct('Area',[],'Centroid',[],'BoundingBox',[]);
for i=1:numcc
    [y,x] = find(label==i);
    cc(i).Area = length(x);
    mx = mean(x);
    my = mean(y);       
    cc(i).Centroid = [mx,my];
    x1 = min(x);
    x2 = max(x);
    y1 = min(y);
    y2 = max(y);
    w = x2-x1+1;
    h = y2-y1+1;
    cc(i).BoundingBox = [x1,y1,w,h];
end
