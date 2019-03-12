function g = myfloodfill(f,seed,newval,thresh,conn)
%MYFLOODFILL - Flood filling
%
%   g = myfloodfill(f,seed,newval)
%   g = myfloodfill(f,seed,newval,thresh)
%   g = myfloodfill(f,seed,newval,thresh,conn)

%% myfloodfill:��ˮ����㷨
% f���Ҷ�ͼ���ɫͼ
% seed�����ӵ�
% newval��������ɫ
% thresh���ж����Ƶ����ֵ
% conn��4�����8����

%% �������
narginchk(3,5);
nargoutchk(0,1);

%% ȱʡ��������
if nargin<4
    thresh = 0;     %Ĭ����ֵΪ0
    conn = 8;   %Ĭ��8����
elseif nargin<5
    conn = 8;   %Ĭ��8����
end

%% ����ֵ���
ch = size(f,3);
if ch == 3 %��ɫͼ
    if length(thresh) == 1
        thresh = repmat(thresh,1,3);
    end 
end

%% ��ˮ���
g = f;

if ch == 3  
    for i=1:3 %3��ͨ���ֱ���
        index = false(size(f));
        index(:,:,i) = myregiongrow(f(:,:,i),seed,thresh(i),conn);
        g(index) = newval(i);       
    end
else 
    index = myregiongrow(f,seed,thresh,conn);
    g(index) = newval;
end


    