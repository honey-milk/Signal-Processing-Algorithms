function g = myregiongrow(f,seed,thresh,conn)
%MYREGIONGROW - Perform segmeation by region graowing
%
%   g = myregiongrow(f,seed,thresh)
%   g = myregiongrow(f,seed,thresh,conn)

%% �������
narginchk(3,4);
nargoutchk(0,1);

%% ȱʡ��������
if nargin<4
    conn = 8;   %Ĭ��8����
end

%% ����ֵ���
%����f
f = double(f);
%����seed
if numel(seed)==1 %seedΪ����
    seedvalue = seed;
    seed = f==seed;
else
    seed = bwmorph(seed,'shrink',inf);
    seedvalue = f(seed);
end

%% ��ֵ����
bw = false(size(f));
nseed = length(seedvalue);
for n=1:nseed
    temp = abs(f-seedvalue(n)) <= thresh;
    bw = bw | temp;
end

%% �������
g = myseedfill(bw,seed,conn);
% g = imreconstruct(seed,bw);

