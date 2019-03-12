function m = mylocalmean(f,nhood)
%MYLOCALMEAN - Local means
%
%   m = mylocalmean(f)
%   m = mylocalmean(f,nhood)

%% �������
narginchk(1,2);
nargoutchk(0,1);

%% ȱʡ��������
if nargin<2
    nhood = ones(3)/9;   %Ĭ��3*3
else
    nhood = nhood/sum(nhood(:));
end

%% ����ֲ���ֵ
m = imfilter(f,nhood,'replicate');


