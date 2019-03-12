function g = mylocalthresh(f,nhood,a,b,meantype)
%MYLOCALTHREESH - Local thresholding
%
%   g = mylocalthresh(f,nhood,a,b)
%   g = mylocalthresh(f,nhood,a,b,meantype)

%% �������
narginchk(4,5);
nargoutchk(0,1);

%% ȱʡ��������
if nargin<5
    meantype = 'global';   %Ĭ��ȫ����ֵ
end

%% �����׼��
sigma = stdfilt(f,nhood);

%% �����ֵ
if strcmpi(meantype,'global')
    m = mean2(f);
else
    m = mylocalmean(f,nhood);
end

%% ��ֵ��
g = (f > a*sigma) & (f > b*m);


