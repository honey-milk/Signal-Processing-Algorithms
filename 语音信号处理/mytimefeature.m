function [frame,M,Z,T] = mytimefeature(x,fs,varargin)
%MYTIMEFEATURE - short-time time domain feature
%
%   This MATLAB function returns [E,M,Z], the short-time time domain freture of the input
%   signal vector x.
%
%   [frame,M,Z,T] = mytimefeature(x,fs)
%   [frame,M,Z,T] = mytimefeature(x,fs,nwin)
%   [frame,M,Z,T] = mytimefeature(x,fs,nwin,noverlap)
%   [frame,M,Z,T] = mytimefeature(x,fs,nwin,noverlap,[threshm,threshz])
%   [frame,M,Z,T] = mytimefeature(x,fs,nwin,noverlap,[threshm,threshz],disp)
%   [frame,M,Z,T] = mytimefeature(...,option)
%   mytimefeature(...)

%   option��{'truncation'},'padding'  

%% ������Ŀ���
narginchk(2,7);
nargoutchk(0,4);

%% ������ȡ
%�ضϡ�����ѡ��
option = 'truncation';
if (nargin > 2 && ischar(varargin{end})) && any(strcmpi(varargin{end},{'truncation','padding'})),
    option = varargin{end};
    varargin(end)=[];
end
%��ȡʣ�����������Ŀ
narg = numel(varargin);
%�������
nwin = 160;     %20ms (fs=8000Hz)
noverlap = round(nwin/2);
disp = false;
thresh = [0,0];
%��ȡ�������ֵ
switch narg
    case 0
    case 1
        nwin = varargin{:};      
    case 2
        [nwin,noverlap] = varargin{:};
    case 3
        [nwin,noverlap,thresh] = varargin{:};
    case 4
        [nwin,noverlap,thresh,disp] = varargin{:};
    otherwise
        error('�����������');
end

%% ����ֵ���
%��xתΪ������
if isvector(x)==1
    x = x(:);
else
    error('�������''x''����Ϊ1ά����');
end
%֡�ص�����noverlap
if noverlap >= nwin
    error('''noverlap''��ֵ����С��''window''�ĳ���');
end
%��ֵthresh
threshm = 0;
threshz = 0;
if length(thresh)==2
    threshm = thresh(1);
    threshz = thresh(2);
end

%% ��֡
frame = myframing(x,nwin,noverlap,option);
nframe = size(frame,1);
for i=1:nframe
    %ȥֱ������
    frame(i,:) = frame(i,:)-mean(frame(i,:));
end

%% ����ʱ������
%��ʱƽ������
M = zeros(nframe,1);
for i=1:nframe
    M(i) = sum(abs(frame(i,:)));
end
%��ʱ������
Z = zeros(nframe,1);
for i=1:nframe
    Z(i) = 0.5*sum(abs(sign(frame(i,2:nwin))-sign(frame(i,1:(nwin-1)))));
end

%% ����ÿһ֡���м��ʱ��(T)
nstride = nwin-noverlap;
T=zeros(nframe,1);
for i=1:nframe
    start_time=(i-1)*nstride;
    T(i)=start_time+nwin/2;
end
T=T/fs;

%% û�����������disp==true������źŲ���
if nargout==0 || disp
    plot(T,M,'r',T,Z,'g');
    strlegend = {'��ʱƽ������','��ʱ������'};
    hold on;
    if threshm~=0
        ThM = T;
        ThM(:) = threshm;
        plot(T,ThM,'r','LineWidth',2);
        strlegend = cat(2,strlegend,'��ʱƽ��������ֵ');
    end  
    if threshz~=0
        ThZ = T;
        ThZ(:) = threshz;
        plot(T,ThZ,'g','LineWidth',2);
        strlegend = cat(2,strlegend,'��ʱ��������ֵ');
    end
    hold off;
    legend(strlegend);
    xlabel('ʱ��(s)');
    ylabel('����');
    title('ʱ����');
end




