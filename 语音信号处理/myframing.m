function varargout = myframing(varargin)
%MYFRAMING - Transfer a signal vector into a matrix of data frames
%
%   This MATLAB function transfers a signal into data frames 
%
%   frame = myframing(x,nwin)
%   frame = myframing(x,nwin,noverlap)
%   frame = myframing(...,option)

%   option can be 'truncation' or 'padding', the default value is
%   'truncation'.

%% ��������
% ��������Ŀ
narginchk(2,4);
nargoutchk(0,1);

% ��ʼ���������
noverlap = [];
option = 'truncation';

% ��ȡ�������option
if (nargin > 2 && ischar(varargin{end})) && any(strcmpi(varargin{end},{'truncation','padding'})),
    option = varargin{end};
    varargin(end)=[];
end

% ��ȡʣ���������ֵ
narg = numel(varargin); % ��ȡʣ�������������
switch narg
    case 2
        [x,nwin] = varargin{:};
    case 3
        [x,nwin,noverlap] = varargin{:};
end

% ����������ֵ
% ������x
if isvector(x)==1
    x = x(:);   % ��xתΪ������
    nx = length(x);
else
    error('�������x����Ϊ1ά����');
end
% ������noverlap
if isempty(noverlap)
    noverlap = round(nwin/2);
elseif (noverlap>=nwin) || (noverlap<0)
    error('noverlap��ֵ����С��nwin��ֵ��Ϊ�Ǹ�ֵ');
end

%% ��֡Ԥ����
% ֡��
nstride = nwin-noverlap; 
% �ź�x���ֳܷ�����֡�����ýضϴ�ʩ 
if strcmpi(option,'truncation')
    nframe = fix((nx-noverlap)/nstride);   %֡��
% �ź�x���ֳܷ�����֡�����ò����ʩ   
else
    nframe = ceil((nx-noverlap)/nstride);  %֡��
    npadding = nframe*nstride+noverlap-nx;
    x = [x;zeros(npadding,1)];  %ĩβ����
end

%% ��֡
frame = zeros(nframe,nwin);
for i=1:nframe
    start_index = (i-1)*nstride+1;
    end_index = start_index+nwin-1;
    frame(i,:) = x(start_index:end_index);
end

%% ���ط�֡���
varargout = {frame};

