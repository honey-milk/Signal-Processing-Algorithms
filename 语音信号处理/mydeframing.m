function varargout = mydeframing(varargin)
%MYDEFRAMING - Transfer a matrix of data frames into a signal vector.
%
%   This MATLAB function transfers data frames into a signal 
%
%   x = mydeframing(frame)
%   x = mydeframing(frame,noverlap)

%% ��������
% ��������Ŀ
narginchk(1,2);
nargoutchk(0,1);

% ��ʼ���������
noverlap = [];

% ��ȡ�������ֵ
switch nargin
    case 1
        frame = varargin{:};
    case 2
        [frame,noverlap] = varargin{:};
end

% ����������ֵ
% ������frame
if ismatrix(frame)
    [nframe,nwin] = size(frame);
else
    error('�������frame����Ϊ����');
end
% ������noverlap
if isempty(noverlap)
    noverlap = round(nwin/2);
elseif (noverlap>=nwin) || (noverlap<0)
    error('noverlap��ֵ����С��nwin��ֵ��Ϊ�Ǹ�ֵ');
end

%% �ϳ�x
% ֡��
nstride = nwin-noverlap; 
% x����
nx = nframe*nstride+noverlap;
x1 = zeros(nx,1);
x2 = zeros(nx,1);
for i=1:nframe %��������
    start_index = (i-1)*nstride+1;
    end_index = start_index+nwin-1;
    x1(start_index:end_index) = frame(i,:);
end
for j=1:nframe %����ǰ��
    i = nframe-j+1;
    start_index = (i-1)*nstride+1;
    end_index = start_index+nwin-1;
    x2(start_index:end_index) = frame(i,:);
end
x = (x1+x2)/2;

%% ���غϳɽ��x
varargout = {x};
