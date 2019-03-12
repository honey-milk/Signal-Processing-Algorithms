function varargout = mylpanalysis(varargin)
%MYLPANALYSIS - Signal analysis by linear prediction
%
%   This MATLAB function finds the parameters of the system and 
%   excitation by linear predition.
%
%   lpcparam = mylpanalysis(x,fs,nwin,p)

%% ��������
% ��������Ŀ
narginchk(4,4);
nargoutchk(0,1);

% ��ȡ�������ֵ
[x,fs,nwin,p] = varargin{:};

% ����������ֵ
% ������x
if isvector(x)==1
    x = x(:);   % ��xתΪ������
else
    error('�������x����Ϊ1ά����');
end
% ������p
if p>=nwin
    error('�������p����С��nwin');
end

%% Ԥ����
% R=1-b*Z^-1
x = filter([1,-0.95],1,x);

%% ��֡
frame = myframing(x,nwin,0,'truncation');
% ֡��
nframe = size(frame,1);

%% LP����
a = zeros(nframe,p+1);      %����Ԥ��ϵ��
amp = zeros(nframe,1);      %����������
e = zeros(nframe,nwin);
for i=1:nframe
    % lpc����
    a(i,:) = mylpc(frame(i,:),p);
    predx = filter([0;-a(i,2:end)],1,frame(i,:));
    e(i,:) = frame(i,:)-predx;
    amp(i) = max(abs(e(i,:)));
end
[period,T] = mypitchtrack(frame,fs);  %������������

%% ����������
lpcparam = struct('fs',[],'nwin',[],'nframe',[],...
    'p',[],'a',[],'period',[],'amp',[]);
lpcparam.fs = fs;
lpcparam.nwin = nwin;
lpcparam.nframe = nframe;
lpcparam.p = p;
lpcparam.a = a;
lpcparam.T = T;
lpcparam.period = period;
lpcparam.amp = amp;
varargout = {lpcparam};
