function varargout = mylpsynthesis(varargin)
%MYLPSYNTHESIS - Signal synthesis by linear prediction
%
%   This MATLAB function syntheses the speech signal with the parameters
%   of the system and the excitation by linear predition.
%
%   x = mylpsynthesis(lpcparam)

%% ��������
% ��������Ŀ
narginchk(1,1);
nargoutchk(0,1);

% ��ȡ�������ֵ
lpcparam = varargin{:};
fs = lpcparam.fs;
nwin = lpcparam.nwin;
nframe = lpcparam.nframe;
a = lpcparam.a;
period = lpcparam.period;
amp = lpcparam.amp;

%% LP�ϳ�
synframe = zeros(nframe,nwin);
for i=1:nframe
    if period(i)>0     %����
        t = 1:nwin;
        d = 1:period(i)*fs:nwin;
        % �������������ڲ��������ź�
        e = pulstran(t,d,'rectpuls');
    else               %��������
        % �ð�������Ϊ�����ź�
        e = randn(nwin,1);
    end
    e = amp(i)*(e/max(abs(e)));
    synframe(i,:) = filter(1,a(i,:),e);
end
synx = mydeframing(synframe,0);

%% ȥ����
% R=1/(1-b*Z^-1)
synx = filter(1,[1,-0.95],synx);

%% ��һ��
amp = max(abs(synx));
if amp>1
    synx = synx/amp;
end

%% ����������
varargout = {synx};
