function [a,e,k] = mylpc(x,p,option)
%MYLPC - Linear prediction filter coefficients
%
%   This MATLAB function finds the coefficients of a pth-order linear predictor (FIR
%   filter) that predicts the current value of the real-valued time series x based
%   on past samples.
%
%   a = mylpc(x,p)
%   a = mylpc(P,p,option)
%   [a,e] = mylpc(...)
%   [a,e,k] = mylpc(...)

%% ��������Ŀ
narginchk(2,3);
nargoutchk(0,3);

%% ȱʡ��������
if nargin<3
    option = 'x';
end

%% ��������
if p>=length(x)
    error('����pֵ����С��x����P���ĳ���');
end

%% ����LPC
% ��������غ���
if strcmpi(option,'x')
    r = ifftshift(xcorr(x)); %����ʱ���ź�
elseif strcmpi(option,'P')
    r = ifft(x); %���빦����
else
    error('option����Ϊ''x''��''p''');
end
r = r(1:p+1);
r = r(:);
a = 1;
epsilon = r(1);
for i=2:p+1
    if epsilon==0
        epsilon = eps;
    end
    gamma = -r(2:i)'*flipud(a)/epsilon;
    a = [a;0]+gamma*[0;conj(flipud(a))];
    epsilon = epsilon*(1-abs(gamma)^2);
end
e = epsilon;
k = gamma;


