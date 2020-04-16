function y = myspecsub(x,param)
%MYSPETSUB - Spectral Subtraction
%
%   y = myspecsub(x)
%   y = myspecsub(x,[a,b])

%% ��������Ŀ
narginchk(1,2);
nargoutchk(0,1);

%% ȱʡ��������
if nargin < 2
    param = [1,0];
end
a = param(1);
b = param(2);

%% ����
nwin = 160; %֡����20ms for fs=8000Hz
noverlap = round(nwin/2); %֡�ص���
nx = length(x);
nfft = 512;

% %% Ԥ���� R=1-Z^-1
% w = [1,-0.95];
% x = filter(w,1,x);

%% ��֡�Ӵ�
frame = myframing(x,nwin,noverlap);
nframe = size(frame,1);
win = repmat(hamming(nwin)',nframe,1);
frame = frame.*win;

%% ����֡����
%�˵���
[~,M,Z,~] = mytimefeature(x,8000,nwin,noverlap);
label = myendpointdetect(frame,8000,M,Z,[median(M),max(Z)+eps],noverlap);

%% �׼�
yframe = zeros(nframe,nwin); %������֡
Py = abs(fft(frame,nfft,2)).^2; %����źŹ�����
phase = angle(fft(frame,nfft,2)); %����ź���λ��
Pn = mean(Py(~label,:));    
for i=1:nframe
    % �׼�
    Px = Py(i,:)-a*Pn;
    Px(Px<0) = b*Pn(Px<0);
    Ax = sqrt(Px);
    % ������λ
    temp = real(ifft(Ax.*exp(1i*phase(i,:)),nfft));
    yframe(i,:) = temp(1:nwin);
end

%% ֡�ϲ�
y = mydeframing(yframe);

% %% ȥ���� R=1/(1-Z^-1)
% w = [1,-0.95];
% y = filter(1,w,y);

%% ��һ��
ny = length(y);
if ny >= nx
    y = y(1:nx);
else
    y = [y;zeros(nx-ny,1)];
end
y = y./max(abs(y));
