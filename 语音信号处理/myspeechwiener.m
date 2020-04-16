function y = myspeechwiener(x,param)
%MYSPEECHWIENER - Speech enhancement by wiener filter
%
%   y = myspeechwiener(x)
%   y = myspeechwiener(x,[a,b,alpha,beta])

%% ��������Ŀ
narginchk(1,2);
nargoutchk(0,1);

%% ȱʡ��������
if nargin < 2
    param = [1,0,1,1];
end
a = param(1);
b = param(2);
alpha = param(3);
beta = param(4);

%% ����
nwin = 160; %֡����20ms for fs=8000Hz
noverlap = 80; %֡�ص���
nx = length(x);
nfft = 512;

%% ��֡�Ӵ�
frame = myframing(x,nwin,noverlap);
nframe = size(frame,1);
win = repmat(hamming(nwin)',nframe,1);
frame = frame.*win;

%% ����֡����
%�˵���
[~,M,Z,~] = mytimefeature(x,8000,nwin,noverlap);
label = myendpointdetect(frame,8000,M,Z,[median(M),max(Z)+eps],noverlap);

%% ά���˲�(Ƶ��)
yframe = zeros(nframe,nwin); %������֡
amp = abs(fft(frame,nfft,2)); %����źŷ�����
phase = angle(fft(frame,nfft,2)); %����ź���λ��
Py = amp.^2; %����źŹ�����
Pn = mean(Py(~label,:));    
for i=1:nframe
    % �׼�
    Px = Py(i,:)-a*Pn;
    Px(Px<0) = b*Pn(Px<0);
    % ����H(w)
    H = (Px./(Px+alpha*Pn)).^beta;
    Y = H.*amp(i,:); %Ƶ���˲�
    temp = real(ifft(Y.*exp(1i*phase(i,:)),nfft));
    yframe(i,:) = temp(1:nwin);
end

%% ֡�ϲ�
y = mydeframing(yframe,noverlap);

%% ��һ��
ny = length(y);
if ny >= nx
    y = y(1:nx);
else
    y = [y;zeros(nx-ny,1)];
end
y = y./max(abs(y));
