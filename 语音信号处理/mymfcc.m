function varargout = mymfcc(x,fs)
%MYMFCC - Mel-Frequency Cepstral Coefficients
%
%   mfc = mymfcc(x,fs)
%   [mfcc,dmfcc,ddmfcc] = mymfcc(x,fs)

%% �����������
narginchk(2,2);
nargoutchk(1,3);

%% ����
n = 24; %Mel�˲�������
p = 12; %���׽���
nwin = 256; %֡��

%% ���Mel�˲�����
bank = melbankm(n,nwin,fs,0,0.5,'t'); 
% ��һ��mel�˲�����ϵ��
bank = full(bank);
bank = bank/max(bank(:));

%% ��һ��������������
w = 1+0.5*p*sin(pi*(1:p)./p);
w = w/max(w);

%% Ԥ�����˲���
x = double(x);
x = filter([1,-0.9375],1,x);

%% �����źŷ�֡
frame = myframing(x,nwin);
nframe = size(frame,1);
%% ����ÿ֡��MFCC����
mfccmat = zeros(nframe,p+1);
for i=1:nframe
    y = frame(i,:)';
    y = y.* hamming(nwin); %�Ӵ�
    energy = log(sum(y.^2)+eps); %����
    y = abs(fft(y));
    y = y(1:fix(nwin/2)+1);
    c = dct(log(bank*y+eps));
    c = c(2:p+1)'.*w; %ȡ2~p+1��ϵ��
    mfcc = [c,energy];
    mfccmat(i,:) = mfcc;
end

%% һ�ײ��MFCCϵ��
dmfccmat = zeros(nframe,p+1);
for i=2:nframe-1
  dmfccmat(i,:) = mfccmat(i,:)-mfccmat(i-1,:);
end

%% ���ײ��MFCCϵ��
ddmfccmat = zeros(nframe,p+1);
for i=3:nframe-2
  ddmfccmat(i,:) = dmfccmat(i,:)-dmfccmat(i-1,:);
end

%% �ϲ�MFCC��һ�ס����ײ��ϵ��
mfc = [mfccmat,dmfccmat,ddmfccmat];
%ȥ����β����֡����Ϊ����֡�Ķ��ײ�ֲ���Ϊ0
mfc = mfc(3:nframe-2,:);

%% �������
switch(nargout)
    case 1
        varargout = {mfc};
    case 2
        varargout = {mfccmat,dmfccmat};
    case 3
        varargout = {mfccmat,dmfccmat,ddmfccmat};
end
