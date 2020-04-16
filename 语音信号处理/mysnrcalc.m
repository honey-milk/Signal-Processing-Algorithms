function snr = mysnrcalc(x,mix)
%MYSNRCALC - Calculate SNR
%
%   snr = mysnrcalc(x,mix)

%% ��������Ŀ
narginchk(2,2);
nargoutchk(0,1);

%% ���������
noise = mix-x; %����
x = x-mean(x); %ȥֱ��
noise = noise-mean(noise); %ȥֱ��
Ex = sum(x.^2); %�źŵ�����
En = sum(noise.^2); %����������
snr = 10*log10(Ex/En); %�źŵ�����������������֮�ȣ�����ֱ�ֵ


