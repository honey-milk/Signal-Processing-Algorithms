function k = myfftopticalflow(fixing, moving, kSize)
%MYFFTOPTICALFLOW - Calculate optical flow by FFT
%
%   k = myfftopticalflow(fixing, moving)
%   k = myfftopticalflow(fixing, moving, [M, N])

%% �������
narginchk(2,3);
nargoutchk(0,1);

%% ȱʡ��������
if nargin < 3
    kSize = [5, 5];
end
M = kSize(1);
N = kSize(2);

%% FFT
% fixing image
F1 = fftshift(fft2(im2double(fixing))); 
Phase1 = angle(F1);
% moving image
F2 = fftshift(fft2(im2double(moving))); 
Phase2 = angle(F2);

%% ��λ���
deltaPhase = Phase1 - Phase2;
deltaPhase = mod(deltaPhase, 2 * pi);

%% ����λ���ݶ�
[Fx,Fy] = gradient(deltaPhase);
Fx = medfilt2(Fx, [1, N]); % ��ֵ�˲�����ȥ��ϵ�Ӱ��
Fy = medfilt2(Fy, [M, 1]);
[rows, cols] = size(deltaPhase);
Fx = Fx * cols / (2 * pi); % ����ƶ���x0
Fy = Fy * rows / (2 * pi); % ����ƶ���y0
H = Fx + Fy * 1i;

%% ����ƽ���ݶ���Ϊ����
k = mean(H(:)); % x0 + j y0

    