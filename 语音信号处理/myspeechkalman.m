function y = myspeechkalman(x,Q,maxiter)
%MYSPEECHKALMAN - Speech enhancement by kalman filter
%
%   y = myspeechkalman(x)
%   y = myspeechkalman(x,Q)
%   y = myspeechkalman(x,Q,maxiter)

%% ��������Ŀ
narginchk(1,3);
nargoutchk(0,1);

%% ȱʡ��������
if nargin < 2
    Q = [];
end
Q_isempty = isempty(Q);
if nargin < 3
    maxiter = 1;
end

%% ����
p = 10;
nwin = 160; %֡����20ms for fs=8000Hz
noverlap = round(nwin/2); %֡�ص���
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
label = myendpointdetect(frame,8000,M,Z,[median(M),max(Z)],noverlap);

%% �׼�����Px��Pn
Px = zeros(nframe,nfft);
Py = abs(fft(frame,nfft,2)).^2; %����źŹ�����
Pn = mean(Py(~label,:));  
for i=1:nframe
    % �׼�
    Ps = Py(i,:)-4*Pn;
    Ps(Ps<0) = 0.001*Pn(Ps<0);
    Px(i,:) = Ps;
end

%% �������˲�
yframe = zeros(nframe,nwin);
x0 = zeros(p,1); %��ʼ��ֵ  
P0 = zeros(p);   %��ʼЭ����  
R = mean(Pn)/nwin; %�۲�����Э����
for i=1:nframe
    %lpc����
    [a,e,~] = mylpc(Px(i,:),p,'P'); %��������֡�Ĺ����׼�������غ������ټ���lpcϵ��a
    %�������˲�
    A = zeros(p);
    A(1:p-1,2:p) = eye(p-1);
    A(p,:) = flipud(-a(2:end));
    H = [zeros(1,p-1),1];
    if Q_isempty
        Q = e.^2; %��������Э����
    end
    G = H';
    for iter=1:maxiter %�����������˲�
        temp = mykalman({A,0,H},x0,P0,[],frame(i,:),Q*(G*G'),R);
        frame(i,:) = temp(p,:);
        [a,e,~] = mylpc(frame(i,:),p); %ֱ����������֡�ļ�������غ������ټ���lpcϵ��a  
        A(p,:) = flipud(-a(2:end));
        if Q_isempty
            Q = e.^2; %��������Э����
        end
    end
    yframe(i,:) = frame(i,:);
end

%% ֡�ϲ�
y = mydeframing(yframe);
y = mymovefilt(y,5); %����ƽ��

%% ��һ��
ny = length(y);
if ny >= nx
    y = y(1:nx);
else
    y = [y;zeros(nx-ny,1)];
end
y = y./max(abs(y));


