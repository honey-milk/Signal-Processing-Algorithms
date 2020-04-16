function x = mySIR(sys, x0, P0, u, z, Q, R, N)
%MYSIR - Sampling Importance Resampling Filter
%
%   x = mySIR(sys, x0, P0, u, z, Q, R, N)
%   x = mySIR(sys, x0, P0, [], z, Q, R, N)

%Ԥ�ⷽ�̣�x(n+1) = Ax(n)+Bu[n]+w[n]
%�������̣�z(n) = Hx(n)+v[n]
%״̬���̵ľ���sys = {A,B,H}

%% ��������Ŀ
narginchk(8,8);
nargoutchk(0,1);

%% ������ȡ
A = sys{1};
B = sys{2};
H = sys{3};
if isempty(u)
    u = zeros(size(z));
end
dx = size(x0, 1);
nx = size(z, 2);

%% SIR�㷨
x_P = x0 + sqrt(P0) * randn(dx, N);   % ��ʼ��N������
x = zeros(dx, nx);                    % �˲����
x(:,1) = x0;
for i=2:nx
    % ������p(x(k)|x(k-1))�в����õ�����  
    x_P_update = A * x_P + B * u(:,i) + sqrt(Q) * randn(dx, N);% ���º������
    % ����p(y(k)|x(k))��������Ȩ��
    z_update = H * x_P_update;                          % ���Ӷ�Ӧ�Ĺ۲�ֵ
    P_w = normpdf(z_update, z(:,i), sqrt(R));           % ���ӵ�Ȩ��
    % ��һ��P_w
    P_w = P_w ./ sum(P_w);
    P_w = cumsum(P_w);
    % �ز���
    for j=1:N    
        idx = find(rand <= P_w, 1);
        x_P(:, j) = x_P_update(:, idx);
    end 
    % ״̬����
    x(:,i) = mean(x_P, 2);
end
