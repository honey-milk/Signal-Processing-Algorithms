function [x,x_e,K] = myextendedkalman(sys,x0,P0,u,z,Q,R)
%MYEXTENDEDKALMAN - Extended Kalman filtering
%
%   [x,x_e,K] = myextendedkalman(sys,x0,P0,u,z,Q,R)
%   [x,x_e,K] = myextendedkalman(sys,x0,P0,[],z,Q,R)

%Ԥ�ⷽ�̣�x(n+1) = f(x(n)) + g(u[n]) + w[n]
%�������̣�z(n) = h(x(n)) + v[n]
%״̬���̵ľ���sys = {f,g,h}

%% ��������Ŀ
narginchk(7,7);
nargoutchk(0,3);

%% ������ȡ
if isempty(u)
    u = zeros(size(z));
end
nx = size(z,2);
symf = sys{1};
symg = sys{2};
symh = sys{3};
symx = symvar(symf)';
symu = symvar(symg)';

%% �����ſɱȾ���
dimx = size(x0,1);
dimz = size(z,1);
symF = sym('df',[dimx,dimx]);
symH = sym('dh',[dimz,dimx]);
for i=1:dimx
    symF(:,i) = diff(symf,symx(i));
    symH(:,i) = diff(symh,symx(i));
end

%% �������˲��㷨
x_e = zeros(dimx,nx);       % ��ֵ����ֵ
P_e = zeros(dimx,dimx,nx);  % Э�������ֵ
x = zeros(dimx,nx);         % ���������ţ���ֵ����ֵ
P = zeros(dimx,dimx,nx);    % ���������ţ�Э�������ֵ
x_e(:,1) = x0;
P_e(:,:,1) = P0;
x(:,1) = x0;
P(:,:,1) = P0;
for i=2:nx
    % ��n-1ʱ��״̬����nʱ�̵�״̬
    x_e(:,i) = subs(symf,symx,x(:,i-1)) + subs(symg,symu,u(:,i));
    F = subs(symF,symx,x(:,i-1));
    P_e(:,:,i) = F * P(:,:,i-1) * F' + Q;
    % ���㿨��������
    H = subs(symH,symx,x_e(:,i));
    K = P_e(:,:,i) * H'/ (H * P_e(:,:,i) * H' + R);
    % ��������ֵ
    x(:,i) = x_e(:,i) + K * (z(:,i) - subs(symh,symx,x_e(:,i)));
    P(:,:,i) = (eye(dimx) - K * H) * P_e(:,:,i);
end







