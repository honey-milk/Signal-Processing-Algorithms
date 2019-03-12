function [x,x_e,G] = mykalman(sys,x0,P0,u,z,Q,R)
%MYKALMAN - Kalman filtering
%
%   [x,x_e,G] = mykalman(sys,x0,P0,u,z,Q,R)
%   [x,x_e,G] = mykalman(sys,x0,P0,[],z,Q,R)

%Ԥ�ⷽ�̣�x(n+1) = Ax(n)+Bu[n]+w[n]
%�������̣�z(n) = Hx(n)+v[n]
%״̬���̵ľ���sys = {A,B,H}

%% ��������Ŀ
narginchk(7,7);
nargoutchk(0,3);

%% ������ȡ
A = sys{1};
B = sys{2};
H = sys{3};
if isempty(u)
    u = zeros(size(z));
end
nx = size(z,2);

%% �������˲��㷨
x_e = zeros(size(A,2),nx); %��ֵ����ֵ
P_e = zeros(size(P0,1),size(P0,2),nx); %Э�������ֵ
x = zeros(size(A,2),nx); %���������ţ���ֵ����ֵ
P = zeros(size(P0,1),size(P0,2),nx); %���������ţ�Э�������ֵ
x_e(:,1) = x0;
P_e(:,:,1) = P0;
x(:,1) = x0;
P(:,:,1) = P0;
for i=2:nx
    %��nʱ��״̬����n+1ʱ�̵�״̬
    x_e(:,i) = A*x(:,i-1)+B*u(:,i);
    P_e(:,:,i) = A*P(:,:,i-1)*A'+Q;
    %���㿨��������
    G = P_e(:,:,i)*H'/(H*P_e(:,:,i)*H'+R);
    %��������ֵ
    x(:,i) = x_e(:,i)+G*(z(:,i)-H*x_e(:,i));
    P(:,:,i) = (eye(size(P0,1),size(P0,2))-G*H)*P_e(:,:,i);
end







