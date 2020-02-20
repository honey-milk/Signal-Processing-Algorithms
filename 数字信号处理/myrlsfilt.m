function [y,e,w] = myrlsfilt(x,d,n,lambda)
%MYRLSFILT - rls filtering
%
%   This MATLAB function applies an order-n adaptive filter to the
%   input vector, x, basing on RLS.
%   
%   y = myrlsfilt(x,d,n,lambda)

%% �������
if isvector(x)
    if n > length(x)
        error('����Ӧ�˲�������n����С�ڵ����ź�x�ĳ���');
    end
else
    error('�������x������һά����');
end

%% ����������㷨
% �����ź���ʱ
nx = length(x);
X = zeros(n,nx); 
X(1,:) = x;
if n > 0
    for i=2:n
        X(i,:) = filter([0,1],1,X(i-1,:)); %��ʱһ����λ
    end
end
% ��ʼ������
w = zeros(1,n);     %�˲���ϵ��
y = zeros(1,nx);    %�˲����
e = zeros(1,nx);    %���
P = 0.01 * eye(n);
% �˲�
for i=1:nx
    y(i) = w * X(:,i);        %�����˲����
    e(i) = d(i) - y(i);       %�������
    K = P * X(:,i) / (lambda + X(:,i)' * P *  X(:,i)); % ��������
    P = lambda^(-1) * (eye(n) - K * X(:,i)') * P;
    w = w + K' * e(i);        %�����˲���ϵ�� 
end    


