function [y,e,w] = mynlmsfilt(x,d,n,miu)
%MYLMSFILT - nlms filtering
%
%   This MATLAB function applies an order-n adaptive filter to the
%   input vector, x, basing on Normalized NLMS.
%   
%   y = mylmsfilt(x,d,n,miu)

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
w = zeros(1,n); %�˲���ϵ��
y = zeros(1,nx);  %�˲����
e = zeros(1,nx);  %���
% �˲�
for i=1:nx
    y(i) = w * X(:,i);        %�����˲����
    e(i) = d(i) - y(i);       %�������
    w = w + miu / (eps + X(:,i)' * X(:,i)) * e(i) * X(:,i)';   %�����˲���ϵ�� 
end    


