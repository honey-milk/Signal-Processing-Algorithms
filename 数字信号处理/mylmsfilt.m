function [y,e,w] = mylmsfilt(x,d,n,u)
%MYLMSFILT - lms filtering
%
%   This MATLAB function applies an order-n adaptive filter to the
%   input vector, x, basing on LMS.
%   
%   y = mylmsfilt(x,d,n,u)

%% �������
if isvector(x)
    if n>length(x)
        error('����Ӧ�˲�������n����С�ڵ����ź�x�ĳ���');
    end
else
    error('�������x������һά����');
end

%% ����������㷨
% �����ź���ʱ
nx = length(x);
X = zeros(n+1,nx); 
X(1,:) = x;
if n>0
    for i=2:n+1
        X(i,:) = filter([0,1],1,X(i-1,:)); %��ʱһ����λ
    end
end
% ��ʼ������
w = zeros(1,n+1); %�˲���ϵ��
y = zeros(1,nx);  %�˲����
e = zeros(1,nx);  %���
% �˲�
for i=1:nx
    y(i) = w*X(:,i);        %�����˲����
    e(i) = d(i)-y(i);       %�������
    w = w+2*u*e(i)*X(:,i)';   %�����˲���ϵ�� 
end    


