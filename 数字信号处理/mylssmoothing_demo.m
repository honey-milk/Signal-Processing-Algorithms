%% mylssmoothing_demo.m
%% LSƽ��
%% 
clear,clc;
close all;

%% �����ź�
N = 50;
n = 1:N;
s = sin(2*pi*n/N)';

%% ������
x = s + 0.2 * randn(N,1);

%% LSƽ��
lambda = 10;
D = toeplitz([-2,1,zeros(1,N-2)]);
y = (eye(N) + lambda * (D' * D)) \ x;

%% ��ʾ
figure;
subplot(3,1,1);plot(s);title('ԭʼ�ź�');
subplot(3,1,2);plot(x);title('�����ź�');
subplot(3,1,3);plot(y);title('ƽ���ź�');
