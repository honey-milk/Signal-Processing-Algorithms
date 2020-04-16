%% myfastica_demo.m
%% Fast ICAʾ��
%%
clc,clear;
close all;

%% ����ԭʼ�ź�
fs = 1000;
T = 1;
t = 0:1/fs:T;
m = 4; % �ź�����
n = length(t); % �źų���
S = zeros(m,n); % ԭʼ�ź�
S(1,:) = sin(2*pi*10*t); % �ź�1
S(2,:) = square(2*pi*13*t); % �ź�2
S(3,:) = sawtooth(2*pi*10*t).^2; % �ź�3
S(4,:) = 0.1*randn(1,n); % �ź�4
% ��ʾԭʼ�ź�
y = S;
figure;
subplot(4,1,1);
plot(t,y(1,:));
title('ԭʼ�ź�');
subplot(4,1,2);
plot(t,y(2,:));
subplot(4,1,3);
plot(t,y(3,:));
subplot(4,1,4);
plot(t,y(4,:));

%% ����������
A = rand(m); %������ɻ�Ͼ���
X = A*S;
% ��ʾ�������
y = X;
figure;
subplot(4,1,1);
plot(t,y(1,:));
title('�������');
subplot(4,1,2);
plot(t,y(2,:));
subplot(4,1,3);
plot(t,y(3,:));
subplot(4,1,4);
plot(t,y(4,:));

%% FastICA��ȡ�����ɷ�
maxiter = 100; % ��������
S = myfastica(X,maxiter);

%% ��ʾ���
y = S;
figure;
subplot(4,1,1);
plot(t,y(1,:));
title('������ź�');
subplot(4,1,2);
plot(t,y(2,:));
subplot(4,1,3);
plot(t,y(3,:));
subplot(4,1,4);
plot(t,y(4,:));
