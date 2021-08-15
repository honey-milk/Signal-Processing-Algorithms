%% mydtw_demo.m
%% ��̬ʱ�����
%%
clc,clear;
close all;

%% ������������
N = 100;
T1 = 2;
T2 = 1;
t1 = (1:N)*T1/N;
t2 = (1:N)*T2/N;
t1 = t1(:);
t2 = t2(:);
x = sin(2*pi/T1*t1);
y = sin(2*pi/T2*t2);

%% ��ʾ��������
figure;
plot(t1,x,'r');
hold on;
plot(t2,y,'g');
title('ԭʼ����');

%% DTWƥ��
[~,indices] = mydtw(x,y);

%% ����ƥ����
figure;
plot(t1,x,'r');
hold on;
plot(t1(indices(:,1)),y(indices(:,2)),'.g');
title('ƥ����');
