%% mylsinterpolation_demo.m
%% LS��ֵ
%% 
clear,clc;
close all;

%% �����ź�
N = 50;
n = 1:N;
s = sin(2*pi*n/N) + sin(2*pi*3*n/N);
s = s';

%% �۲�
m = 20;
n = 30;
H = eye(N);
Sx = H([1:m-1,n+1:end],:);
Sc = H(m:n,:);
x = Sx * s;

%% LS��ֵ
D = toeplitz([-2,1,zeros(1,N-2)]);
v = -(Sc * (D' * D) * Sc') \ Sc * (D' * D) * Sx' * x; 
y = Sx' * x + Sc' * v;

%% ��ʾ
figure;
subplot(3,1,1);plot(s,'.');title('��ʵ�ź�');
subplot(3,1,2);plot(x,'.');title('�۲��ź�');
subplot(3,1,3);plot(y,'.');title('��ֵ�ź�');
