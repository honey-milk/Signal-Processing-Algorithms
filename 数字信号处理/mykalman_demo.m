%% mykalman_demo.m
%% �������˲�ʾ��
%%
clc,clear;
close all;

%% ����
N = 100;  
t = 1:N;  

%% �������˲���������ֱ���˶�����
k = 3;
b = 0;
u = k * t + b;      %��ʵ�ź�
v = 0 * t + k;
x0 = [0,0];         %��ʼ��ֵ  
P0 = ones(2);       %��ʼЭ����  
Q = 1 * ones(2,2);  %��������Э����  
R = 100;              %�۲�����Э����  
noise = sqrt(R) * randn(1, N); 
z = u + noise;        %�������Ĺ۲�ֵ 
[x,x_e,G] = mykalman({[1,1;0,1],[0;0],[1,0]},x0,P0,[],z,Q,R);
  
%% ��ͼ
figure;  
plot(t,u,'r.-',t,x_e(1,:),'k.-',t,z,'b.-',t,x(1,:),'g.-', 'linewidth', 2);  
legend('��ʵֵ','����ֵ','�۲�ֵ','���Ź���ֵ');
title('λ�ƹ���');
figure;
plot(t,v,'r.-',t,x_e(2,:),'k.-',t,x(2,:),'g.-');  
legend('��ʵֵ','����ֵ','���Ź���ֵ');
title('�ٶȹ���');
figure;  
plot(t,z,'b.-', t,x(1,:),'g.-');  
title('λ�����Ź���ֵ');
