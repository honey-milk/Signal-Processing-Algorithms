%mykalman_demo.m

%%
clc,clear;
close all;

%% ����
N = 300;  
t = 1:N;  
CON = 25;%�����¶ȣ��ٶ��¶��Ǻ㶨��  

%% �������˲� 
z = CON+randn(1,N); %�������Ĺ۲�ֵ 
x0 = 1; %��ʼ��ֵ  
P0 = 10; %��ʼЭ����  
Q = cov(randn(1,N));%��������Э����  
R = cov(randn(1,N));%�۲�����Э����  
[x,x_e,G] = mykalman({1,0,1},x0,P0,[],z,Q,R);
  
%% ��ͼ
figure;  
expValue = zeros(1,N);  
expValue(:) = CON;  
plot(t,expValue,'r',t,x_e,'m',t,z,'b',t,x,'g');  
legend('��ʵֵ','����ֵ','�۲�ֵ','���Ź���ֵ');  

%% ����
fs = 1;
T = 100;
t = 0:1/fs:T;
nx = length(t);

% %% �������˲���������ֱ���˶�����
% k = 2;
% b = -5;
% u = k*t+b; %��ʵ�ź�
% v = 0*t+k;
% % noise = 10*randn(1,nx); 
% noise = 20*sin(2*pi*0.1*t); 
% z = u+noise; %�������Ĺ۲�ֵ 
% x0 = zeros(2,1); %��ʼ��ֵ  
% P0 = ones(2); %��ʼЭ����  
% Q = 0.0001; %��������Э����  
% R = 1000; %�۲�����Э����  
% [x,x_e,G] = mykalman({[1,1;0,1],[0;0],[1,0]},x0,P0,[],z,Q,R);
%   
% %% ��ͼ
% figure;  
% plot(t,u,'r.-',t,x_e(1,:),'k.-',t,z,'b.-',t,x(1,:),'g.-');  
% legend('��ʵֵ','����ֵ','�۲�ֵ','���Ź���ֵ');
% title('λ�ƹ���');
% figure;
% plot(t,v,'r.-',t,x_e(2,:),'k.-',t,x(2,:),'g.-');  
% legend('��ʵֵ','����ֵ','���Ź���ֵ');
% title('�ٶȹ���');
% figure;  
% plot(t,x(1,:),'g.-');  
% title('λ�����Ź���ֵ');

