% myextendedkalman_demo.m
%%
clc,clear;
close all;

%% ����ֱ���˶�����
nx = 100;  
T = 10;
dt = T / nx;
t = (1:nx) * dt;  
v = [1;1]; % �ٶ�
x0 = [0;0]; % ��λ��
x_gt = x0 + v * t;
z = [sqrt(sum((x_gt-[0;10]).^2));sqrt(sum((x_gt-[10;0]).^2))] + 0.1 * randn(2,nx); 

%% ������ʵ�켣
figure;
plot(x_gt(1,:),x_gt(2,:));
figure;
subplot(2,1,1);plot(z(1,:));
subplot(2,1,2);plot(z(2,:));

%% ��չ�������˲�
dimx = 4;
dimz = 1;
x0 = zeros(dimx,1); % ��ʼ��ֵ  
P0 = eye(dimx);     % ��ʼЭ����  
Q = 0.01 * eye(dimx);  % ��������Э����  
R = 1 * eye(dimz);  % �۲�����Э����  
symx = sym('x',[dimx,1],'real');
symu = sym('u','real');
symf = [1,0,dt,0;0,1,0,dt;0,0,1,0;0,0,0,1] * symx;
symg = [0;0;0;0] * symu;
symh = [sqrt(symx(1)^2+(symx(2)-10)^2);sqrt((symx(1)-10)^2+symx(2)^2)];
x = myextendedkalman({symf,symg,symh},x0,P0,[],z,Q,R);
  
%% ��ͼ
figure; 
plot(x_gt(1,:),x_gt(2,:),'r',x(1,:),x(2,:),'g');
legend('��ʵֵ','���Ź���ֵ'); title('λ�ƹ���');

