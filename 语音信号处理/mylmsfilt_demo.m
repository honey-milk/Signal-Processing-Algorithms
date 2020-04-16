%% mylmsfilt_demo.m
%% LMS��������
%%
clc,clear;
close all;

%%
addpath('../�����źŴ���');

%% ��ȡ�����ź�
[s, ~] = audioread('../src/sound/handel.wav');
[x,fs] = audioread('../src/sound/handel_echo.wav');
s = s';
x = x';
nx = length(x);
t = (1:nx) / fs;

%% ���������ź�
% pause;
% soundsc(s,fs);
% pause;
% soundsc(x,fs);

%% LMS����Ӧ�˲�
p = 3000;
[y,e,w] = mylmsfilt(x(1:nx-p),x(p+1:nx),1,0.01);
y = [zeros(1,p), y];
e = [x(1:p), e];

%% ����ʱ����
figure;
subplot(4,1,1)
plot(t,s);
xlabel('ʱ��(s)');ylabel('����');title('ԭʼ����');ylim([-1,1]);
subplot(4,1,2)
plot(t,x);
xlabel('ʱ��(s)');ylabel('����');title('������������');ylim([-1,1]);
subplot(4,1,3)
plot(t,y);
xlabel('ʱ��(s)');ylabel('����');title('Ԥ��Ļ�������');ylim([-1,1]);
subplot(4,1,4)
plot(t,e);
xlabel('ʱ��(s)');ylabel('����');title('�˲��������');ylim([-1,1]);
% pause;
% soundsc(e,fs);

%% ��������ͼ
figure;
spectrogram(s,256,128,256,1000,'yaxis');
title('ԭʼ����');
figure;
spectrogram(x,256,128,256,1000,'yaxis');
title('������������');
figure;
spectrogram(e,256,128,256,1000,'yaxis');
title('�˲��������');
