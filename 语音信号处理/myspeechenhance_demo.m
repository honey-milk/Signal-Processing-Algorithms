%% myspeechenhance_demo.m
%% ������ǿʾ��
%%
clc,clear;
close all;

%% 
addpath('../�����źŴ���');

%% ��ȡ�����ź�
[s,~] = audioread('../src/sound/speech_clean.wav'); % ԭʼ�����ź�
[x,fs] = audioread('../src/sound/noisy.wav');       % ���������������ź�
s = s(:,1);
x = x(:,1);
nx = length(x);
t = (1:nx) / fs;

%% Ԥ����
s = s - mean(s);      % ȥֱ��
s = s / max(abs(s));  % ��һ��
x = x - mean(x);      % ȥֱ��
x = x / max(abs(x));  % ��һ��

%% ���������ź�
% pause;
% soundsc(s,fs);
% pause;
% soundsc(x,fs);

%% ������ǿ
y1 = myspecsub(x,[4,0.001]);            % �׼���
y2 = myspeechwiener(x,[4,0.001,1,1]);   % ά���˲�
y3 = myspeechwavelet(x,4);              % С���˲�
y4 = myspeechkalman(x,[],2);            % �������˲�

%% ���������
snr = mysnrcalc(s,x);
snr1 = mysnrcalc(s,y1);
snr2 = mysnrcalc(s,y2);
snr3 = mysnrcalc(s,y3);
snr4 = mysnrcalc(s,y4);
fprintf('snr=%f,snr1=%f,snr2=%f,snr3=%f,snr4=%f\n',snr,snr1,snr2,snr3,snr4);

%% ����ʱ����
figure;
subplot(3,2,1);
plot(t,s);
xlabel('ʱ��(s)');ylabel('����');title('ԭʼ����');ylim([-1,1]);
subplot(3,2,2);
plot(t,x);
xlabel('ʱ��(s)');ylabel('����');title('��������');ylim([-1,1]);
subplot(3,2,3);
plot(t,y1);
xlabel('ʱ��(s)');ylabel('����');title('�׼����˲�����');ylim([-1,1]);
subplot(3,2,4);
plot(t,y2);
xlabel('ʱ��(s)');ylabel('����');title('ά���˲�����');ylim([-1,1]);
subplot(3,2,5);
plot(t,y3);
xlabel('ʱ��(s)');ylabel('����');title('С���˲�����');ylim([-1,1]);
subplot(3,2,6);
plot(t,y4);
xlabel('ʱ��(s)');ylabel('����');title('�������˲�����');ylim([-1,1]);

%% ��������ͼ
figure;
subplot(3,2,1);
spectrogram(s,256,128,256,1000,'yaxis');
title('ԭʼ����');
subplot(3,2,2);
spectrogram(x,256,128,256,1000,'yaxis');
title('��������');
subplot(3,2,3);
spectrogram(y1,256,128,256,1000,'yaxis');
title('�׼����˲�����');
subplot(3,2,4);
spectrogram(y2,256,128,256,1000,'yaxis');
title('ά���˲�����');
subplot(3,2,5);
spectrogram(y3,256,128,256,1000,'yaxis');
title('С���˲�����');
subplot(3,2,6);
spectrogram(y4,256,128,256,1000,'yaxis');
title('�������˲�����');

%% ����
% pause;
% soundsc(y1,fs);
% pause;
% soundsc(y2,fs);
% pause;
% soundsc(y3,fs);
% pause;
% soundsc(y4,fs);
