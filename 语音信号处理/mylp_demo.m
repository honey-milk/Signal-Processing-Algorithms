%% mylp_demo.m
%% ��������Ԥ��ʾ��
%%
clc,clear;
close all;

%% ����
srcFilename = '../src/sound/six.wav';
dstFilename = '../src/sound/six.txt';
fs = 8000;
nwin = 160;
p = 10;

%% ��ȡ�����ź�
[x,Fs] = audioread(srcFilename); % ԭʼ�����ź�
x = x(:,1);
if Fs ~= fs
    x = resample(x,fs,Fs);
end

%% ����Ԥ�����
lpcparam = mylpanalysis(x,fs,nwin,p);
mylpaudiowrite(dstFilename,lpcparam);

%% ����Ԥ��ϳ�
y = mylpsynthesis(lpcparam);

%% ����ʱ����
figure;
subplot(2,1,1);
t = (0:length(x)-1) / fs;
plot(t,x);xlabel('ʱ��(s)');ylabel('����');title('ԭʼ����');ylim([-1,1]);
subplot(2,1,2);
t = (0:length(y)-1) / fs;
plot(t,y);xlabel('ʱ��(s)');ylabel('����');title('�ϳ�����');ylim([-1,1]);

%% ����
pause;
soundsc(x,fs);
pause;
soundsc(y,fs);
