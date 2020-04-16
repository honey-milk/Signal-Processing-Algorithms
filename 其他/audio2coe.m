% audio2coe.m
% ��Ƶת.coe�ļ�
%%
clear,clc;
close all;

%% ��ȡ��Ƶ�ź�
soundFilename = '../src/sound/beyond - �������.wav';
coeFilename = '../src/sound/rom.coe';
[y,fs] = audioread(soundFilename); 
totalTime = 16;
y = y(1:(fs*totalTime-1),1); % ȡһ������
y = (y + 1) / 2; % תΪ��ֵ
sound(y,fs);
figure;plot(y);

%% ����Ϊcoe�ļ�
y = round(y * 225);
fid = fopen(coeFilename,'wt');
fprintf(fid,'memory_initialization_radix = 16;\nmemory_initialization_vector = \n');
ny = length(y);
fprintf(fid,'%x,\n',y);
fclose(fid); % �ر��ļ�
