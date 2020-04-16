% filter2coe.m
% �˲���ϵ��ת.coe�ļ�
%%
clear,clc;
close all;

%% ����
coeFilename = '../src/sound/filter.coe';

%% ��ȡ�˲���ϵ��
b = fir1(63,0.25,'low');
freqz(b,1);

%% ����Ϊcoe�ļ� 
fid = fopen(coeFilename,'w');
fprintf(fid,'# banks: 1\n');
fprintf(fid,'# coeffs: %d\n',length(b));
fprintf(fid,'%f,\n',b);
fclose(fid); % �ر��ļ�