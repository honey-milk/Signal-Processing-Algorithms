% filerename.m
% �ļ�����������
%%
clear,clc;
close all;

%% ·��
srcPath = uigetdir(pwd,'ԴĿ¼');
dstPath = uigetdir(pwd,'Ŀ��Ŀ¼');
fileList = dir(strcat(src,'\*.jpg'));
numFiles = length(fileList);

%% ���������ļ�
for i=1:numFiles
    srcFilename = [srcPath,'\',fileList(i).name];
    dstFilename = [dstPath,'\',num2str(i,'%06d'),'.jpg'];
    movefile(srcFilename, dstFilename);
    fprintf('���ȣ�%d/%d\n',i,numFiles);
end