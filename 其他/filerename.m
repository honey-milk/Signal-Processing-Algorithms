%filerename.m
clear,clc;
%%
src = uigetdir(pwd,'ԴĿ¼');
folder_list = dir(src);
folder_list(1:2) = []; %ǰ2��Ϊ��Ч�ļ�
folder_num = length(folder_list);
dst = 'E:\������\�����źŴ���\exercise\exercise_9(hmmtb)\sample\ָ��';
%%
for i = 1:folder_num
    file_list = dir(strcat(src,'\',folder_list(i).name,'\*.wav'));
    file_num = length(file_list);
    for j=1:file_num
        img_src_name = [src,'\',folder_list(i).name,'\',file_list(j).name];
        img_dst_name = [dst,'\',folder_list(i).name,'\',file_list(j).name];
        copyfile(img_src_name,img_dst_name);        
    end
    fprintf('�Ѵ����ļ���������%d\n',i);
end