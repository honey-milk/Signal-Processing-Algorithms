%filerename.m
clear,clc;
%%
src = uigetdir(pwd,'ԴĿ¼');
dst = uigetdir(pwd,'Ŀ��Ŀ¼');
file_list = dir(strcat(src,'\*.jpg'));
file_num = length(file_list);
%%
for i = 1:file_num
    img_src_name = [src,'\',file_list(i).name];
    %pos = strfind(file_list(i).name, '.');
    %id = str2double(file_list(i).name(6:pos(1)-1));
    img_dst_name = [dst,'\',num2str(id,'%09d'),'.png'];
    movefile(img_src_name, img_dst_name);
    fprintf('�Ѵ����ļ�������%d\n',i);
end