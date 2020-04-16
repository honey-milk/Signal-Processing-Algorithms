function [J,Sk] = myhisteq(I)
%MYHISTEQ - Histogram plot
%
%   J = myhisteq(I)
%   [J,Sk] = myhisteq(I)

%%
%��ȡͼ���С
[row,col,c] = size(I);
J = uint8(zeros(row,col)); %��ʼ��J
%��ͼ���ǲ�ɫͼ����תΪ�Ҷ�ͼ
if c == 3
    I = rgb2gray(I);
end

%�����һ��ֱ��ͼ
Pk = myimhist(I) / numel(I);

%����Ҷ�ӳ���ϵ
Sk = cumsum(Pk);
Sk = round(Sk * 255);

%%
%����ӳ���ϵ�����лҶ�ת��
for i=1:row
    for j=1:col
        J(i,j) = Sk(I(i,j)+1);
    end
end
