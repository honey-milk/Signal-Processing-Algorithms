function [J,Sk] = myimprovedhisteq(I)
%MYIMPROVEDHISTEQ - Enhance contrast using improved histogram equalization
%
%   J = myimprovedhisteq(I)

%% ͼ��Ԥ����
%��ȡͼ���С
[row,col,C] = size(I);
%��ͼ���ǲ�ɫͼ����תΪ�Ҷ�ͼ
if C == 3
    I = rgb2gray(I);
end

J = uint8(zeros(row,col));   %������ͼ��
img_grad = zeros(row,col);   %4�������ݶ�

%% �����ݶ�
img_expanded = zeros(row+2,col+2);%���ڼ����ݶȵ�ͼ��ͼ��4�����ȫ����(����)
img_expanded(2:(row+1),2:(col+1)) = I;
p = [0,1,0;1,-1,1;0,1,0];         %�ݶ�����
for i=1:row
    for j=1:col
        block = img_expanded(i:i+2,j:j+2);
        tamp = p .* block;
        grad_y1 = tamp(2,2) + tamp(3,2);    %y�������ݶ�
        grad_y2 = tamp(2,2) + tamp(1,2);    %y�Ḻ���ݶ�
        grad_x1 = tamp(2,2) + tamp(2,3);    %x�������ݶ�
        grad_x2 = tamp(2,2) + tamp(2,1);    %x�Ḻ���ݶ�
        %���ݶ���Ч
        grad_y1 = max(grad_y1,0);
        grad_y2 = max(grad_y2,0);
        grad_x1 = max(grad_x1,0);
        grad_x2 = max(grad_x2,0);
        %�ݶ����
        img_grad(i,j) = grad_y1 + grad_y2 + grad_x1 + grad_x2;
    end
end

%% �����Ȩ��һ��ֱ��ͼ
Pk = zeros(256,1);
N = sum(sum(img_grad));   %��ĸ
for k=1:256
    pix = I == (k - 1);    %Ѱ�һҶ�ֵΪk�����ص�
    Nk = sum(sum(img_grad .* pix));%����
    Pk(k) = Nk / N;
end

%% ����Ҷ�ӳ���ϵ
Sk = cumsum(Pk);
Sk = round(Sk * 255);

%% ����ӳ���ϵ�����лҶ�ת��
for i=1:row
    for j=1:col
        J(i,j) = Sk(I(i,j)+1);
    end
end
