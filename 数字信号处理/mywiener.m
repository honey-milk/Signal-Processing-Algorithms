function [y,e,w] = mywiener(x,d,n)
%MYWIENER - 1-D wiener filtering
%
%   This MATLAB function applies an order-n one-dimensional wiener filter to the
%   input vector, x.
%   
%   y = mywiener(x,d,n)
%   [y,e] = mywiener(x,d,n)
%   [y,e,w] = mywiener(x,d,n)

%% �������
if isvector(x)
    if n>length(x)
        error('ά���˲�������n����С�ڵ����ź�x�ĳ���');
    end
else
    error('�������x������һά����');
end

%% ��������ء�����ص��㷨
rx = ifftshift(xcorr(x));   %���������ź�x������غ���
rx = rx(1:n+1);             %ȡǰn+1�������ϵ��
Rxx = toeplitz(rx);         %���n+1������ؾ���
Rdx = ifftshift(xcorr(d,x));%�õ��ο��ź�d�������ź�x�Ļ���غ���
Rdx = Rdx(1:n+1);           %ȡǰn+1�������ϵ��
w = Rxx\Rdx';               %����ά���˲���ϵ��
y = filter(w,1,x); 
e = d-y; 

%% ����������㷨
% % �����ź���ʱ
% nx = length(x);
% x_in = zeros(n+1,nx); %�����ź���ʱ��ľ���
% x_in(1,:) = x;
% if n>0
%     for i=2:n+1
%         x_in(i,:) = filter([0,1],1,x_in(i-1,:)); %��ʱһ����λ
%     end
% end
% R_xx = x_in*x_in';     %����n������ؾ���
% R_xx = toeplitz(R_xx(1,:));
% R_dx = d*x_in';        %����n�׻��������
% w = R_xx\(R_dx');      %����ά���˲���ϵ��
% y = w'*x_in;           %����ά���˲�
% e = d-y;               %���
