%% mysvd_demo.m
%% SVD�ֽ⽵ά
%%
clc,clear;
close all;

%% ������������
N = 100;
x = rand(N,1)*10; % x��0~10
y = x+10; % y=k*x+b
y = y+rand(N,1); % ������
X = [x,y]; % ����2ά�ռ�ĵ�

%% ��ʾ��������
figure;
plot(X(:,1),X(:,2),'or');
axis('equal');
title('ԭʼ����');
hold on;

%% SVD��ά
means = mean(X); 
X1 = X-means; % ȥ��ֵ
[Y,U,S,V] = mysvd(X1,[]);
dim = size(Y,2);

%% ������������
nv = size(V,2); % V������Ϊ���������ĸ���
for i=1:nv
    v = V(:,i)*10; % �Ŵ����������ĳ��ȣ�������ʾ
    quiver(0,0,v(1),v(2),'b','linewidth',2);
end
hold off;

%% ���ƾ�����������ת������
Z = X1*V;
figure;
plot(Z(:,1),Z(:,2),'or',Y,zeros(N,1),'ob');
axis('equal');
legend('��ת�������','��ά�������');

%% ���ƽ�ά���ݻ�ԭ������
Z = Y*V(:,1:dim)'+means;
figure;
plot(X(:,1),X(:,2),'or',Z(:,1),Z(:,2),'ob');
axis('equal');
legend('ԭʼ����','��ԭ������');

