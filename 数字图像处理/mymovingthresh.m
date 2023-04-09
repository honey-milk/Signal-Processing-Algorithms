function g = mymovingthresh(f,n,K)
%MYMOVINGTHRESH - Moving average threshold
%
%   g = mymovingthresh(f,n,K)

[M,N] = size(f);
f = double(f);

%% Zigzagɨ��
f(2:2:end,:) = fliplr(f(2:2:end,:));
f = f';
f = f(:);

%% �����ƶ���ֵ
b = ones(1,n)/n;
ma = filter(b,1,f);

%% ��ֵ����
g = f > K*ma;

%% ��Zigzagɨ��
g = reshape(g,N,M)';
g(2:2:end,:) = fliplr(g(2:2:end,:));
1 2 3
4 5 6
7 8 9

1 6 7
2 5 8
3 4 9

1 2 4
7 5 3
6 8 9

1 7 6
2 5 8
4 3 9
