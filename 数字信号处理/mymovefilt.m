function y = mymovefilt(x,n)
%MYMOVEFILT - Moving average filter
%
%   y = mymovefilt(x,n)

%% ��������Ŀ
narginchk(2,2);
nargoutchk(0,1);

%% ��������
nx = length(x);
if n>nx
    error('����n����С�ڵ���x�ĳ���');
end
if rem(n,2)==1
    p = (n-1)/2;
else
    error('����n����Ϊ����');
end

%% ����ƽ���˲�
y = x;
for i=p+1:nx-p
    y(i) = sum(x(i-p:i+p))/n;
end

