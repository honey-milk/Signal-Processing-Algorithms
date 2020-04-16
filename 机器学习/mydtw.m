function dist = mydtw(x,y)
%MYDTW - Dynamic Time Warping
%
%   dist = mydtw(t,r)

%% �������
narginchk(2,2);
nargoutchk(0,1);

%% �������ֵ���
if isvector(x)
    x = x(:);
    y = y(:);
end

n = size(x,1);
m = size(y,1);
% ֡ƥ��������
d = zeros(n,m);
for i = 1:n
    for j = 1:m
        d(i,j) = sum((x(i,:)-y(j,:)).^2);
    end
end
% �ۻ��������
D = ones(n+1,m+1)*realmax;
D(1,1) = 0;
% ��̬�滮
for i = 1:n
    for j = 1:m
        D1 = D(i,j+1); %�·�
        D2 = D(i+1,j); %��
        D3 = D(i,j); %���·�
        D(i+1,j+1) = d(i,j) + min([D1,D2,D3]);
    end
end
dist = D(n+1,m+1);
