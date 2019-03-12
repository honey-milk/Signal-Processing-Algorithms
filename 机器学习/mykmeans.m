function [IDX,Y] = mykmeans(X,k,option)
%MYKMEANS - K-means clustering
%
%    This MATLAB function partitions the points in the n-by-p data matrix X into k
%    clusters.
%
%   IDX = mykeams(X,k)
%   IDX = mykeams(X,k,option)
%   [IDX,Y] = mykeams(...)

%% �������
narginchk(2,3);
nargoutchk(1,2);

%% �������ֵ���
% ����X
if ismatrix(X)
    [num,~] = size(X);
    X = double(X);
else
    error('�������X�����Ǿ���');
end
% ����k
if k>num
    error('�������kֵ����X����������');
end
% ����option
if nargin<3
    option = struct('MaxIter',100,'Delta',1e-4,'Display','off');
end
% ��ȡmaxiter
if isfield(option,'MaxIter')
    maxiter = option.MaxIter;
else
    maxiter = 100;
end
% ��ȡdelta
if isfield(option,'Delta')
    delta = option.Delta;
else
    delta = 1e-4;
end
% ��ȡdispaly
if isfield(option,'Display')
    display = option.Display;
else
    display = 'off';
end

%% ��ó�ʼ���ӵ㣨�������ģ�
[~, index] = crossvalind('LeaveMOut', num, k);  %X�е�num���������ѡȡk����Ϊ��ʼ����
Y = X(index,:);

%% ����
IDX = zeros(num,1); % X������Vj��������
d = zeros(1,k); % X������Vj��k���������ĵ�ŷʽ����
D = zeros(1,maxiter); % ��i�ε���ʱ���ܾ���
deltai = zeros(1,maxiter); % ��i�ε���ʱ�ܾ������Ըı���
for i=1:maxiter %����һ�θ���һ�ξ�������
    for j=1:num %����X��ÿ������
        % ����X������Vj��k���������ĵ�ŷʽ����d
        Vj = X(j,:);
        for n=1:k
            Yn = Y(n,:);
            d(n) = sqrt(sum((Vj-Yn).^2)); %d(n) = ((x1-y1)^2+...+(xm-ym)^2)^0.5
        end
         % ��Vj��Ϊ�����������
        [dmin,pos] = min(d);
        D(i) = D(i)+dmin;
        IDX(j) = pos;
    end
    % ����k����������
    for j=1:k
        % ѡ��X�����ڵ�j�������
        Class = X(IDX==j,:); 
        % �����µľ�������
        if ~isempty(Class)
            if size(Class,1)==1
                Y(j,:) = Class;
            else
                Y(j,:) = mean(Class);
            end
        end
    end
    % ����D(i)�ĸı�������С����
    if i==1
        deltai(i) = 1;
    else
        deltai(i) = abs(D(i)-D(i-1))/(D(i-1)+eps);
        if deltai(i)<=delta % �ñ���С����ֵdelta���˳�����
            break;
        end
    end
end

%% ����deltai����
if strcmp(display,'on')
    plot(deltai);
    xlabel('��������');
    ylabel('�ܾ�����Ըı���');
end
