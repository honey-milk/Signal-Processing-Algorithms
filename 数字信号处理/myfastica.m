function S = myfastica(X,maxiter)
%MYFASTICA - Fast Independent Component Analysis.
%
%   S = myfastica(X)

% X���������ݣ�m*n���󣬼�m������
% S���ֽ�����Ķ����ɷ֣���m*n���󣬼�m�������ɷ�

%% ��������Ŀ
narginchk(1,2);
nargoutchk(1,1);

%% ��ȡ����
if nargin < 2
    maxiter = 100;
end
[m,~] = size(X);
alpha = 1;
thresh = 0.0001; % �ж���������ֵ

%% ����Ԥ����
% ���Ļ�
means = mean(X,2);
X = X-means; 
% �׻�
[U,D] = eig(X*X'); % ��X��Э��������������ֵ�ֽ�
V = sqrt(D)\U'; % �׻�����
X = V*X;

%% �����Ż�W
% �����ʼ��W
W = rand(m);
% ����
converge = false;
for i=1:maxiter
    % �������ֵ
    S = W*X; 
    % �������ֵ��tanh����ֵ���䵼��ֵ
    GX = tanh(alpha*S);
    G_X = alpha*(1-GX.^2); % tanh�ĵ���
    % ����W
    EG_X = mean(G_X,2);
    W1 = GX*X'-EG_X.*W;
    % ������һ����ȥ��أ��Ͱ׻�����һ����
    [U,D] = eig(W1*W1'); 
    W1 = sqrt(D)\U'*W1;
    % �ж��Ƿ�����
    A = W1*W';
    W = W1;
    delta = max(abs(abs(diag(A))-1));
    if delta < thresh
        converge = true;
        break;
    end
end
if ~converge
    disp('δ�������������������!!!');
end

%% ��������ɷ�
S = W*X;
