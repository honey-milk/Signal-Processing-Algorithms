function varargout = mypca(X,dim)
%MYPCA - Principal Component Analysis.
%
%   [V,D] = mypca(X)
%   Y = mypca(X,dim)
%   [Y,V,D] = mypca(X,dim)
%   [Y,V,D] = mypca(X,[])

% X���������ݣ�m*n���󣬼�m��nά����
% Y����ά������ݣ�m*dim����(dim<n)
% V��������������n*n����
% D������ֵ����n*n����
% Э�������A = Cov(X)
% Э������������ֵ�ֽ⣺A = V*D*V'

%% ��������Ŀ
narginchk(1,2);
nargoutchk(1,3);

%% ��ȡ����
[~,n] = size(X); % ����ά��

%% Э������������ֵ�ֽ�
A = cov(X); % ����X��Э�������A
[V,D] = eig(A); % ��Э�������A��������ֵ�ֽ�
lamda = diag(D); % ������ֵ��~�Ǿ���תΪ������
[lamda,idx] = sort(lamda,'descend'); % ������ֵ���н�������
V = V(:,idx); % ������ֵ��С˳��ı�����������˳��
D = diag(lamda);

%% ��ά���
if nargin == 2
    if isempty(dim) % dim = [] ��ѡȡռ����ֵ����90%������ֵ
        en = sum(lamda.^2);
        thresh = en*0.9;
        acc_en = cumsum(lamda.^2);
        pos = find(acc_en > thresh);
        dim = pos(1);
    elseif dim <=0 || dim >= n
        error('����dim��ΧΪ1~n-1');
    end
    Y = X*V(:,1:dim);
end
    
%% ���
if nargin < 2
    varargout = {V,D};
else
    if nargout == 1
        varargout = {Y};
    else
        varargout = {Y,V,D};
    end
end







