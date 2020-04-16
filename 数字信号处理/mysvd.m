function varargout = mysvd(X,dim)
%MYSVD - Data Dimension Reduction by SVD.
%
%   [U,S,V] = mysvd(X)
%   Y = mysvd(X,dim)
%   [Y,U,S,V] = mysvd(X,dim)
%   [Y,U,S,V] = mysvd(X,[])

% X���������ݣ�m*n���󣬼�m��nά����
% Y����ά������ݣ�m*dim����(dim<n)
% U����������������m*m����
% S������ֵ����m*n����
% V����������������n*n����
% ����ֵ�ֽ⣺X = U*S*V'

%% ��������Ŀ
narginchk(1,2);
nargoutchk(1,4);

%% ��ȡ����
[~,n] = size(X); % ����ά��

%% ����ֵ�ֽ�
[U,S,V] = svd(X); % ��X��������ֵ�ֽ⣬����ֵ�Ѿ�����������
lamda = diag(S);

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
    varargout = {U,S,V};
else
    if nargout == 1
        varargout = {Y};
    else
        varargout = {Y,U,S,V};
    end
end
