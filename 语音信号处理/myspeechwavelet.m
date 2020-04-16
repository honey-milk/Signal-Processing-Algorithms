function y = myspeechwavelet(x,n)
%MYSPEECHWAVELET - Speech enhancement by wavelet
%
%   y = myspeechwavelet(x,n)

%% ��������Ŀ
narginchk(2,2);
nargoutchk(0,1);

%% С������
[C,L] = wavedec(x,n,'db4');
new_C = [];
start = 1;
for i=1:n+1
    w = C(start:start+L(i)-1);
    start = start+L(i);
    thresh = 2*median(abs(w));
    index = abs(w)<thresh;
    w = sign(w).*(abs(w)-thresh);
    w(index) = 0;
    new_C = cat(1,new_C,w);
end
y = waverec(new_C,L,'db4');

%% ��һ��
y = y./max(abs(y));
