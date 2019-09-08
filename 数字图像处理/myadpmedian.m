function g = myadpmedian(f,maxw)
%MYADPMEDIAN - Adptive median filter
%
%   g = myadpmedian(f)
%   g = myadpmedian(f,maxw)

%% �������
narginchk(1,2);
nargoutchk(0,1);

%% ȱʡ��������
if nargin<2
    maxw = 7;   %Ĭ���������Χ
end

%% �㷨
[rows,cols] = size(f);
g = f;
flag = false(rows,cols);
%i������,j������
for i=1:rows       
    for j=1:cols    
        w = 1;
        %%%%%%%%ȷ������
        while flag(i,j)==0
            left = j-w;
            right = j+w;
            top = i-w;
            bottom = i+w;
            left = max(left,1);
            right = min(right,cols);
            top = max(top,1);
            bottom = min(bottom,rows);        
            %����ȷ�����
            region = f(top:bottom,left:right);
            region = region(:);     %תΪ������
            smin = min(region);
            smax = max(region);
            smed = median(region);
            sij = f(i,j);
            %ȷ�������Сֵ����
            if (smed>smin) && (smed<smax)
                if (sij>smin) && (sij<smax)
                    g(i,j) = sij; 
                else
                    g(i,j) = smed;
                end
                flag(i,j) = 1; %������ɱ��
            else
                if w <= maxw %��������Χ
                    w = w+1; 
                else
                    g(i,j) = smed;
                    flag(i,j) = 1; %������ɱ��
                end
            end
        end    
    end       
end            
    