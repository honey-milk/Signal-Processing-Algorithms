function dist = mydist(test,data)
%MYDIST - Compute the distance
%   
%   dist = mydist(test,data)

rows = size(data,1); %����data����
test = repmat(test,rows,1); %��test��չΪrows��
diffmat = test-data; %��test������data����
dist = sqrt(sum(diffmat.^2,2)); %����test������data��ŷʽ����