function dist = mydist(test,data)
%MYDIST - Compute the distance
%   
%   dist = mydist(test,data)

%% ����ŷʽ����
dist = vecnorm(data-test,2,2);
