function [model,matches] = myransacunmatched(callbacks,model,data1,data2,modelPoints,minNumInliers,threshold,maxIters)
%MYRANSACUNMATCHED - RANdom SAmple Consensus algorithm of unmatched data
%
%   model = myransacunmatched(callbacks,model,data1,data2,modelPoints,minNumInliers,threshold)
%   model = myransacunmatched(callbacks,model,data1,data2,modelPoints,minNumInliers,threshold,maxIters)
%   [model,matches] = myransacunmatched(_)


%% �������
narginchk(7,8);
nargoutchk(1,2);

%% ȱʡ��������  
if nargin < 8
    maxIters = 1000;
end

%% ��ȡ�ص�����
calcModel = callbacks.calcModel;
findInliers = callbacks.findInliers;

%% RANSAC�㷨
matches = [];
data1Len = size(data1,1);
data2Len = size(data2,1);
minDataLen = min(data1Len,data2Len);
if minDataLen < minNumInliers
    return;
end
maxNumInliers = minNumInliers;
bestModel = [];
bestMatches = [];
for iter=1:maxIters
    %% ���ѡ��modelPoints��������Ϊ�ڵ�
    idx1 = randperm(data1Len,modelPoints);
    idx2 = randperm(data2Len,modelPoints);

    %% ����ģ��
    model = calcModel(model,data1(idx1,:),data2(idx2,:));

    %% ���Ʒ���ģ�͵��ڵ�
    matches = findInliers(model,data1,data2,threshold);
    numInliers = size(matches,1);

    %% ����
    if numInliers >= maxNumInliers
        maxNumInliers = numInliers;
        bestModel = model;
        bestMatches = matches;
    end
end
if ~isempty(bestModel)
    model = calcModel(bestModel,data1(bestMatches(:,1)),data2(bestMatches(:,2)));
    matches = bestMatches;
end
