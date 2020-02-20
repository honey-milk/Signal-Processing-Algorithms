function [model,matches] = myransacunmatched(cb,model,data1,data2,modelPoints,minNumInlier,threshold,maxIters)
%MYRANSACUNMATCHED - RANdom SAmple Consensus algorithm of unmatched data
%
%   model = myransacunmatched(cb,model,data1,data2,modelPoints,minNumInlier,threshold)
%   model = myransacunmatched(cb,model,data1,data2,modelPoints,minNumInlier,threshold,maxIters)
%   [model,matches] = myransacunmatched(_)


%% �������
narginchk(7,8);
nargoutchk(1,2);

%% ȱʡ��������  
if nargin < 8
    maxIters = 1000;
end

%% ��ȡ�ص�����
calcModel = cb.calcModel;
findInliers = cb.findInliers;

%% RANSAC�㷨
matches = [];
data1Len = size(data1,1);
data2Len = size(data2,1);
minDataLen = min(data1Len,data2Len);
maxNumInlier = minNumInlier;

if minDataLen < modelPoints
    return;
else
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
        numInlier = size(matches,1);
        
        %% ����
        if numInlier >= maxNumInlier
            maxNumInlier = numInlier;
            bestModel = model;
            bestMatches = matches;
        end
    end
    if ~isempty(bestModel)
        model = calcModel(bestModel,data1(bestMatches(:,1)),data2(bestMatches(:,2)));
        matches = bestMatches;
    end
end
