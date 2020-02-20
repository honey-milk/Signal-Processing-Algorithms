function [models,masks] = myransacmatched(cb,data,modelPoints,nModels,minNumInlier,threshold,confidence,maxIters)
%MYRANSACMATCHED - RANdom SAmple Consensus algorithm of matched data
%
%   models = myransacmatched(cb,data,modelPoints,nModels,minNumInlier,threshold)
%   models = myransacmatched(cb,data,modelPoints,nModels,minNumInlier,threshold,confidence)
%   models = myransacmatched(cb,data,modelPoints,nModels,minNumInlier,threshold,confidence,maxIters)
%   [models,masks] = myransacmatched(_)


%% �������
narginchk(6,8);
nargoutchk(1,2);

%% ȱʡ��������
if nargin < 7
    confidence = 0.99;   
    maxIters = 1000;    
elseif nargin < 8
    maxIters = 1000;
end

%% ��ȡ�ص�����
calcModel = cb.calcModel;
findInliers = cb.findInliers;

%% RANSAC�㷨
models = [];
masks = logical([]);
dataLen = size(data,1);
totalMask = false(1,dataLen);
for i=1:nModels
    maxNumInlier = minNumInlier;
    niters = maxIters;
    leftIndex = find(totalMask==0);
    leftDataLen = length(leftIndex);
    if leftDataLen < modelPoints
        break;
    elseif leftDataLen == modelPoints
        bestModel = calcModel(data(leftIndex,:));
        bestMask = totalMask==0;
        totalMask(bestMask) = true;
    else
        bestModel = [];
        bestMask = logical([]);
        for iter=1:maxIters
            if iter > niters
                break;
            end
            %% ���ѡ��modelPoints��������Ϊ�ڵ�
            idx = randperm(leftDataLen,modelPoints);
            idx = leftIndex(idx);
            subData = data(idx,:);
            %% ����ģ��
            model = calcModel(subData);
            if isempty(model)
                continue;
            end
            %% ���Ʒ���ģ�͵��ڵ�
            idx = findInliers(model,data(leftIndex,:),threshold);
            mask = false(1,dataLen);
            mask(leftIndex(idx)) = true;
            numInlier = sum(mask);
            %% ����
            if numInlier >= maxNumInlier
                maxNumInlier = numInlier;
                bestModel = model;
                bestMask = mask;
                newniters = ceil(log(1 - confidence) / log(1 - (numInlier / leftDataLen) ^ modelPoints));
                if newniters < niters
                    niters = newniters;
                end
            end
        end
        if ~isempty(bestModel)
            bestModel = calcModel(data(bestMask,:));
            totalMask(bestMask) = true;
        end
    end
    models = cat(1,models,bestModel);
    masks = cat(1,masks,bestMask);
end

    