function clusters = myclustering(map)
%MYCLUSTERING - Clustering algorithm 
%
%   clusters = myclustering(map)

%% �������
narginchk(1,1);
nargoutchk(1,1);

%% ����
nData = size(map,1);
labels = zeros(nData,1);
label = 0;
orderList = ones(nData,1);
for i=1:nData
    [~,idx] = max(orderList);
    orderList(idx) = 0;
    indexes = map(:,idx) & (orderList > 0);  
    orderList(indexes) = i + 1;
    if labels(idx) == 0
        label = label + 1;   
        labels(idx) = label;
    end
    labels(indexes) = labels(idx);
end       

%% ����Ϊclusters
clusters = [];
nClusters = max(labels);
numInliers = zeros(nClusters,1);
for i=1:nClusters
    indexes = find(labels == i);
    clusters = cat(1,clusters,{indexes});
    numInliers(i) = length(indexes);
end

%% ����
[~,indexes] = sort(numInliers,'descend');
clusters = clusters(indexes);
