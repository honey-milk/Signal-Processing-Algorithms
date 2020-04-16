function clusters = mytlinkage(P)
%MYTLINKAGE - T-Linkage clustering algorithm 
%
%   clusters = mytlinkage(P)

%% �������
narginchk(1,1);
nargoutchk(1,1);

%% ����
keepClustering = true;
step = 0;
nClusters = size(P,1);
clusters = num2cell((1:nClusters)');
while keepClustering
    keepClustering = false;
    
    %% Ѱ�Ҿ�����С��2��
    R = P * P';
    distance = R ./ max(diag(R) + diag(R)' - R,eps);
    distance = triu(distance,1);
    maxDistance = max(distance(:));
    if maxDistance > 0
        keepClustering = true;
        step = step + 1;
        [rows,cols] = find(distance == maxDistance);
        idx1 = rows(1);
        idx2 = cols(1);

        %% �ϲ���2��
        clusters{idx1} = cat(1,clusters{idx1},clusters{idx2});
        clusters(idx2) = [];
        set1 = P(idx1,:);
        set2 = P(idx2,:);
        mergedSet = set1 & set2;
        P(idx1,:) = mergedSet;
        P(idx2,:) = [];
    end
end
fprintf("Clustering finished after %d steps\n",step);

%% ����
nClusters = length(clusters);
numInliers = zeros(nClusters,1);
for i=1:nClusters
    numInliers(i) = length(clusters{i});
end
[~,indexes] = sort(numInliers,'descend');
clusters = clusters(indexes);
