function labelMap = myclustering(callback,data,threshold)
%MYCLUSTERING - Clustering algorithm 
%
%   labelMap = myclustering(callback,data,threshold)


%% �������
narginchk(3,3);
nargoutchk(1,1);

%% �������
distMap = callback(data);
logicMap = distMap <= threshold;

%% ����
nData = size(data,1);
labelMap = zeros(nData,1);
label = 0;
orderMap = ones(nData,1);
for i=1:nData
    [~,idx] = max(orderMap);
    orderMap(idx) = 0;
    indexes = logicMap(:,idx) & (orderMap > 0);  
    orderMap(indexes) = i + 1;
    if labelMap(idx) == 0
        label = label + 1;   
        labelMap(idx) = label;
    end
    labelMap(indexes) = labelMap(idx);
end
       
