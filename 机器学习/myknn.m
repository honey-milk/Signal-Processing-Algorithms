function class = myknn(test,data,label,k)
%MYKNN - K-Nearest Neighbor clustering
%
%   class = myknn(test,data,label,k)

rows = size(data,1); %����ѵ������(data)����
test = repmat(test,rows,1); %����������(test)��չΪrows��
diffMat = test-data; %����������(test)������ѵ������(data)����
distanceMat = sqrt(sum(diffMat.^2,2)); %�����������(test)������ѵ������(data)��ŷʽ����
[~,index] = sort(distanceMat,'ascend'); %��distanceMat����������
len = min(k,rows); %ȡk����������rows�е���Сֵ
class = mode(label(index(1:len))); %ȡǰlen��label������