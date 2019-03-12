function [class,posteriori] = mybayesclassify(model,data)
%MYBAYECLASSIFY - Classify by Bayes classifier
%
%   class = mybayesclassify(model,test)
%   [class,posteriori] = mybayesclassify(model,test)

%% �������
narginchk(2,2);
nargoutchk(0,2);

%% ��ȡ����
table = model.table; %�����ṹ���������
nclass = numel(table.classset); %�������
nfeature = numel(table.featureset); %��������
probability = model.probability; %������ʺ���������

%% ����������
%�������
posteriori = zeros(nclass,1);
for i=1:nclass
    posteriori(i) = probability{i,1}; %�������
    %��������
    for j=1:nfeature
        value = data.feature{j};
        valueset = table.featureset{j};
        if ischar(value) %�ַ���
            index = find(strcmp(value,valueset));
        else %����
            index = find(value == cell2mat(valueset));
        end
        posteriori(i) = posteriori(i)*probability{i,j+1}(index); %��������
    end
end
posteriori = posteriori/sum(posteriori);
%ȡ�����������
[~,index] = max(posteriori);
class = table.classset{index};


