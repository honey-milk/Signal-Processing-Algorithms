function model = mybayesinit(traindata)
%MYBAYESINIT - Initial bayes classifier
%
%   model = mybayesinit(traindata)

%% �������
narginchk(1,1);
nargoutchk(0,1);

%% ��ȡ����
if isempty(traindata)
    model = [];
    return;
end
data = traindata(1);
nfeature = numel(data.feature); %��������

%% ��ʼ��������
table.classset = [];
table.featureset = cell(1,nfeature);

%% ����������

%����ѵ������
nsample = numel(traindata); %��������
for i=1:nsample
    data = traindata(i);
    class = data.class;
    classset = table.classset;
    if ~any(strcmp(class,classset)) %classset�в�����class
        classset = cat(1,classset,{class}); %��class��ӵ�classset
        table.classset = classset;
    end
    %��������
    for j=1:nfeature
        value = data.feature{j};
        valueset = table.featureset{j};
        if ischar(value) %�ַ���
            if ~any(strcmp(value,valueset)) %valueset�в�����value
                valueset = cat(1,valueset,{value}); %��value��ӵ�valueset
                table.featureset{j} = valueset;
            end        
        else %����
            if ~any(value == cell2mat(valueset)) %valueset�в�����value
                valueset = cat(1,valueset,{value}); %��value��ӵ�valueset
                table.featureset{j} = valueset;
            end                
        end
    end
end

%% ����������
model.table = table;
