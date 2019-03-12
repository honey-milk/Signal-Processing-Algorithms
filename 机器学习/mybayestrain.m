function model = mybayestrain(model,traindata)
%MYBAYESTRAIN - Train bayes classifier
%
%   model = mybayestrain(traindata)

%% �������
narginchk(2,2);
nargoutchk(0,1);

%% ��ʼ��ģ��
if isempty(model)
    return;
end

%% ��ȡ����
table = model.table; %�����ṹ���������
nclass = numel(table.classset); %�������
nfeature = numel(table.featureset); %��������

%% ��ʼ������
probability = cell(nclass,nfeature+1); %������ʺ���������
for i=1:nclass
    %�������
    probability{i,1} = 0; 
    %��������
    for j=1:nfeature
        nvalue = numel(table.featureset{j}); %��j��������ȡֵ����
        probability{i,j+1} = zeros(1,nvalue);
    end
end

%% ͳ�ƴ���
%����ѵ������
nsample = numel(traindata); %�������� 
for i=1:nsample
    data = traindata(i);
    class = data.class;
    classset = table.classset;
    indexclass = find(strcmp(class,classset));
    %��������1
    probability{indexclass,1} = probability{indexclass,1}+1; 
    %��������
    for j=1:nfeature
        value = data.feature{j};
        valueset = table.featureset{j};
        if ischar(value) %�ַ���
            index = find(strcmp(value,valueset));
        else %����
            index = find(value == cell2mat(valueset));
        end
        probability{indexclass,j+1}(index) = probability{indexclass,j+1}(index)+1; %��j������ȡ��index��ֵ�Ĵ�����1
    end
end

%% �������
for i=1:nclass
    num = probability{i,1}; %��i����ִ���
    %�������
    probability{i,1} = probability{i,1}/nsample; %��nsample��һ��
    %��������
    for j=2:nfeature+1
        probability{i,j} = probability{i,j}/num; %��num��һ��
    end
end

%% �������
model.probability = probability;
