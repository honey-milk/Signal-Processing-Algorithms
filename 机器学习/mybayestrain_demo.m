% mybayestrain_demo.m
% ��Ҷ˹������ʾ��
%%
clc,clear;
close all;

%% ����
% traindata(1).class = 'cold';
% traindata(1).feature = {'sneeze','nurse'};
% traindata(2).class = 'allergy';
% traindata(2).feature = {'sneeze','farmer'};
% traindata(3).class = 'concussion';
% traindata(3).feature = {'headache','worker'};
% traindata(4).class = 'cold';
% traindata(4).feature = {'headache','worker'};
% traindata(5).class = 'cold';
% traindata(5).feature = {'sneeze','teacher'};
% traindata(6).class = 'concussion';
% traindata(6).feature = {'headache','teacher'};
traindata(1).class = 'cold';
traindata(1).feature = {1,1};
traindata(2).class = 'allergy';
traindata(2).feature = {1,2};
traindata(3).class = 'concussion';
traindata(3).feature = {2,3};
traindata(4).class = 'cold';
traindata(4).feature = {2,3};
traindata(5).class = 'cold';
traindata(5).feature = {1,4};
traindata(6).class = 'concussion';
traindata(6).feature = {2,4};

%% ѵ��
model = mybayesinit(traindata);
model = mybayestrain(model,traindata);

%% ����
test.feature = {1,3};
[class,posteriori] = mybayesclassify(model,test);
