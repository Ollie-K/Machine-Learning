%clear all data etc.
clear all; clc; close all;
%import data file, including headers
breast = readtable('train.csv', 'PreserveVariableNames',true); %import data
rng(1); % For reproducibility

X = breast(:,[1 3 8]); %select features
Y = breast.Classification; %select targets

cvpt1 = cvpartition(Y, 'KFold', 10); %creating 10-fold partitioning for cross-validation of trained model
opt = struct('CVPartition', cvpt1, 'MaxObjectiveEvaluations', 200)
t = templateTree('Reproducible',true); %for reproducibility
mdl = fitcensemble(X,Y,'OptimizeHyperparameters',{'MinLeafSize', 'MaxNumSplits', 'SplitCriterion', 'NumVariablesToSample'},...
    'Learners', t, 'Method', 'bag', 'NumLearningCycles', 20, 'HyperparameterOptimizationOptions', opt)

error = resubLoss(mdl); %calculating the training loss




cvmdl = crossval(mdl, 'CVPartition', cvpt1); %cross-validating the model

kfloss = kfoldLoss(cvmdl); %calculating the cross-validation loss


[predclass, score] = kfoldPredict(cvmdl); %applying the cross validated model to the data
figure(4)
confusionchart(Y, predclass) %displaying a confusion chart of the results

e = kfoldEdge(cvmdl); %calculate the classification edge


results = table(error, kfloss, e, [sum((predclass == 0) & (Y == 0))], ...
    [sum((predclass == 1) & (Y == 0))], [sum((predclass == 0) & (Y == 1))], ...
    [sum((predclass == 1) & (Y == 1))], ...
    [100 * sum((predclass == 1) & (Y == 1))  / sum(Y == 1)], ...
    [100 * sum((predclass == 0) & (Y==0))/ sum(Y == 0)]); %tabulating results
results.Properties.VariableNames = {'training error', 'crossval error', ...
    'class edge', 'TN', 'FP', 'FN', 'TP', ...
    'Sensitivity (%)', 'Specificity (%)'} %adding headers to table

%save('rf_opt.mat', 'mdl')