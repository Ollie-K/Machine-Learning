%clear all data etc.
clear all; clc; close all;
%import data file, including headers
breast = readtable('train.csv', 'PreserveVariableNames',true); %import data
rng('default'); % For reproducibility

X = breast(:,[1 3 8]); %select features
Y = breast.Classification; %select targets
cvpt1 = cvpartition(Y, 'KFold', 10); %creating 10-fold partitioning for cross-validation of trained model
t = templateTree('Reproducible',true, 'MinLeafSize', 2, 'MaxNumSplits', 10); %for reproducibility
mdl = fitcensemble(X,Y, 'Learners',t, 'Method', 'bag', 'NumLearningCycles', 15)

error = resubLoss(mdl)

cvmdl = crossval(mdl, 'CVPartition', cvpt1); %cross-validating the model

kfloss = kfoldLoss(cvmdl); %calculating the cross-validation loss
kflc = kfoldLoss(cvmdl,'Mode','individual');
   
figure(4);
plot(kflc);
ylabel('10-fold Misclassification rate');
xlabel('Learning cycle');

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
