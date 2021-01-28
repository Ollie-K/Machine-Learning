%clear all data etc.
clear all; clc; close all;
%import data file, including headers
breast = readtable('train.csv', 'PreserveVariableNames',true); %import data
rng('default'); % For reproducibility

cont = breast(breast.Classification == 0, :); %partition control group
canc = breast(breast.Classification == 1, :); %partition cancer group

Z0 = zscore(cont{:,1:9}); %calculate z scores for control group
[zr, zc] = find(abs(Z0)>4); %index those with z scores above 4
contOutlier = cont(zr, :) %select the relevant rows
cont1 = setdiff(cont(:,:), contOutlier); %remove those rows

Z1 = zscore(canc{:,1:9}); %calculate z scores for control group
[zr, zc] = find(abs(Z1)>4); %index those with z scores above 4
cancOutlier = canc(zr, :) %select the relevant rows
canc1 = setdiff(canc(:,:), cancOutlier); %remove those rows

breast1 = [cont1; canc1]; %rejoin the two groups into one table

X = breast1(:,1:9); %select features
Y = breast1.Classification; %select targets

cvpt1 = cvpartition(Y, 'KFold', 10); %creating 10-fold partitioning for cross-validation of trained model

mdl = fitcnb(X, Y); %fitting a Naive Bayes classifier to the training data

error = resubLoss(mdl); %calculating the training loss

[~,score_nb] = resubPredict(mdl); %calculate posterior probabilities


cvmdl = crossval(mdl, 'CVPartition', cvpt1); %cross-validating the model

kfloss = kfoldLoss(cvmdl); %calculating the cross-validation loss

[predclass, score] = kfoldPredict(cvmdl); %applying the cross validated model to the data
figure(2)
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