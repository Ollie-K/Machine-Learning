%clear all data etc.
clear all; clc; close all;
%import data file, including headers
breast = readtable('train.csv', 'PreserveVariableNames',true); %import data
rng('default'); % For reproducibility

feats = [1 3 8];

X = breast(:, feats); %select features
Y = breast.Classification; %select targets

cvpt1 = cvpartition(Y, 'KFold', 10); %creating 10-fold partitioning for cross-validation of trained model
cvpt2 = cvpartition(Y, 'KFold', 5); %creating 5-fold partitioning for cross-validation of trained model

mdl = fitcnb(X, Y, 'Weights', breast.weights, ...
    'DistributionNames', 'kernel', 'Kernel', 'normal', 'Width', 6.7532, ...
    'Cost', struct('ClassNames',{{'0','1'}}, 'ClassificationCosts',[0 0.7; 1 0])); %fitting

error = resubLoss(mdl); %calculating the training loss

cvmdl = crossval(mdl, 'CVPartition', cvpt1); %10-foldcross-validating the model
cvmdl1 = crossval(mdl, 'CVPartition', cvpt2); %cross-validating the model

kfloss = kfoldLoss(cvmdl); %calculating the cross-validation loss
kflc = kfoldLoss(cvmdl,'Mode','individual');
figure(4);
plot(kflc);
ylabel('10-fold Misclassification rate');
xlabel('Learning cycle');

[predclass, score] = kfoldPredict(cvmdl); %applying the 10-fold cross validated model to the data
figure(5)
confusionchart(Y, predclass) %displaying a confusion chart of the results
[X10,Y10,T10,AUC10] = perfcurve(Y, score(:,2), 1); %calculating ROC
figure
plot(X10,Y10); %plotting ROC

e = kfoldEdge(cvmdl); %calculate the classification edge

[predclass5, score5] = kfoldPredict(cvmdl1); %applying the 5-fold cross validated model to the data
figure
confusionchart(Y, predclass5) %displaying a confusion chart of the results
[X5,Y5,T5,AUC5] = perfcurve(Y, score5(:,2), 1); %calculate ROC
figure
plot(X5,Y5); %plotting ROC


results = table(error, kfloss, e, AUC10, [sum((predclass == 0) & (Y == 0))], ...
    [sum((predclass == 1) & (Y == 0))], [sum((predclass == 0) & (Y == 1))], ...
    [sum((predclass == 1) & (Y == 1))], ...
    [100 * sum(sum((predclass == 0) & (Y == 0)) + sum((predclass == 1) & (Y == 1))) / sum((Y==1) | (Y==0))],...
    [100 * sum((predclass == 1) & (Y == 1))  / sum(Y == 1)], ...
    [100 * sum((predclass == 0) & (Y==0))/ sum(Y == 0)]); %tabulating 10-fold results
results.Properties.VariableNames = {'training error', 'crossval error', ...
    'class edge', 'ROC AUC (10-Fold)', 'TN', 'FP', 'FN', 'TP', ...
    'Accuracy (%)', 'Sensitivity (%)', 'Specificity (%)'} %adding headers to table

%save('nb_final.mat', 'mdl')