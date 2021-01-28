%clear all data etc.
clear all; clc; close all;
%import data file, including headers
breast = readtable('train.csv', 'PreserveVariableNames',true); %import data
rng('default'); % For reproducibility

X = breast(:,[1 3 8]); %select features
Y = breast.Classification; %select targets

cvpt1 = cvpartition(Y, 'KFold', 10); %creating 10-fold partitioning for cross-validation of trained model

opt = struct( 'CVpartition', cvpt1, 'AcquisitionFunctionName', 'expected-improvement-plus'); %optimization parameters
mdl = fitcnb(X,Y,'Weights',breast.weights, 'DistributionNames', 'Kernel',...
    "OptimizeHyperparameters", {'Width', 'Kernel'}, ...
    'HyperparameterOptimizationOptions', opt, ...
    'Cost', struct('ClassNames',{{'0','1'}},'ClassificationCosts',[0 0.7; 1 0])); %fitting a weighted Naive Bayes classifier to the training data

error = resubLoss(mdl); %calculating the training loss

[~,score_nb] = resubPredict(mdl); %calculate posterior probabilities

cvmdl = crossval(mdl, 'CVPartition', cvpt1); %cross-validating the model

kfloss = kfoldLoss(cvmdl); %calculating the cross-validation loss
kflc = kfoldLoss(cvmdl,'Mode','individual');
figure(3);
plot(kflc);
ylabel('10-fold Misclassification rate');
xlabel('Learning cycle');

[predclass, score] = kfoldPredict(cvmdl); %applying the cross validated model to the data
figure(4)
confusionchart(Y, predclass) %displaying a confusion chart of the results

e = kfoldEdge(cvmdl); %calculate the classification edge

%plot classification success
figure(5)
C = repmat([0,1],numel(X),1); %assign colours
scatter3(X{:,1},X{:,2},X{:,3}, 100, Y, 'LineWidth', 2) %plot true classes as circles
hold on
scatter3(X{:,1},X{:,2},X{:,3}, 100, predclass, 'Marker',...
    'x', 'LineWidth', 2) %plot predictions as crosses on the same axes
hold off
xlabel('Age (standardized)') %label axes
ylabel('Glucose (standardized)')
zlabel('Resistin (standardized)')
colormap(jet) %assign colours (red = cancer, blue = control)

results = table(error, kfloss, e, [sum((predclass == 0) & (Y == 0))], ...
    [sum((predclass == 1) & (Y == 0))], [sum((predclass == 0) & (Y == 1))], ...
    [sum((predclass == 1) & (Y == 1))], ...
    [100 * sum((predclass == 1) & (Y == 1))  / sum(Y == 1)], ...
    [100 * sum((predclass == 0) & (Y==0))/ sum(Y == 0)]); %tabulating results
results.Properties.VariableNames = {'training error', 'crossval error', ...
    'class edge', 'TN', 'FP', 'FN', 'TP', ...
    'Sensitivity (%)', 'Specificity (%)'} %adding headers to table