%clear all data etc.
clear all; clc; close all;
%import data file, including headers
breast = readtable('train.csv', 'PreserveVariableNames',true); %import data
rng('default'); % For reproducibility

X = breast(:,[1 3 8]); %select features
Y = breast.Classification; %select targets

cvpt1 = cvpartition(Y, 'KFold', 10); %creating 10-fold partitioning for cross-validation of trained model
r=1
for v = 1:9
    for l = 1:20
        t = templateTree('Reproducible',true, 'MaxNumSplits', l, 'NumVariablesToSample', v); %for reproducibility
        mdl = fitcensemble(X, Y, 'Method', 'bag', 'Learners', t); %fitting a Random Forest classifier to the training data

        error = resubLoss(mdl); %calculating the loss for this model

        [~,score_rf] = resubPredict(mdl); %calculate posterior probabilities

        cvmdl = crossval(mdl, 'CVPartition', cvpt1); %cross-validating the model

        kfloss = kfoldLoss(cvmdl); %calculating the cross-validation loss

        kflc = kfoldLoss(cvmdl,'Mode','cumulative');
        figure(3);
        plot(kflc);
        ylabel('10-fold Misclassification rate');
        xlabel('Learning cycle');
        [predclass, score] = kfoldPredict(cvmdl); %applying the cross validated model to the data
        figure(2)
        confusionchart(Y, predclass) %displaying a confusion chart of the results

        e = kfoldEdge(cvmdl); %calculate the classification edge


        results(r, :) = table(v, l, error, kfloss, e, [sum((predclass == 0) & (Y == 0))], ...
            [sum((predclass == 1) & (Y == 0))], [sum((predclass == 0) & (Y == 1))], ...
            [sum((predclass == 1) & (Y == 1))], ...
            [100 * sum((predclass == 1) & (Y == 1))  / sum(Y == 1)], ...
            [100 * sum((predclass == 0) & (Y==0))/ sum(Y == 0)]); %tabulating results
        r= r+1
    end
end
results.Properties.VariableNames = {'NumVariables', 'MaxNumSplits', 'training error', 'crossval error', ...
    'class edge', 'TN', 'FP', 'FN', 'TP', ...
    'Sensitivity (%)', 'Specificity (%)'} %adding headers to table