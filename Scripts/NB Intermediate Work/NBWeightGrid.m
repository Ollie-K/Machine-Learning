%clear all data etc.
clear all; clc; close all;
%import data file, including headers
train = readtable('train.csv', 'PreserveVariableNames',true); %import data
rng('default'); % For reproducibility
smaller = linspace(0.0001,0.4751,20);
small = linspace(0.05,1,20);

 %Generating weights to diminish impact of outliers
    Z1 = zscore(train{:,1:9}); %calculate z scores for model
    [zr4, zc4] = find(abs(Z1)>4); %find outliers
    [zr5, zc5] = find(abs(Z1)>5); %find outliers

for i = 1:20
    for k = 1:20
        if smaller(i) < small(k)
            weights = ones(size(train(:,10),1),1); %generate weighting of 1 for all values
            weights(zr4) = smaller(i); %reduce weighting for outliers
            weights(zr5) = small(k); %reduce weighting for outliers
            weights = array2table(weights); %change to table
            train(:,10) = weights; %join the table of weights to main table of data
            breast = train;
            X = breast(:,1:9); %select features
            Y = breast.Classification; %select targets

            cvpt1 = cvpartition(Y, 'KFold', 10); %creating 10-fold partitioning for cross-validation of trained model

            mdl = fitcnb(X, Y, 'Weights', breast.weights); %fitting a weighted Naive Bayes classifier to the training data



            cvmdl = crossval(mdl, 'CVPartition', cvpt1); %cross-validating the model

            kfloss = kfoldLoss(cvmdl); %calculating the cross-validation loss
        else
            kfloss = NaN;
        end
        
        results(k,i) = kfloss; %tabulating results
    end
end
colnames = string(round(smaller,3));
rownames = string(small);
results = array2table(results);
results.Properties.VariableNames = colnames; %adding headers to table
results.Properties.RowNames = rownames; %adding headers to table

imagesc(results{:,:}); %plot z scores by colour
colorbar; %include a colour bar for reference
set(gca, 'XTick', [1:1:21], 'XTickLabels',results.Properties.VariableNames);
xtickangle(270);
set(gca, 'YTick', [1:1:21], 'YTickLabels',results.Properties.RowNames);
title('10 Fold Loss');
ylabel('Weighting (4<z<=5)'); %label x axis
xlabel('Weighting (z>5)'); %label y axis