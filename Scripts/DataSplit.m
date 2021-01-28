%clear all data etc.
clear all; clc; close all;
%import data file, including headers
breast = readtable('dataR2.csv', 'PreserveVariableNames',true); %import data
rng('default'); % For reproducibility

breast{:,10} = breast{:,10} - 1; %set targets to 0 for control, 1 for cancer

%breast = [normalize(breast(:,1:9)) breast(:,10)]; %standardize all variables except class

%partitioning data into testing and training sets
cvpart = cvpartition(breast{:,10},'holdout',0.33); %partition data for training and testing
train = breast(training(cvpart),:); %training data
test = breast(test(cvpart),:); %test data

%Generating weights to diminish impact of outliers
Z1 = zscore(train{:,:}); %calculate z scores for model
[zr, zc] = find(abs(Z1)>4); %find outliers
weights = ones(size(train(:,10),1),1); %generate weighting of 1 for all values
weights(zr) = 0.3; %reduce weighting for outliers
[zr, zc] = find(abs(Z1)>5); %find outliers
weights(zr) = 0.125; %reduce weighting for outliers
weights = array2table(weights); %change to table
train = [train weights]; %join the table of weights to main table of data
train = movevars(train,'weights', 'Before', 'Classification'); %move weights to the left of patient class

%save these tables for use in all following work
writetable(train, 'train.csv'); %training set
writetable(test, 'test.csv'); %testing set