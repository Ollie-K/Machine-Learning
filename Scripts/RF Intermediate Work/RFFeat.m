%clear all data etc.
clear all; clc; close all;
%import data file, including headers
breast = readtable('train.csv', 'PreserveVariableNames',true); %import data
rng('default'); % For reproducibility

X = breast(:,1:9); %select features
Y = breast.Classification; %select targets



cvpt1 = cvpartition(Y, 'KFold', 10); %creating 10-fold partitioning for cross-validation of trained model

opts = statset('Display','iter'); %display iterations
fun = @(XT,YT,Xt,Yt)loss(fitcensemble(XT,YT, 'Learners', 'tree'),Xt,Yt); %define function 
[fs,history] = sequentialfs(fun,table2array(X),Y,'cv', cvpt1, 'options',opts, ...
    'direction', 'backward', 'keepin', [3], 'keepout', [4 5]); %use sequential feature selection
fs %show final feature indexes
history.In %show how they were added iteratively
