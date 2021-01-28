%clear all data etc.
clear all; clc; close all;
%import data file, including headers
rng('default'); % For reproducibility

test = readtable('test.csv', 'PreserveVariableNames',true); %import unseen test data
Xt = test(:,[1 3 8]); %select features (age, glucose & resistin)
Yt = test.Classification; %select targets (cancer status)

RF = load('RFfinal.mat').MdlFinal; %load final RF model
NB = load('nb_final.mat').mdl %import best NB model

YNB = predict(NB, Xt); %predict using NB
YRF = predict(RF, Xt); %predict using RF
[h, p, e1, e2] = testcholdout(YNB, YRF, Yt, 'Test', 'midp') %test null hypothesis (models equally accurate)