%clear all data etc.
clear all; clc; close all;
%import data file, including headers
rng('default'); % For reproducibility

test = readtable('test.csv', 'PreserveVariableNames',true); %import unseen test data
Xt = test(:,[1 3 8]); %select features (age, glucose & resistin)
Yt = test.Classification; %select targets (cancer status)

MdlFinal = load('RFfinal.mat').MdlFinal; %load final model

tic %start timer for prediction
for i = 1:1000 %due to function being very quick, looping to improve accuracy of timing
    [predclass,scores] = predict(MdlFinal, Xt); %applying the model to the unseen test data
end
prediction_time = toc/1000 %divide timer by 1000 to achieve time taken to predict
individual_pred_time = prediction_time/length(Yt) %time to predict for one patient

TestLoss = loss(MdlFinal, Xt, Yt) %calculating the loss on the test data
figure(1) %new figure
confusionchart(Yt, predclass) %displaying a confusion chart of the results
e = edge(MdlFinal,Xt,Yt); %calculate test sample edge
m = margin(MdlFinal, Xt, Yt);  %calculate test sample margin
figure
[X,Y,T,AUC] = perfcurve(Yt, scores(:,2), 1); %calculate ROC
plot(X,Y) %plot ROC

%plotting 3d predictive space using one variable per axis, using posterior region. 
%Commented out to save on computational load.

%figure(2)
%xMax = max(table2array(Xt)); %determine maximum values
%xMin = min(table2array(Xt)); %determine minimum values
%h = 0.5; %set interval for plotting
%[x1Grid,x2Grid,x3Grid] = meshgrid(xMin(1):h:xMax(1),xMin(2):h:xMax(2),xMin(3):h:xMax(3)); %create a 3d array of regularly spaced points in space from the minimum to maximum values of each variable
%[~,PosteriorRegion] = predict(MdlFinal,[x1Grid(:),x2Grid(:),x3Grid(:)]);%calculate the posterior probabilities for all of these points
%PosteriorRegion(:,3) = zeros(size(PosteriorRegion(:,2)));%creating list of zeros for colour mapping
%PosteriorRegion = PosteriorRegion(:,[2 3 1]);  %moving zeros to 'green'
%h = scatter3(x1Grid(:),x2Grid(:),x3Grid(:),5,PosteriorRegion);%plotting these points coloured according to posterior
%h.MarkerEdgeAlpha = 0.05;%setting transparency

%hold on %use the same axes
%scatter3(Xt{:,1},Xt{:,2},Xt{:,3}, 100, Yt, 'LineWidth', 2) %plot the true classes as circles

%scatter3(Xt{:,1},Xt{:,2},Xt{:,3}, 100, predclass, 'Marker',...
%    'x', 'LineWidth', 2) %plot the predicted classes as crosses
%hold off %stop using the same axes
%xlabel('Age (years)') %label x axis
%ylabel('Glucose (mg/dL)') %label y axis
%zlabel('Resistin (ng/mL)') %label z axis
%colormap(jet) %assign red for cancer, blue for no cancer

%view(90,270) %set views [comment out, use (0,0), (90,0) and (90,270)]

view(MdlFinal.Trained{1}, 'Mode', 'graph'); %view sample decision tree from ensemble

results = table(individual_pred_time, TestLoss, AUC, e, [sum((predclass == 0) & (Yt == 0))], ...
    [sum((predclass == 1) & (Yt == 0))], [sum((predclass == 0) & (Yt == 1))], ...
    [sum((predclass == 1) & (Yt == 1))], ...
    [100 * sum(sum((predclass == 0) & (Yt == 0)) + sum((predclass == 1) & (Yt == 1))) / sum((Yt==1) | (Yt==0))],...
    [100 * sum((predclass == 1) & (Yt == 1)) / sum(Yt == 1)], ...
    [100 * sum((predclass == 0) & (Yt==0))/ sum(Yt == 0)]); %tabulating results
results.Properties.VariableNames = {'Prediction Time', 'Test Loss', 'ROC AUC', 'edge', 'TN', 'FP', 'FN', 'TP', ...
    'Accuracy(%)', 'Sensitivity (%)', 'Specificity (%)'} %adding headers to table