%clear all data etc.
clear all; clc; close all;
%import data file, including headers
breast = readtable('train.csv', 'PreserveVariableNames',true); %import data
rng('default'); % For reproducibility

X = breast(:,[1 3 8]); %select features
Y = breast.Classification; %select targets

cvpt1 = cvpartition(Y, 'KFold', 10); %creating 10-fold partitioning for cross-validation of trained model

%loop below inspired by example code on https://uk.mathworks.com/help/stats/fitcensemble.html

maxNumSplits = [10]; %choose values for maximum numbers of splits
numMNS = numel(maxNumSplits); %define number of items in this dimension
numTrees = [20]; %choose values for numbers of learners
numT = numel(numTrees); %define number of items in this dimension
minLeafSize = [2]; %choose values for minimum leaf sizes
numL = numel(minLeafSize); %define number of items in this dimension
mdl = cell(numMNS,numT,numL); %Define a cube of side lengths outlined above 
counter = 1; %to show progress through iterations
tic %time function
for k = 1:numT %iterating over Trees
    for j = 1:numMNS %iterating over maximum number of splits
        for l=1:numL %iterating over minimum leaf size
            t = templateTree('Reproducible',true, 'MaxNumSplits',maxNumSplits(j),'MinLeafSize',(l)); %for reproducibility, variables set by loops
            mdl{j,k,l} = fitcensemble(X, Y, 'Learners', t,...
                'CVPartition', cvpt1, 'Method', 'bag',...
                'NumLearningCycles', numTrees(k)); %fitting a partitioned Random Forest classifier to the training data, iterating over number of trees
            counter = counter + 1 %display iteration progress
        end
    end
end

kflAll = @(x)kfoldLoss(x); %define k fold loss function
error = cellfun(kflAll,mdl); %apply k fold loss function to all models, storing them in a cube
kfp = @(x)kfoldPredict(x); %define prediction function
predclass = cellfun(kfp,mdl, 'UniformOutput',false); %apply to each model and store
confusionchart(Y,predclass{1,1}) %display confusion chart (used for final model only)

[minErr,minErrIdxLin] = min(error(:)); %find the minimum error and its location
[idxMNS,idxT,idxL, idxLR] = ind2sub(size(error),minErrIdxLin); %convert to coordinates

fprintf('\nMinimum error = %0.5f',minErr) %display minimum error

fprintf('\nOptimal Parameter Values:'); %display best values

fprintf('\nMaxNumSplits = %d\nMaxNumTrees = %d\nMinLeafSize = %d\n',...
    maxNumSplits(idxMNS),numTrees(idxT),minLeafSize(idxL)) %locate & print optimal values for each parameter using coordinates
toc %stop timer




tFinal = templateTree('Reproducible',true, 'MaxNumSplits',maxNumSplits(idxMNS), ...
    'MinLeafSize', minLeafSize(idxL)); %define optimal template tree
MdlFinal = fitcensemble(X,Y,'NumLearningCycles',numTrees(idxT),...
    'Method', 'bag', 'Learners',tFinal) %define optimal final model

cvmdl = crossval(MdlFinal, 'CVPartition', cvpt1); %cross-validating the model
[predclass10, score10] = kfoldPredict(cvmdl); %applying the cross validated model to the data
[X10,Y10,T10,AUC10] = perfcurve(Y, score10(:,2), 1); %calculating ROC
figure
plot(X10,Y10); %plotting ROC

cvpt2 = cvpartition(Y, 'KFold', 5); %creating 5-fold partitioning for cross-validation of trained model
cvmdl1 = crossval(MdlFinal, 'CVPartition', cvpt2); %cross-validating the model
[predclass, score5] = kfoldPredict(cvmdl1); %applying the cross validated model to the data
figure(4)
confusionchart(Y, predclass) %displaying a confusion chart of the results
[X5,Y5,T5,AUC5] = perfcurve(Y, score5(:,2), 1); %calculate ROC
figure
plot(X5,Y5); %plotting ROC

Loss = resubLoss(MdlFinal) %calculating the resubstitution loss



%save('RFfinal.mat', 'MdlFinal') %save final model
