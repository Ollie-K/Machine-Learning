%clear all data etc.
clear all; clc; close all;

%import data file, including headers
breast = readtable('dataR2.csv', 'PreserveVariableNames',true); %import data

%obtain shape and size of data
[rows cols] = size(breast) %find number of rows and columns

breast{:,10} = breast{:,10} - 1; %change classification to 0 & 1 for control & cancer, respectively 

%split data into healthy controls and cancer patients
cont = breast(breast.Classification == 0, :); %create table of control group
canc = breast(breast.Classification == 1, :); %create table of cancer patients

%summary stats
summ(1,:) = mean(cont{:,1:9}); %find mean for control group
summ(2, :) = mean(canc{:,1:9}); %find mean for cancer group
summ(3, :) = std(cont{:,1:9}); %find standard deviation for control group
summ(4, :) = std(canc{:,1:9}); %find standard deviation for cancer group
summ(5, :) = median(cont{:,1:9}); %find median for control group
summ(6, :) = median(canc{:,1:9}); %find median for cancer group
rownames = {'mean(cancer)'; 'mean(control)'; 'std(cancer)';...
    'std(control)'; 'median(cancer)'; 'median(control)'}; %create list of row names
summary = array2table(summ, 'VariableNames', ...
    breast.Properties.VariableNames(1:9), 'RowNames', rownames) %create summary table of the above

% view histograms of each
figure(1)
for i = 1:9 %loop through each variable
    subplot(3,3,i); %select subplot corresponding to that variable on 3x3 grid
    r = max(breast{:,i})-min(breast{:,i}); %find range of that variable
    h1 = histogram(cont{:,i}, 'Normalization','pdf','BinWidth', r/10); ... 
        %plot normalised histogram of control patients, with 10 bins for
        %variable across both groups
    set(h1, 'FaceColor', 'blue'); %set colour to blue
    hold on %use same axes
    h2 = histogram(canc{:,i}, 'Normalization','pdf', 'BinWidth', r/10); ...
        %superimpose a  normalised histogram of cancer patients, with 10 bins for
        %variable across both groups
    set(h2, 'FaceColor', 'red') %colour this red 
    title(breast.Properties.VariableNames(i)); %title the histogram with the variable
    hold off %leave these axes for the next item in loop
end
legend('Control', 'Cancer') %add legend to the final plot

%identify outliers by z score
Z0 = zscore(breast{:,1:9}); %find z score for all variables except patient classification
figure(2); %start new figure
imagesc(abs(Z0)); %plot z scores by colour
colorbar; %include a colour bar for reference
set(gca, 'XTickLabels',breast.Properties.VariableNames); %label the x axis with variable names
[zr, zc] = find(abs(Z0)>5); %find the coordinates of those 
Outliers = breast(zr, :) %list those with very high z scores

%pp plot to visualise whether normally distributed
figure(3); %new figure
for i = 1:9 %loop through variables
    subplot(3,3,i); %select relevant supbplot on 3x3 grid
    pp1 = probplot(cont{:,i}) %create normal probability plot for that variable, for control group 
    set(pp1, 'color', 'blue') %colour it blue
    hold on %use same axes
    pp2 = probplot(canc{:,i})%add normal probability plot for that variable, for cancer group
    set(pp2, 'color', 'red') %colour it red
    title(breast.Properties.VariableNames(i)); %title it with the variable name
    hold off %leave these axes for the next item in loop
end

%log transformed pp plot
figure(4);
for i = 1:9 %loop through variables
    subplot(3,3,i); %select relevant supbplot on 3x3 grid
    pp1 = probplot(log(cont{:,i})) %create normal probability plot for the log of that variable, for control group
    set(pp1, 'color', 'blue') %colour it blue
    hold on % use the same axes
    pp2 = probplot(log(canc{:,i})) %add normal probability plot for the log of that variable, for cancer group
    set(pp2, 'color', 'red') %colour it red
    title(breast.Properties.VariableNames(i)); %title it with the variable name
    hold off %leave these axes for the next item in loop
end

%correlation heatmap
figure(5) %new figure
imagesc(corrcoef(breast{:,1:9})) %create a heatmap of correlation coefficients for each variable
set(gca, 'XTickLabels',breast.Properties.VariableNames, 'YTickLabels',breast.Properties.VariableNames) %label it
xtickangle(90) %put labels at 90 degrees for legibility
colormap('pink') %select appropriate colourmap
colorbar %include a colourbar for reference

%boxplot of variables
figure(6) %new figure
for i = 1:9 %loop through variables
    subplot(3,3,i); %select relevant subplot on 3x3 grid
    b1 = boxplot(breast{:,i}, breast.Classification); %create boxplot of values for that variable, for each class 
    xlabel('Patient Classification'); %label the x axis 
    title(breast.Properties.VariableNames(i)); %title each plot with the variable name

end

normalbreast = normalize(breast{:,1:9}); %create matrix of standardized values for each variable
normalbreast(:,10) = breast{:,10}; %append the classifications to this matrix.

%parallel coordinate plot to visualise differences between variables
figure(7); %new figure
parallelcoords(normalbreast(:,1:9), 'Group', normalbreast(:,10), 'Quantile', 0.25); ...
    %create parallel coordinate plot of each variable, with dotted line
    %for each quantile
xticklabels(breast.Properties.VariableNames(1:9)); %label the axes

%ranking of predictor importance to aid in feature selection
[idx,scores] = fscmrmr(breast{:,1:9},breast{:,10}); %rank features by minimum redundancy maximum relevance 
figure(8) %new figure
bar(scores(idx)); %plot bar chart of these scores
xlabel('Predictor rank'); %label x axis
ylabel('Predictor importance score'); %label y axis
xticklabels(strrep(breast.Properties.VariableNames(idx),'_','\_')); %add variable names
xtickangle(90); %rotate x ticks for legibility