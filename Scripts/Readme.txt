README

** Quickstart Guide **
NBTest.m and RFTest.m are the scripts to run final Naive Bayes & Random Forest models on test set.


######################
All code has been created and tested using MATLAB 2020a Update 5 (9.8.0.1451342), 64-bit Mac OS version.

Code has been tested on City Lab Machines using MATLAB 2020a - all modelling scripts run perfectly, but normalisation behaviour is different which causes some differences in graph production in the basic stats file. As this is not critical for modelling, I have not adjusted these pieces of code to work on the city machines.


#####################
File Guidance:


- BasicStats.m - code to generate basic statistics and visualisations of the data

- dataR2.csv - accessed from the UCI ML Database: https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Coimbra

- DataSplit.m - code to partition the data into training and test sets, and to apply the weights found to be optimal during workflow (see supplementary material for details)

- FOLDER: Figs: - contains figures generated from scripts for quick reference. Many of these are also included in poster.

- FOLDER: NB Intermediate Work: - contains scripts used during development, details of what each was used for and key findings are given in supplementary material. Provided for completeness, not necessary to run.

- nb_final.mat - final Naive Bayes model for use on test set

- NBBest.m - script to generate final Naive Bayes model and cross-validate

- NBTest.m - script to apply the best Naive Bayes model to the test data, and analyse performance

- FOLDER: RF Intermediate Work: - contains scripts used during development, details of what each was used for and key findings are given in supplementary material. Provided for completeness, not necessary to run.


- Readme.txt - README file

- RFfinal.mat - final Random Forest model for use on test set

- RFLoop.m - script used for manual grid search of parameters, and then final training. I've removed all of the values in the For loops that I used during my grid search besides the ones I found to be optimal, to facilitate faster computation.

- RFOpt.m - script to generate a tuned Random Forest model and cross validate 
*** WARNING - will take c. 20 minutes to run, and may provide different results on each run****

- RFTest.m - script to apply the optimised Random Forest model to the test data, and analyse results.

- Significance.m - script to run a McNemar test on the hypothesis statement NB and RF would predict holdout data equally well.

- test.csv - test data (33%), generated using DataSplit.m

- train.csv - training data (66%), generated using DataSplit.m


##############
All scripts are written to run on data found in the same directory as the scripts, so if you are looking to run any of my intermediate works you will need to move the relevant csv file into that folder to do so. 
