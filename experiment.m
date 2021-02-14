% In this script, we are testing the classifier with multiple combinations
% of our parameters, namely: the number of features selected and the number
% of neighbors for the relief algorithm

clc;
close all;
clear all;

import zemberek.morphology.TurkishMorphology

%Loading data
% spam
data = readtable('sms_spam.txt','delimiter','\n', 'TextType','string', 'ReadVariableNames', 0, 'Encoding', 'UTF-8');
% adding a label with value 0
data.('label') = zeros(420,1);
% legitimate
data2 = readtable('sms_legitimate.txt','delimiter','\n', 'TextType','string', 'ReadVariableNames', 0, 'Encoding', 'UTF-8');
% adding a label with value 1
data2.('label') = ones(430,1);

% combining the two tables
dataset = [data;data2];

dataset.Properties.VariableNames([1]) = {'message'};

% partitioning data to train and test sets
cvp = cvpartition(dataset.label,'Holdout',0.3);
dataTrain = dataset(cvp.training,:);
dataTest = dataset(cvp.test,:);

% Extracting the messages and labels from the table
messageTrain = dataTrain.message;
messageTest = dataTest.message;
labelTrain = dataTrain.label;
labelTest = dataTest.label;

% Preprocessing: 
documents_train = preprocess(messageTrain);
documents_test = preprocess(messageTest);


% defining the options for each parameter 
neighbors = [3, 5, 10, 15, 20, 25];
feat_no = [50, 60, 70, 80, 90, 100];
accuracies = zeros(6); 

%feature selection
bag = bagOfWords(documents_train);
% removing words that appear in one document only
bag = removeInfrequentWords(bag,2);
% removing empty documents from the training set, if any
[bag,idx] = removeEmptyDocuments(bag);
labelTrain(idx) = [];

% applying the TF-IDF weighting to training and testing sets
messageTrain = tfidf(bag);
messageTest = tfidf(bag,documents_test);

for i=1:6
    % calculating the weights of features for the i^th option of k
    k = neighbors(1,i);
    [idx,weights] = relieff(messageTrain, labelTrain, k, 'Method', 'classification');
    for j=1:6
        % selection the fn most important features  from the training and testing sets 
        % where fn = j^th option for the features number
        fn = feat_no(1,j);
        feat_idx = idx(1, 1:fn);
        new_train_labels = labelTrain;
        new_train = messageTrain(:, feat_idx);
        new_test = messageTest(:, feat_idx);
       
        %training the Classifier: 
        mdl = fitclinear(new_train,new_train_labels);
        % testing the classifier
        predictions = predict(mdl,new_test);
        acc = sum(predictions == labelTest)/numel(labelTest);
        % saving the accuracy for the (i,j) combination
        accuracies(i, j) = acc;
    end   
end
% finding the maximum accuracy
maximum = max(max(accuracies));
% finding the optimal parameters, if more than one are found take the first
% option
[k_op, fn_op]=find(accuracies == maximum);
op_k = neighbors(1,k_op(1,1));
op_f = feat_no(1,fn_op(1,1));

% displaying the results on the screen
Res1 = sprintf('The most optimal number of features is %d ', op_f);
Res2 = sprintf('The most optimal number of neighbors for relief is %d ', op_k);
disp(Res1);
disp(Res2);

