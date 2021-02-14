%Group members: 
%Nedzma Dervisbgoviç: 99209546914
%Kaouther MOUHEB: 99926527616

clc;
close all;
clear all;

%Loading data
% spam
data = readtable('sms_spam.txt','delimiter','\n', 'TextType','string', 'ReadVariableNames', 0, 'Encoding', 'UTF-8');
% adding a label with value 0 for 'spam'
data.('label') = zeros(420,1);
% legitimate
data2 = readtable('sms_legitimate.txt','delimiter','\n', 'TextType','string', 'ReadVariableNames', 0, 'Encoding', 'UTF-8');
% adding a label with value 1 for 'legitimate'
data2.('label') = ones(430,1);

% combining the two tables
dataset = [data;data2];

% updating the column name for message content in the table
dataset.Properties.VariableNames([1]) = {'message'};

% partitioning data to train and test sets
cvp = cvpartition(dataset.label,'Holdout',0.3);
dataTrain = dataset(cvp.training,:);
dataTest = dataset(cvp.test,:);

% extracting the messages and labels from the table
messageTrain = dataTrain.message;
messageTest = dataTest.message;
labelTrain = dataTrain.label;
labelTest = dataTest.label;

% preprocessing: 
documents_train = preprocess(messageTrain);
documents_test = preprocess(messageTest);

% feature selection
[new_train, new_train_labels, new_test, bag, feat_idx] = feature_selection(documents_train, labelTrain, documents_test, 80, 15);

% training the classifier:
mdl = fitclinear(new_train,new_train_labels);

% testing the classifer on the test data
predictions = predict(mdl,new_test);

% building the confusion matrix
confusion_mat = confusionmat(predictions,labelTest);
confusionchart(confusion_mat)

TP = confusion_mat(1,1);
FN = confusion_mat(1,2);
FP = confusion_mat(2,1);
TN = confusion_mat(2,2);

% calculating the assessment metrics for the model
accuracy = (TP+TN) /(TP+FN+FP+TN);
recall = TP / (TP+FN);
precision = TP / (TP+FP);
f_score = 2*(precision*recall)/(precision + recall);

f_s_s = sprintf('The f-score is %d ', f_score);
disp(f_s_s);





