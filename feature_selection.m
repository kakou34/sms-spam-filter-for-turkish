function [new_train, new_train_labels, new_test, bag, feat_idx] = feature_selection(documents_train, labelTrain, documents_test, feature_no, k)

    %converting documents to bag of words
    bag = bagOfWords(documents_train);

    %removing words that appear in less than 2 docs
    bag = removeInfrequentWords(bag,2);

    % deleting empty documents from training set, if any
    [bag,idx] = removeEmptyDocuments(bag);
    labelTrain(idx) = [];

    % weighting the words in each document using TF-IDF
    messageTrain = tfidf(bag);
    messageTest = tfidf(bag,documents_test);

    % ordering features by weights using relieff algorithm
    [idx,weights] = relieff(messageTrain, labelTrain, k, 'Method', 'classification');

    % choosing the most important features and returning the updated sets
    feat_idx = idx(1, 1:feature_no);
    new_train_labels = labelTrain;
    new_train = messageTrain(:, feat_idx);
    new_test = messageTest(:, feat_idx);

end

