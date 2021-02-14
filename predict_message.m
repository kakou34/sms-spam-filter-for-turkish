function [prediction] = predict_message(message, bag, feat_idx, mdl)
    % preprocess the message text
    document = preprocess(message);
    message_data = tfidf(bag,document);
    message_data = message_data(1, feat_idx);

    % predict using the given model
    prediction = predict(mdl,message_data);

    % converting the result to the class name
    if prediction == 0
        prediction = 'Spam';
    elseif prediction == 1
        prediction = 'Legitimate';  
    end
end

