% Loading the model
load model.mat
load bag.mat
load feats.mat

open = true;
% Testing for individual message
while (open)
    prompt = 'Enter your sms for testing or "close" to quit\n'; 
    in = input(prompt, 's');
    
    r = strcmp(in, "close");
    if (r==1)
        open = false;
    else
        message = in;
        % predicting 
        prediction = predict_message(message, bag, feat_idx, mdl);
        % displaying the result
        result = sprintf('Your SMS is %s', prediction);
        disp(result)        
    end        
end