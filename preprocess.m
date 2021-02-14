function documents = preprocess(text)

    % importing and inisializing the Resha turkish stemmer
    import com.hrzafer.reshaturkishstemmer.Resha
    stemmer = Resha.Instance;

    % text tokenization
    documents = tokenizedDocument(text);

    % erasing ponctuation
    documents = erasePunctuation(documents);

    % converting all characters to lower case
    documents = lower(documents);

    % removing numerals
    voc = documents.Vocabulary;
    pat = digitsPattern;
    idx = find(contains(voc,pat));
    documents = removeWords(documents,idx);

    % Removing urls, emails
    idx = find(contains(voc,'.'));
    documents = removeWords(documents,idx);

    % removing stop words
    stopwords = readtable('stopwords.txt','delimiter','\n', 'TextType','string', 'ReadVariableNames',0);
    stopwords = table2array(stopwords);
    stopwords = stopwords';
    documents = removeWords(documents, stopwords);

    % stemming 
    voc = documents.Vocabulary;
    [r, c] = size(voc);
    newVoc = strings(r,c);
    for i=1:c
        wor = voc(1,i);
        stem = stemmer.stem(wor);
        if  stem.length > 5
            % first n-prefix algorithm
            stem = stem.substring(0,5);    
        end
        newVoc(1,i) = stem;
    end
    % replacing each word with its root
    documents = replaceWords(documents,voc,newVoc);

end

