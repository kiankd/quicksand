import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

def load_preproc_tweets():
    return np.load('../../data/preprocessed_tweets.npy')

def get_all_data():
    preproc = load_preproc_tweets()
    train, test = train_test_split(preproc, test_size=0.2, shuffle=True, random_state=1917)

    # unigrams
    vect = CountVectorizer(max_features=10000, ngram_range=(1,2))
    print('Fitting count vect...')
    train_toks = [' '.join(tweet.uncorrected_tokens) for tweet in train]
    xtrain = vect.fit_transform(train_toks)
    xtest = vect.transform([' '.join(tweet.uncorrected_tokens) for tweet in test])

    return xtrain, [tweet.topic for tweet in train], xtest, [tweet.topic for tweet in test]


