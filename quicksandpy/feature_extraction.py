from nltk.corpus import sentiwordnet as swn
from sklearn.feature_extraction.text import CountVectorizer
from quicksand.quicksandpy.util import *

# primary function
def extract_features(train_tweets, test_tweets):
    all_tweets = train_tweets + test_tweets

    # ngrams
    unigramer = CountVectorizer(ngram_range=(1,1), min_df=5, max_features=5000)
    bigramer = CountVectorizer(ngram_range=(2,2), min_df=5, max_features=5000)

    # note we are extracting n-grams from the uncorrected tokens!
    train_docs = [' '.join(tweet.uncorrected_tokens) for tweet in train_tweets]
    for extractor, feature_name in [(unigramer, UNIGRAMS), (bigramer, BIGRAMS)]:
        extractor.fit(train_docs)
        for tweet in all_tweets:
            tweet.features[feature_name] = extractor.transform([' '.join(tweet.uncorrected_tokens)])

    all_tweets[0].get_feature_vector()

    # word embeddings!
    # wemb_dict = get_glove_data()
    # for tweet in all_tweets:
    #     embs = []
    #     for word in tweet.corrected_tokens:
    #         if word in wemb_dict:
    #             embs.append(wemb_dict[word])
    #     if len(embs) == 0:
    #         avg_emb = np.zeros((EMB_SIZE))
    #     else:
    #         avg_emb = np.mean(embs, axis=1)
    #     pass
