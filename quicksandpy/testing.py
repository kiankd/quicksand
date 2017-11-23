import numpy as np
from collections import defaultdict
from quicksand.quicksandpy.classifier import test_classifiers
from quicksand.quicksandpy.load_data import get_all_data
from quicksand.quicksandpy.tweet import load_tweets
from quicksand.quicksandpy.preprocessing import preprocess_tweets


def test_classifiers_basic():
    trainx, trainy, testx, testy = get_all_data()
    test_classifiers(trainx, testx, trainy, testy, 5)


if __name__ == '__main__':
    test_classifiers_basic()

    # test preprocessing on small data.
    if False:
        tweets = load_tweets('../../data/test_tweets.csv')
        preprocess_tweets(tweets, verbose=False)
        np.save('../../data/test_preproc.npy', np.array(tweets))
        tweets = np.load('../../data/test_preproc.npy')
        for tweet in tweets:
            print(tweet.orig_text)
            print(tweet.uncorrected_tokens)
            print(tweet.corrected_tokens)
            print()

    if False:
        # test loading all data
        all_tweets = load_tweets('../../data/tweets_final.csv')
        preprocess_tweets(all_tweets, verbose=False)
        print(f'Number of tweets: {len(all_tweets)}')
        print(f'Topic distribution:')
        topics = defaultdict(lambda: [])
        for tweet in all_tweets:
            topics[tweet.topic].append(tweet)
        for topic in topics:
            print(f'\tTopic {topic}: {len(topics[topic])} tweets.')

        np.save('../../data/preprocessed_tweets.npy', np.array(all_tweets))