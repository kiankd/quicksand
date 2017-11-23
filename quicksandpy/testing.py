import numpy as np
from quicksand.quicksandpy.tweet import load_tweets
from quicksand.quicksandpy.preprocessing import preprocess_tweets
from collections import defaultdict

if __name__ == '__main__':
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

    if True:
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