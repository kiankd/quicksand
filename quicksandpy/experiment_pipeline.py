import numpy as np
from collections import defaultdict
from csv import DictReader
from sklearn.model_selection import train_test_split
from quicksand.quicksandpy.util import *
from quicksand.quicksandpy.feature_extraction import extract_features
from quicksand.quicksandpy.preprocessing import preprocess_tweets
from quicksand.quicksandpy.tweet import Tweet

# primary function
def load_tweets_from_csv(fname, serialize=True):
    # Load the data into memory
    ids_to_content = defaultdict(lambda: [])
    with open(fname) as f:
        csv_reader = DictReader(f)
        for i, row in enumerate(csv_reader):
            ids_to_content[row[ID_KEY]].append(row)

    # construct the tweets and labels
    tweets = []
    for tid in ids_to_content.keys():
        sample = ids_to_content[tid]
        first_tweet = sample[0]

        # skip the test questions!
        if first_tweet[IS_GOLD_KEY] == 'true':
            continue

        # build up the tweet statistics of labels
        tweet_stats = {s: 0 for s in LABELS}
        for labelling in sample:
            tweet_stats['obj'] += 1 if labelling[HAS_SENTIMENT_KEY] == 'no' else 0
            for key in CSV_LABELS:
                tweet_stats[key[0:3]] += 1 if labelling[POS_NEG_COM_KEY] == key else 0

        # extract the necessary data
        tweet = Tweet(first_tweet[TWEET_ID], first_tweet['text'], first_tweet['topic'])
        tweet.labelling = tweet_stats
        tweets.append(tweet)

    # always want to preprocess
    preprocess_tweets(tweets, verbose=False)

    # save data if desired
    if serialize:
        train_tweets, test_tweets = train_test_split(tweets, test_size=0.2, shuffle=True, random_state=1917)
        extract_features(train_tweets, test_tweets)
        np.save('../labelled_data/train_tweets_with_labels.npy', np.array(train_tweets))
        np.save('../labelled_data/test_tweets_with_labels.npy', np.array(test_tweets))

    return tweets

def run_pipeline(fname, label_option):
    all_x, all_y = load_tweets_from_csv(fname, serialize=False)

if __name__ == '__main__':
    # run_pipeline('../../data/f1209851.csv', '')

    # this is for initializing the data and getting word embeddings
    if False:
        tweets = load_tweets_from_csv('../../data/f1209851.csv', serialize=False)
        vocab = []
        for tweet in tweets:
            vocab += tweet.corrected_tokens
        raw_load_and_extract_glove(vocab, EMB_SIZE, '../labelled_data/')

    # feature extraction
    if True:
        tweets = load_tweets_from_csv('../../data/f1209851.csv', serialize=True)
