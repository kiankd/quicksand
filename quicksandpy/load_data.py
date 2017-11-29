import numpy as np
from collections import defaultdict
from csv import DictReader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from quicksand.quicksandpy.preprocessing import preprocess_tweets
from quicksand.quicksandpy.tweet import Tweet

def load_preproc_tweets():
    return np.load('../../data/preprocessed_tweets.npy')

def load_csv_tweets():
    ids_to_content = defaultdict(lambda: [])
    with open('../../data/f1209851.csv') as f:
        csv_reader = DictReader(f)
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            ids_to_content[row['_unit_id']].append(row)
    return ids_to_content

def csv_tweets_to_xy(csv_tweets):
    new_x = []
    new_y = []
    for tid in csv_tweets.keys():
        sample = csv_tweets[tid]
        tweet_stats = {s: 0 for s in ['pos', 'neg', 'com', 'obj']}
        for labelling in sample:
            tweet_stats['obj'] += 1 if labelling['does_the_author_express_sentiment_in_this_tweet_'] == 'no' else 0
            for key in ['positive', 'negative', 'complicated']:
                tweet_stats[key[0:3]] += 1 if labelling['is_the_sentiment_expressed_positive_or_negative_'] == key else 0
        label = max(tweet_stats, key=tweet_stats.get)
        new_x.append(sample[0]['text'])
        new_y.append(label)
        # print(f'\nTweet: {new_x[-1]}\nLabel: {label}')
        # print(tweet_stats)
        # continue
    return new_x, new_y

def get_all_data():
    preproc = np.load('../../data/preprocessed_csv_tweets_basic_labels.npy')
    train, test = train_test_split(preproc, test_size=0.2, shuffle=True, random_state=1917)

    trainx = [data[0] for data in train]
    trainy = [data[1] for data in train]

    testx = [data[0] for data in test]
    testy = [data[1] for data in test]

    # unigrams
    vect = CountVectorizer(max_features=10000, ngram_range=(1,2))
    print('Fitting count vect...')
    train_toks = [' '.join(tweet.uncorrected_tokens) for tweet in trainx]
    xtrain = vect.fit_transform(train_toks)
    xtest = vect.transform([' '.join(tweet.uncorrected_tokens) for tweet in testx])

    return xtrain, trainy, xtest, testy


if __name__ == '__main__':
    x, y = csv_tweets_to_xy(load_csv_tweets())
    tweets = [Tweet(i, t, 'N/A') for i, t in enumerate(x)]
    preprocess_tweets(tweets)
    np.save('../../data/preprocessed_csv_tweets_basic_labels.npy', np.array(list(zip(tweets, y))))

    # tweets = np.load('../../data/preprocessed_csv_tweets_basic_labels.npy')
    # for data in tweets:
    #     tweet = data[0]
    #     label = data[1]
    #     print(tweet.orig_text)
    #     print(label)
    #     print(tweet.uncorrected_tokens)
    #     print(tweet.corrected_tokens)
    #     print()