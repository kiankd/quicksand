# quick fix
import sys
sys.path.append('/home/ml/kkenyo1/quicksand/')

import numpy as np
import argparse
from collections import defaultdict
from csv import DictReader, DictWriter
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.metrics import classification_report, accuracy_score, f1_score
from quicksand.quicksandpy.util import *
from quicksand.quicksandpy.feature_extraction import extract_features
from quicksand.quicksandpy.preprocessing import preprocess_tweets
from quicksand.quicksandpy.classifier import LogisticRegression, HierClassifier, LinearSVC
from quicksand.quicksandpy.tweet import Tweet

# utility
def get_y(tweets, label_setting):
    return np.array([tweet.get_labelling(label_setting) for tweet in tweets])

def get_X(tweets, features=ALL_FEATURES):
    return np.array([tweet.get_feature_vector(selected_feats=features) for tweet in tweets])

def get_results(gold_labels, pred_labels):
    s = classification_report(gold_labels, pred_labels) + '\n'
    s += '\nFinal F1-Accuracy: {}\nFinal normal Accuracy: {}'.format(
        f1_score(gold_labels, pred_labels, average='weighted'),
        accuracy_score(gold_labels, pred_labels),
    )
    return s

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
    print('Preprocessing data...')
    preprocess_tweets(tweets, verbose=False)

    # save data if desired
    if serialize:
        train_tweets, test_tweets = train_test_split(tweets, test_size=0.2, shuffle=True, random_state=1917)
        print('Extracting features...')
        extract_features(train_tweets, test_tweets)
        np.save('../labelled_data/train_tweets_with_labels.npy', np.array(train_tweets))
        np.save('../labelled_data/test_tweets_with_labels.npy', np.array(test_tweets))

    return tweets

def load_train_test():
    train_tweets = np.load('../labelled_data/train_tweets_with_labels.npy')
    test_tweets = np.load('../labelled_data/test_tweets_with_labels.npy')
    return list(train_tweets), list(test_tweets)

# classification testing
def run_leave_one_out(tweets, label_setting, model, model_name, features=BEST_FEATURES):
    X = get_X(tweets, features=features)
    y = get_y(tweets, label_setting)

    all_model_preds = []
    test_tweets = []

    loo = LeaveOneOut()
    loo.get_n_splits(X, y)

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        assert(len(y_test) == 1)

        print('Fitting model {}...'.format(len(test_tweets)))
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        assert(len(pred) == 1)
        all_model_preds.append(pred[0])
        test_tweets.append(tweets[test_idx[0]])

        if len(test_tweets) % 10 == 0:
            print('Completed {} tests out of {}...'.format(len(test_tweets), len(X)))
            print('Aka, {}% complete...\n'.format(float(len(test_tweets)) / len(X)))

    # save the results in explicit detail over all samples
    fname_with_path = get_loo_results_fname(model_name, label_setting)
    with open(fname_with_path + '.csv', 'w') as csvf:
        writer = None
        for i, tweet in enumerate(test_tweets):
            d = {
                'tid': tweet.tid,
                'gold_label': tweet.get_labelling(label_setting),
                'model_pred_label': all_model_preds[i],
                'text': '\"{}\"'.format(tweet.orig_text.strip('\"')),
                'topic': tweet.topic,
            }
            annotated_labellings = {'{}_annotation_count'.format(label): tweet.labelling[label] for label in LABELS}
            d.update(annotated_labellings)

            # set the headers
            if i == 0:
                field_names = list(d.keys())
                writer = DictWriter(csvf, fieldnames=field_names)
                writer.writeheader()

            writer.writerow(d)

    # save the specific results
    with open(fname_with_path + '.txt', 'w') as f:
        detail_results = classification_report(y[:len(all_model_preds)], all_model_preds, digits=3)
        final_f1 = f1_score(y[:len(all_model_preds)], all_model_preds, average='weighted')
        final_acc = accuracy_score(y[:len(all_model_preds)], all_model_preds)
        f.write(detail_results)
        f.write('\n\n----------------------------\n')
        f.write('Final F1 Accuracy: {}\n'.format(final_f1))
        f.write('Final Accuracy: {}\n'.format(final_acc))

def feature_ablation(train, test, model, label_setting):
    print('Feature ablation results:\n\n')
    feature_sets = [
        ALL_FEATURES,
        {UNIGRAMS, BIGRAMS},
        {UNIGRAMS, BIGRAMS, WEMB},
        {SENTIWN_WEMBS, SENTIWN},
    ]
    for feature in ALL_FEATURES:
        feature_sets.append(ALL_FEATURES.difference({feature}))
        feature_sets.append({feature})

    best_f1_feats = [0, set([])]
    best_acc_feats = [0, set([])]

    for feats in feature_sets:
        train_X = get_X(train, feats)
        train_y = get_y(train, label_setting)

        test_X = get_X(test, feats)
        test_y = get_y(test, label_setting)

        model.fit(train_X, train_y)
        preds = model.predict(test_X)

        f1_acc = f1_score(test_y, preds, average='weighted')
        acc = accuracy_score(test_y, preds)
        for score, holder in [(f1_acc, best_f1_feats), (acc, best_acc_feats)]:
            if score > holder[0]:
                holder[0] = score
                holder[1] = feats

        fstring = 'USING FEATURE SET {}'.format(feats)
        print('TEST SET RESULTS {}:\n'.format(fstring))
        print(get_results(test_y, preds))
        print('-----------------------------\n')

    print('\n\nBEST RESULTS:')
    print('Best F1-accuracy: {}\nwith features: {}\n'.format(best_f1_feats[0], best_f1_feats[1]))
    print('Best accuracy: {}\nwith features: {}\n'.format(best_acc_feats[0], best_acc_feats[1]))

# results analyzing
def analyze_results(fname):
    results = []
    with open(fname + '.csv', 'r') as csvf:
        reader = DictReader(csvf)
        for row in reader:
            results.append(row)

    # first, print statistics of the data
    gold_label_counts = defaultdict(lambda: 0)
    pred_label_counts = defaultdict(lambda: 0)
    topic_counts = defaultdict(lambda: 0)
    for row in results:
        gold_label_counts[row['gold_label']] += 1
        pred_label_counts[row['model_pred_label']] += 1
        topic_counts[row['topic']] += 1

    datas = [
        (gold_label_counts, 'Gold Label Counts'),
        (pred_label_counts, 'Predicted Label Counts'),
        (topic_counts, 'Topic Counts'),
    ]

    print('\n\n--DATASET STATISTICS--')
    for d, name in datas:
        print('{}:'.format(name))
        for key, value in d.items():
            print('\t{} - {}'.format(key, value))
    print('\n')

    # now get results over the different levels of agreement
    for percent_agreement in [0, 0.5, 0.79, 0.99]:
        filtered_tweets = []
        for row in results:
            labelling = {}
            for label in LABELS:
                labelling[label] = int(row['{}_annotation_count'.format(label)])
            total = sum(labelling.values())
            if max(labelling.values()) / float(total) > percent_agreement:
                filtered_tweets.append(row)

        # now we print the results for this specific granularity
        gold_labels = [row['gold_label'] for row in filtered_tweets]
        pred_labels = [row['model_pred_label'] for row in filtered_tweets]

        print('\n\n\n--RESULTS WHEN EVALUATING ONLY TWEETS THAT HAD > {} PERCENT AGREEMENT--'.format(percent_agreement))
        print(get_results(gold_labels, pred_labels))


# initialization function
def initialize(load_glove=False):
    # this is for initializing the data and getting word embeddings
    if load_glove:
        tweets = load_tweets_from_csv('../../data/f1209851.csv', serialize=False)
        vocab = []
        for tweet in tweets:
            vocab += tweet.corrected_tokens
        raw_load_and_extract_glove(vocab, EMB_SIZE, '../labelled_data/')

    # feature extraction and serialization
    load_tweets_from_csv('../../data/f1209851.csv', serialize=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'label_setting', type=str,
        help='label setting used to decide on how we use annotations to set labels'
    )
    parser.add_argument(
        'model', type=str,
        help='model that we will test with'
    )
    parser.add_argument(
        '-a', '--analyze', action='store_true',
        help='analyze results from the specified model'
    )
    parser.add_argument(
        '-f', '--ablation', action='store_true',
        help='perform feature ablation study'
    )
    parser.add_argument(
        '-i', '--initialize', action='store_true',
        help='intialize the data'
    )
    args = parser.parse_args()


    # run leave-one-out CV
    print('Loading data...')
    train, test = load_train_test()

    # get label setting and model from argv
    models = {
        LOGISTIC_REGRESSION: LogisticRegression(),
        HIERARCHICAL: HierClassifier(),
        LINEAR_SVM: LinearSVC(),
    }

    label_setting = args.label_setting
    model = models[args.model]

    if args.analyze:
        print('Analyzing results for model {} with label setting {}...'.format(args.model, label_setting))
        analyze_results(get_loo_results_fname(args.model, label_setting))
    elif args.ablation:
        feature_ablation(train, test, model, label_setting)
    elif args.initialize:
        initialize(load_glove=False)
    else:
        print('Running leave-one-out cross validation with model {} and labels {}...'.format(args.model, label_setting))
        run_leave_one_out(train + test, label_setting, model, args.model)

