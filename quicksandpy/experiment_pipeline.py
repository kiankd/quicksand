# quick fix
import sys
sys.path.append('/home/ml/kkenyo1/quicksand/')

import numpy as np
import argparse
from random import randint
from collections import defaultdict, Counter
from csv import DictReader, DictWriter
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from quicksand.quicksandpy.util import *
from quicksand.quicksandpy.feature_extraction import extract_features
from quicksand.quicksandpy.preprocessing import preprocess_tweets
from quicksand.quicksandpy.classifier import LogisticRegression, HierClassifier, LinearSVC, RandomForestClassifier
from quicksand.quicksandpy.tweet import Tweet


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

def filter_for_agreement(tweet_list, threshold, get_shit=False, upper=1):
    if get_shit:
        return [tweet for tweet in tweet_list if tweet.get_agreement() <= 0.5]
    return [tweet for tweet in tweet_list if threshold <= tweet.get_agreement() < upper]

def big_dataset_analysis(train, test, just_all=False):
    if just_all:
        assert(len(test)==0)
        data_and_names = [(list(train), 'Full data set (100%)')]
    else:
        all_d = train + test
        data_and_names = [
            (train, 'Random training set (80%'),
            (test, 'Random testing set (20%)'),
            (all_d, 'Full data set (100%)')
        ]

    first = True
    for dataset, dname in data_and_names:
        # first, print statistics of the data
        majority_labels = defaultdict(lambda: 0)
        complicated_labels = defaultdict(lambda: 0)
        topic_counts = defaultdict(lambda: 0)
        for tweet in dataset:
            majority_labels[tweet.get_labelling(MAJORITY_RULE)] += 1
            complicated_labels[tweet.get_labelling(MORE_COMPLICATED)] += 1
            topic_counts[tweet.topic] += 1

        # feature vector statistics
        tweet = dataset[0]
        feature_vect_sizes = {}
        for feature in ALL_FEATURES:
            feature_vect_sizes[feature] = len(tweet.get_feature_vector(selected_feats={feature}))
        feature_vect_sizes['total'] = sum(feature_vect_sizes.values())

        datas = [
            (majority_labels, 'Majority Rule Labelling Counts'),
            (complicated_labels, 'More Complicated Labelling Counts'),
            (topic_counts, 'Topic Counts'),
            (feature_vect_sizes, 'Length of Feature Vectors')
        ]
        if not first:
            datas = datas[:-1]

        print('\n\n--DATASET STATISTICS for **{}**--'.format(dname))
        for d, name in datas:
            print('\n{}:'.format(name))
            for key, value in d.items():
                print('  {} - {}'.format(key, value))
        print('\n')

        # now get results over the different levels of agreement
        for percent_agreement in [0, 0.5, 0.79, 0.99]:
            filtered_tweets = []
            for tweet in dataset:
                labelling = {}
                for label in LABELS:
                    labelling[label] = int(tweet.labelling[label])
                total = sum(labelling.values())
                if max(labelling.values()) / float(total) > percent_agreement:
                    filtered_tweets.append(tweet)
            print('Proportion of data with {} percent agreement:'.format(percent_agreement))
            print('  {} out of {}; {} percent of the data.'.format(
                len(filtered_tweets), len(dataset), float(len(filtered_tweets)) / len(dataset)
            ))

        first = False

# loading functions
def load_tweets_from_csv(fname, serialize=True, use_all_for_ngrams=False):
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

        if sum(tweet_stats.values()) < 5:
            print('shit tweet...')
            continue

        # extract the necessary data
        tweet = Tweet(first_tweet[TWEET_ID], first_tweet['text'], first_tweet['topic'])
        tweet.labelling = tweet_stats
        tweets.append(tweet)

    # always want to preprocess
    print('Preprocessing data...')
    preprocess_tweets(tweets, verbose=False)

    # save data if desired
    if serialize:
        if use_all_for_ngrams:
            train_tweets, test_tweets = tweets, []
        else:
            train_tweets, test_tweets = train_test_split(tweets, test_size=0.2, shuffle=True, random_state=1917)

        print('Extracting features...')
        extract_features(train_tweets, test_tweets)

        if use_all_for_ngrams:
            np.save('../labelled_data/all_tweets_with_labels.npy', np.array(train_tweets))
        else:
            np.save('../labelled_data/train_tweets_with_labels_full.npy', np.array(train_tweets))
            np.save('../labelled_data/test_tweets_with_labels_full.npy', np.array(test_tweets))

    return tweets

def load_train_test(get_all=False):
    if get_all:
        return list(np.load('../labelled_data/all_tweets_with_labels.npy')), []
    else:
        train_tweets = np.load('../labelled_data/train_tweets_with_labels_full.npy')
        test_tweets = np.load('../labelled_data/test_tweets_with_labels_full.npy')
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

def basic_testing(train, test, model, label_setting):
    train_X = get_X(train, features=BEST_FEATURES)
    train_y = get_y(train, label_setting=label_setting)
    test_X = get_X(test, features=BEST_FEATURES)
    test_y = get_y(test, label_setting=label_setting)

    model.fit(train_X, train_y)
    preds = model.predict(test_X)
    print(get_results(test_y, preds))

    print('\nMore results...')
    train_y_consensus = get_y(train, label_setting=MORE_COMPLICATED)
    train_y_majority = get_y(train, label_setting=MAJORITY_RULE)
    test_y = get_y(test, label_setting=MAJORITY_RULE)

    for m in (LogisticRegression(), DummyClassifier(),):
        print(f'Results with model {m} - TRAINED ON CONSENSUS:')
        m.fit(train_X, train_y_consensus)
        preds = m.predict(test_X)
        print(classification_report(test_y, preds))
        print(confusion_matrix(test_y, preds))

        print(f'\nResults with model {m} - TRAINED ON MAJORITY RULE:')
        m.fit(train_X, train_y_majority)
        preds = m.predict(test_X)
        print(classification_report(test_y, preds))
        print(confusion_matrix(test_y, preds))

        print('\n\n')

    complicated_tests(train, test)

def get_model_acc(X_train, y_train, X_test, y_test, model, verbose=False,):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    if verbose:
        print(classification_report(y_test, preds))
        print(dict(Counter(list(preds))))
    return f1_score(y_test, preds, average='weighted')

def agreement_level_experiments(all_tweets, label_setting, model=LogisticRegression(), do_upper_bound=False):
    big_dataset_analysis(all_tweets, [], just_all=True)
    print('Testing model {} with label setting {}...'.format(model, label_setting))

    K = 5
    kf = KFold(n_splits=K, shuffle=True, random_state=1917)
    agreement_threshs = [0.51, 0.8, 1]

    k = 0
    thresh_to_fold_results = defaultdict(lambda: [])
    for train_idx, test_idx in kf.split(all_tweets):
        print('Current fold is {}...'.format(k))
        k += 1

        train_tweets, test_tweets = all_tweets[train_idx], all_tweets[test_idx]

        # unfiltered
        X_train = get_X(train_tweets, features=BEST_FEATURES)
        y_train = get_y(train_tweets, label_setting)
        X_test = get_X(test_tweets, features=BEST_FEATURES)
        y_test = get_y(test_tweets, label_setting)

        # shit data
        shit_tweets = filter_for_agreement(test_tweets, None, get_shit=True)
        X_shit_test = get_X(shit_tweets, features=BEST_FEATURES)
        y_shit_test = get_y(shit_tweets, label_setting)

        # final exp CV results
        keyall = 'trainALL-testALL'
        keyshit = 'trainALl-testSHIT'
        print('TRAINING ON ALL')
        print('testing on all')
        thresh_to_fold_results[keyall].append(get_model_acc(X_train, y_train, X_test, y_test, model, verbose=True))
        print('\ntesting on shit')
        thresh_to_fold_results[keyshit].append(get_model_acc(X_train, y_train, X_shit_test, y_shit_test, model, verbose=True))

        print('\n\n\n\n\n')
        # print(thresh_to_fold_results)

        for i, thresh in enumerate(agreement_threshs):
            # filtering train by agreement
            try:
                upper_bound = agreement_threshs[i+1]
            except IndexError:
                upper_bound = 1.01
            if not do_upper_bound:
                upper_bound = 1.01

            filtered_train = filter_for_agreement(train_tweets, thresh, upper=upper_bound)
            X_flt_train = get_X(filtered_train, features=BEST_FEATURES)
            y_flt_train = get_y(filtered_train, label_setting)

            # filtering test by agreement
            filtered_test = filter_for_agreement(test_tweets, thresh, upper=upper_bound)
            X_flt_test = get_X(filtered_test, features=BEST_FEATURES)
            y_flt_test = get_y(filtered_test, label_setting)

            # three different experiment settings per threshold
            # - train on all, test on filtered
            # - train on filtered, test on all
            # - train on filtered, test on filtered
            key1 = 'trainALL-testFILTERED-thresh{}'.format(thresh)
            key2 = 'trainFILTERED-testALL-thresh{}'.format(thresh)
            key3 = 'trainFILTERED-testFILTERED-thresh{}'.format(thresh)
            key4 = 'trainFILTERED-testSHIT-thresh{}'.format(thresh)

            experiments = [
                (key1, X_train, y_train, X_flt_test, y_flt_test),
                (key2, X_flt_train, y_flt_train, X_test, y_test),
                (key3, X_flt_train, y_flt_train, X_flt_test, y_flt_test),
                (key4, X_flt_train, y_flt_train, X_shit_test, y_shit_test),
            ]

            for key, xtr, ytr, xte, yte in experiments:
                thresh_to_fold_results[key].append(get_model_acc(xtr, ytr, xte, yte, model))

        with open('../agreement_level_results_labels{}_dummy_bounded.tsv'.format(label_setting), 'w') as tsvf:
            field_names = ['test_setting', 'mean_score'] + ['fold_{}'.format(i) for i in range(K)]
            writer = DictWriter(tsvf, fieldnames=field_names)
            writer.writeheader()

            for test_setting, fold_results in thresh_to_fold_results.items():
                d = {'test_setting': test_setting}
                d.update({'fold_{}'.format(i): fold_results[i] for i in range(len(fold_results))})
                d['mean_score'] = np.mean(fold_results)
                writer.writerow(d)

def complicated_tests(train, test):
    X_train = get_X(train, features=BEST_FEATURES)
    y_train = get_y(train, label_setting=MAJORITY_RULE)
    X_test = get_X(test, features=BEST_FEATURES)
    y_test = get_y(test, label_setting=MAJORITY_RULE)

    # convert to com non-com
    y_train[y_train != 'com'] = 'ncom'
    y_test[y_test != 'com'] = 'ncom'

    # balance data
    for model in (LogisticRegression(), DummyClassifier(),):
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        print(f'\n\nCOMPLICATED TESTS WITH {model}')
        print(classification_report(y_test, preds))
        print(confusion_matrix(y_test, preds))


# results analyzing
def analyze_results(all_tweets, fname):
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

    # feature vector statistics
    tweet = all_tweets[0]
    feature_vect_sizes = {}
    for feature in ALL_FEATURES:
        feature_vect_sizes[feature] = len(tweet.get_feature_vector(selected_feats={feature}))
    feature_vect_sizes['total'] = sum(feature_vect_sizes.values())

    datas = [
        (gold_label_counts, 'Gold Label Counts'),
        (pred_label_counts, 'Predicted Label Counts'),
        (topic_counts, 'Topic Counts'),
        (feature_vect_sizes, 'Length of Feature Vectors')
    ]

    print('\n\n--DATASET STATISTICS--')
    for d, name in datas:
        print('\n{}:'.format(name))
        for key, value in d.items():
            print('  {} - {}'.format(key, value))
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
def initialize(data_file, load_glove=False, use_all_ngrams=False):
    # this is for initializing the data and getting word embeddings
    if load_glove:
        tweets = load_tweets_from_csv('../../data/{}'.format(data_file), serialize=False)
        vocab = []
        for tweet in tweets:
            vocab += tweet.corrected_tokens
        raw_load_and_extract_glove(vocab, EMB_SIZE, '../labelled_data/')

    # feature extraction and serialization
    load_tweets_from_csv('../../data/{}'.format(data_file), serialize=True, use_all_for_ngrams=use_all_ngrams)


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
        '-t', '--testing', action='store_true',
        help='do basic testing of models on train and test sets'
    )
    parser.add_argument(
        '-g', '--agreement', action='store_true',
        help='5-fold CV agreement level testing'
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
    parser.add_argument(
        '-A', '--data_analysis', action='store_true',
        help='analyze just the dataset'
    )
    args = parser.parse_args()

    # get the data
    if not args.initialize:
        print('Loading data...')
        train, test = load_train_test(get_all=not args.testing)

    # get label setting and model from argv
    models = {
        LOGISTIC_REGRESSION: LogisticRegression(),
        HIERARCHICAL: HierClassifier(),
        LINEAR_SVM: LinearSVC(),
        RANDOM_FOREST: RandomForestClassifier(n_estimators=10),
        GUESSER: DummyClassifier(),
    }

    label_setting = args.label_setting
    model = models[args.model]

    if args.analyze:
        print('Analyzing results for model {} with label setting {}...'.format(args.model, label_setting))
        analyze_results(train + test, get_loo_results_fname(args.model, label_setting))

    elif args.ablation:
        feature_ablation(train, test, model, label_setting)

    elif args.initialize:
        initialize(ALL_DATA_FILE, load_glove=True, use_all_ngrams=True)

    elif args.data_analysis:
        big_dataset_analysis(train, test)

    elif args.agreement:
        agreement_level_experiments(np.array(train), label_setting, model=model, do_upper_bound=True)

    elif args.testing:
        results = ''
        for label_setting in (MAJORITY_RULE, MORE_COMPLICATED,):
            for mname, model in models.items():
                print('Fitting model: {}...'.format(mname))
                results += '\n\n------------------------------'
                results += 'Label setting: {}. Model: {}.'.format(label_setting, mname)
                results += basic_testing(train, test, model, label_setting)
        with open('../results/prelim_full_data_results.txt', 'w') as f:
            f.write(results)
    else:
        print('Running leave-one-out cross validation with model {} and labels {}...'.format(args.model, label_setting))
        run_leave_one_out(train + test, label_setting, model, args.model)

