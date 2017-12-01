import numpy as np

### Globals ###
LABELS = ('pos', 'neg', 'com', 'obj')
COMPLICATED = 'com'
CSV_LABELS = ('positive', 'negative', 'complicated')
HAS_SENTIMENT_KEY = 'does_the_author_express_sentiment_in_this_tweet_'
POS_NEG_COM_KEY = 'is_the_sentiment_expressed_positive_or_negative_'
IS_GOLD_KEY = '_golden'
ID_KEY = '_unit_id'
TWEET_ID = 'tweet_id'
ALL_DATA_FILE = 'f1211086.csv'
SMALL_DATA_FILE = 'f1209851.csv'

# Labelling options
MAJORITY_RULE = 'majority'
MORE_COMPLICATED = 'complicated'
SOFTMAX = 'softmax'

# model options
LOGISTIC_REGRESSION = 'logreg'
HIERARCHICAL = 'hier_logreg'
LINEAR_SVM = 'linearsvm'
RANDOM_FOREST = 'forest'

# feature sets
UNIGRAMS = 'unigrams'
BIGRAMS = 'bigrams'
WEMB = 'wemb'
SENTIWN = 'swn'
SENTIWN_WEMBS = 'swn_wemb'
ALL_FEATURES = {UNIGRAMS, BIGRAMS, WEMB, SENTIWN, SENTIWN_WEMBS}
BEST_FEATURES = {UNIGRAMS, BIGRAMS, WEMB, SENTIWN}

# utility
GLOVE_DATA_PATH = '/mnt/data/glove/'
EMB_SIZE = 200

def raw_load_and_extract_glove(vocab_set, dimensions, extract_dir, serialize_embeddings=True):
    vocab = set(vocab_set)
    embeddings = {}

    print('Getting glove data {} dimensions, {} size of vocab... '
          'May take some time...'.format(dimensions, len(vocab)))

    glove_path = GLOVE_DATA_PATH + 'glove.twitter.27B.{}d.txt'.format(dimensions)

    with open(glove_path, 'r') as f:
        for line in f.readlines():
            data = line.split()
            word = data[0]
            if word in vocab:
                embeddings[word] = np.array(list(map(float, data[1:])))
                vocab.remove(word)
    print('There are {} words without glove embeddings. E.g.,'.format(len(vocab)))
    print('\n'.join(list(vocab)[:20]))

    # save the embeddings with numpy for quick access
    if serialize_embeddings:
        np.save(get_glove_fname(extract_dir, dimensions), np.array([embeddings]))

    return embeddings

def get_glove_fname(path, dim):
    return '{}vocab_embs{}'.format(path, dim)

def get_loo_results_fname(model_name, label_setting, path='../results/loo_tests/'):
    return '{}loo-results_model-{}_label-{}'.format(path, model_name, label_setting)

def get_glove_data():
    print('Loading word embeddings...')
    return np.load(get_glove_fname('../labelled_data/', 200) + '.npy')[0]
