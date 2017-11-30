from nltk.corpus import sentiwordnet as swn
from sklearn.feature_extraction.text import CountVectorizer
from quicksand.quicksandpy.util import *
from copy import deepcopy

# swn utility
def get_scores(synset):
    return np.array([synset.pos_score(), synset.neg_score(), synset.obj_score()])

def get_most_common_sense_scores(synsets):
    syns = synsets[0]
    return get_scores(syns)

def get_avg_sense_scores(synsets):
    scores = [get_scores(syns) for syns in synsets]
    avg = np.mean(np.array(scores), axis=0)
    assert(len(avg) == 3)
    return avg

# primary function
def extract_features(train_tweets, test_tweets):
    try:
        all_tweets = train_tweets + test_tweets
    except ValueError:
        all_tweets = list(train_tweets) + list(test_tweets)

    # ngrams
    unigramer = CountVectorizer(ngram_range=(1,1), min_df=2, max_features=5000)
    bigramer = CountVectorizer(ngram_range=(2,2), min_df=2, max_features=5000)

    # note we are extracting n-grams from the uncorrected tokens!
    train_docs = [' '.join(tweet.uncorrected_tokens) for tweet in train_tweets]
    for extractor, feature_name in [(unigramer, UNIGRAMS), (bigramer, BIGRAMS)]:
        extractor.fit(train_docs)
        for tweet in all_tweets:
            tweet.features[feature_name] = extractor.transform([' '.join(tweet.uncorrected_tokens)])

    # word embeddings!
    wemb_dict = get_glove_data()
    for tweet in all_tweets:
        embs = []
        for word in tweet.corrected_tokens:
            if word in wemb_dict:
                embs.append(wemb_dict[word])

        # make the average
        if len(embs) == 0:
            print('Warning: no valid word embeddings!')
            indicator = np.array([0])
            avg_emb = np.zeros((EMB_SIZE))
        else:
            indicator = np.array([1])
            avg_emb = np.mean(embs, axis=0)

        assert(len(avg_emb) == EMB_SIZE)
        tweet.features[WEMB] = np.concatenate((indicator, avg_emb))

        # senti word net features
        # need to get from most common synset, and then average over all synsets
            # for each of those, we want:
                # most pos, neg, obj words and their valence
                # average score for pos, neg, obj

        swn_feats = []
        swn_wemb_feats = np.array([])
        for syns_fun in (get_most_common_sense_scores, get_avg_sense_scores):
            all_scores = {}
            best_scores = [0.01, 0.01, 0.01] # don't allow it to avg over 0 scores

            # get all and the best scores
            for word in tweet.corrected_tokens:
                synsets = list(swn.senti_synsets(word))
                if len(synsets) == 0:
                    continue

                scores = syns_fun(synsets)
                all_scores[word] = scores # for averaging

                # getting specifics
                for i, score in enumerate(scores):
                    if score >= best_scores[i]:
                        best_scores[i] = score

            if len(all_scores) == 0:
                # this means that there are no swn features for this guy!
                indicator = [0]
                swn_feats += indicator + [0 for _ in range(3 + 3)]
            else:
                # specific sentiwordnet features, best and average
                indicator = [1]
                best_scores = [0 if x==0.01 else x for x in best_scores] # size is 3
                swn_feats += indicator + best_scores + list(np.mean(list(all_scores.values()), axis=0)) # size is 3

            # wemb based features
            for i in range(len(best_scores)):
                emb_mat = None
                for word, scores in all_scores.items():
                    if scores[i] == best_scores[i]:
                        try:
                            if emb_mat is None:
                                emb_mat = deepcopy(wemb_dict[word].reshape(-1, 1))
                            else:
                                emb_mat = np.concatenate((emb_mat, wemb_dict[word].reshape(-1, 1)), axis=1)
                        except KeyError:
                            pass
                if emb_mat is None:
                    avg_emb = np.zeros(EMB_SIZE + 1) # including indicator feature
                else:
                    avg_emb = np.concatenate((np.array([1]), np.mean(emb_mat, axis=1))) # including indicator feature
                swn_wemb_feats = np.concatenate((swn_wemb_feats, avg_emb))

        # after the two functions, then set the final features
        tweet.features[SENTIWN] = np.array(swn_feats)
        tweet.features[SENTIWN_WEMBS] = swn_wemb_feats

    # check
    num_feats = len(all_tweets[0].get_feature_vector())
    print('We have {} features.'.format(num_feats))
    for tweet in all_tweets[1:]:
        assert(len(tweet.get_feature_vector()) == num_feats)
