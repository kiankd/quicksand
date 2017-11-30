"""
Simple file for storing tweet objects.
"""
import numpy as np
from quicksand.quicksandpy.util import *


class Tweet(object):
    def __init__(self, tid, text, topic):
        self.tid = tid
        self.orig_text = text
        self.topic = topic
        self.labelling = None # this is a dictionary of labels
        self.corrected_tokens = []
        self.uncorrected_tokens = []
        self.features = {}

    def get_labelling(self, option):
        label_counts = [self.labelling[label] for label in LABELS]
        total = sum(self.labelling.values())
        max_count = max(label_counts)
        agreement = max_count / float(total)

        # softmax normalizes the labelling
        if option == SOFTMAX:
            return list(map(lambda x: x / total, label_counts))

        # if the labelling has less than 51% agreement, return complicated.
        elif option == MAJORITY_RULE:
            if agreement > 0.50:
                return max(self.labelling, key=self.labelling.get)
            else:
                return COMPLICATED

        # if the labelling has less than 80% agreement, return complicated.
        elif option == MORE_COMPLICATED:
            if agreement >= 0.80:
                return max(self.labelling, key=self.labelling.get)
            else:
                return COMPLICATED

    def get_feature_vector(self):
        all_feats = np.array([])
        for vector in self.features.values():
            try:
                all_feats = np.concatenate((all_feats, np.array(vector.todense()).flatten()))
            except AttributeError:
                all_feats = np.concatenate((all_feats, vector))
        return all_feats


def load_tweets(fname):
    rows = []
    with open(fname, 'r') as f:
        i = 0
        for line in f.readlines():
            if i > 0:
                splitted = line.split(',')
                l = [splitted[0]] # tid
                text = splitted[1:-3] # fix error in formatting
                l.append((','.join(text)))
                l.append(splitted[-3]) # topic
                rows.append(l)
            i = 1
    tweets = []
    for row in rows:
        tweets.append(Tweet(row[0], row[1], row[2]))
    return tweets
