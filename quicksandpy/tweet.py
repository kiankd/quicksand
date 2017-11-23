"""
Simple file for storing tweet objects.
"""
class Tweet(object):
    def __init__(self, tid, text, topic):
        self.tid = tid
        self.orig_text = text
        self.topic = topic
        self.corrected_tokens = []
        self.uncorrected_tokens = []

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
