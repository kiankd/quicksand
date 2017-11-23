"""
Simple file for storing tweet objects.
"""

class Tweet(object):
    def __init__(self, tid, text, topic):
        self.tid = tid
        self.orig_text = text
        self.topic = topic
        self.tokens = []


