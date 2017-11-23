import wordninja
import preprocessor
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import TweetTokenizer
from autocorrect import spell
from string import punctuation
from quicksand.quicksandpy.tweet import Tweet

"""
Global objects.
"""
LEMMATIZER = WordNetLemmatizer()
TOKENIZER = TweetTokenizer(preserve_case=False, reduce_len=True)
PUNCTUATION = set(punctuation)


"""
Utility functions.
"""
def should_be_spell_corrected(string):
    return len(set(string).intersection(PUNCTUATION)) == 0


"""
Preprocessing functions in the pipeline order.
"""
def tokenize(text):
    return TOKENIZER.tokenize(text)

def split_hashtags(tokens):
    """
    Applies hashtag splitting on list of tokens into most likely words.
    E.g., '#trumpisbad' -> ['#', 'trump', 'is', 'bad']
    :param tokens: list of strings
    :return: list of strings
    """
    new_toks = []
    for token in tokens:
        if token.startswith('#'):
            splits = wordninja.split(token[1:])
            new_toks.append('#')
            for w in splits:
                new_toks.append(w)
        else:
            new_toks.append(token)
    return new_toks

def autocorrect(tokens):
    """
    Applies autocorrect on list of strings (only if they don't have punctuation
    in them. E.g., 'dancin' -> 'dancing'.
    :param tokens: list of strings
    :return: list of strings
    """
    corrected = []

    # labelled looks like this: [steve, is, :), happy] -> [steve, is, $EMOTICON$, happy]
    labelled = map(preprocessor.tokenize, tokens)
    for token, label in zip(tokens, labelled):
        if should_be_spell_corrected(label):
            corrected.append(spell(token))
        else:
            corrected.append(token)
    return corrected

def lemmatize(tokens):
    return list(map(LEMMATIZER.lemmatize, tokens))


"""
Primary application functions.
"""
def preprocess_tweets(tweets):
    # Pipeline is a list of tuples of functions and optional arguments.
    pipeline = [
        (tokenize, {}),
        (split_hashtags, {}),
        (autocorrect, {}),
        (lemmatize, {})
    ]
    for tweet in tweets:
        apply_pipeline(pipeline, tweet)

def apply_pipeline(pipeline, tweet, verbose=True):
    output = tweet.orig_text
    if verbose:
        print(f'Current tweet text: {output}')

    for fun, args in pipeline:
        orig = output
        output = fun(output, *args)

        if verbose:
            print(f'Applying function \"{fun.__name__}\" with args {args} onto:')
            print(' ,'.join(orig))
            print(f'Now have: {" ,".join(output)}\n')

    tweet.tokens = output
