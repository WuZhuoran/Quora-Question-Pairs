import re

import distance
import pandas as pd
import numpy as np
import gensim
from fuzzywuzzy import fuzz
from nltk.corpus import stopwords
from nltk import word_tokenize

SAFE_DIV = 0.0001
STOP_WORDS = stopwords.words("english")
model = gensim.models.KeyedVectors.load_word2vec_format('../../input/glove.840B.300d.txt')
norm_model = gensim.models.KeyedVectors.load_word2vec_format('../../input/glove.840B.300d.txt')


def preprocess(x):
    x = str(x).lower()
    x = x.replace(",000,000", "m").replace(",000", "k").replace("′", "'").replace("’", "'") \
        .replace("won't", "will not").replace("cannot", "can not").replace("can't", "can not") \
        .replace("n't", " not").replace("what's", "what is").replace("it's", "it is") \
        .replace("'ve", " have").replace("i'm", "i am").replace("'re", " are") \
        .replace("he's", "he is").replace("she's", "she is").replace("'s", " own") \
        .replace("%", " percent ").replace("₹", " rupee ").replace("$", " dollar ") \
        .replace("€", " euro ").replace("'ll", " will")
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    x = re.sub('[“”\(\'…\)\!\^\"\.;:,\-\?？\{\}\[\]\\/\*@]', ' ', x)
    x = re.sub(r"([0-9]+)000000", r"\1m", x)
    x = re.sub(r"([0-9]+)000", r"\1k", x)
    x = re.sub(r"[^A-Za-z0-9]", " ", x)
    x = re.sub(r"what's", "", x)
    x = re.sub(r"What's", "", x)
    x = re.sub(r"\'s", " ", x)
    x = re.sub(r"\'ve", " have ", x)
    x = re.sub(r"can't", "cannot ", x)
    x = re.sub(r"n't", " not ", x)
    x = re.sub(r"I'm", "I am", x)
    x = re.sub(r" m ", " am ", x)
    x = re.sub(r"\'re", " are ", x)
    x = re.sub(r"\'d", " would ", x)
    x = re.sub(r"\'ll", " will ", x)
    x = re.sub(r"60k", " 60000 ", x)
    x = re.sub(r" e g ", " eg ", x)
    x = re.sub(r" b g ", " bg ", x)
    x = re.sub(r"\0s", "0", x)
    x = re.sub(r" 9 11 ", "911", x)
    x = re.sub(r"e-mail", "email", x)
    x = re.sub(r"\s{2,}", " ", x)
    x = re.sub(r"quikly", "quickly", x)
    x = re.sub(r" usa ", " America ", x)
    x = re.sub(r" USA ", " America ", x)
    x = re.sub(r" u s ", " America ", x)
    x = re.sub(r" uk ", " England ", x)
    x = re.sub(r" UK ", " England ", x)
    x = re.sub(r"india", "India", x)
    x = re.sub(r"switzerland", "Switzerland", x)
    x = re.sub(r"china", "China", x)
    x = re.sub(r"chinese", "Chinese", x)
    x = re.sub(r"imrovement", "improvement", x)
    x = re.sub(r"intially", "initially", x)
    x = re.sub(r"quora", "Quora", x)
    x = re.sub(r" dms ", "direct messages ", x)
    x = re.sub(r"demonitization", "demonetization", x)
    x = re.sub(r"actived", "active", x)
    x = re.sub(r"kms", " kilometers ", x)
    x = re.sub(r"KMs", " kilometers ", x)
    x = re.sub(r" cs ", " computer science ", x)
    x = re.sub(r" upvotes ", " up votes ", x)
    x = re.sub(r" iPhone ", " phone ", x)
    x = re.sub(r"\0rs ", " rs ", x)
    x = re.sub(r"calender", "calendar", x)
    x = re.sub(r"ios", "operating system", x)
    x = re.sub(r"gps", "GPS", x)
    x = re.sub(r"gst", "GST", x)
    x = re.sub(r"programing", "programming", x)
    x = re.sub(r"bestfriend", "best friend", x)
    x = re.sub(r"dna", "DNA", x)
    x = re.sub(r"III", "3", x)
    x = re.sub(r"the US", "America", x)
    x = re.sub(r"Astrology", "astrology", x)
    x = re.sub(r"Method", "method", x)
    x = re.sub(r"Find", "find", x)
    x = re.sub(r"banglore", "Banglore", x)
    x = re.sub(r" J K ", " JK ", x)
    return x


def get_token_features(q1, q2):
    token_features = [0.0] * 10

    q1_tokens = q1.split()
    q2_tokens = q2.split()

    if len(q1_tokens) == 0 or len(q2_tokens) == 0:
        return token_features

    q1_words = set([word for word in q1_tokens if word not in STOP_WORDS])
    q2_words = set([word for word in q2_tokens if word not in STOP_WORDS])

    q1_stops = set([word for word in q1_tokens if word in STOP_WORDS])
    q2_stops = set([word for word in q2_tokens if word in STOP_WORDS])

    common_word_count = len(q1_words.intersection(q2_words))
    common_stop_count = len(q1_stops.intersection(q2_stops))
    common_token_count = len(set(q1_tokens).intersection(set(q2_tokens)))

    token_features[0] = common_word_count / (min(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[1] = common_word_count / (max(len(q1_words), len(q2_words)) + SAFE_DIV)
    token_features[2] = common_stop_count / (min(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[3] = common_stop_count / (max(len(q1_stops), len(q2_stops)) + SAFE_DIV)
    token_features[4] = common_token_count / (min(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[5] = common_token_count / (max(len(q1_tokens), len(q2_tokens)) + SAFE_DIV)
    token_features[6] = int(q1_tokens[-1] == q2_tokens[-1])
    token_features[7] = int(q1_tokens[0] == q2_tokens[0])
    token_features[8] = abs(len(q1_tokens) - len(q2_tokens))
    token_features[9] = (len(q1_tokens) + len(q2_tokens)) / 2
    return token_features


def get_longest_substr_ratio(a, b):
    strs = list(distance.lcsubstrings(a, b))
    if len(strs) == 0:
        return 0
    else:
        return len(strs[0]) / (min(len(a), len(b)) + 1)


def extract_features(df):
    df["question1"] = df["question1"].fillna("").apply(preprocess)
    df["question2"] = df["question2"].fillna("").apply(preprocess)

    print("token features...")
    token_features = df.apply(lambda x: get_token_features(x["question1"], x["question2"]), axis=1)
    df["cwc_min"] = list(map(lambda x: x[0], token_features))
    df["cwc_max"] = list(map(lambda x: x[1], token_features))
    df["csc_min"] = list(map(lambda x: x[2], token_features))
    df["csc_max"] = list(map(lambda x: x[3], token_features))
    df["ctc_min"] = list(map(lambda x: x[4], token_features))
    df["ctc_max"] = list(map(lambda x: x[5], token_features))
    df["last_word_eq"] = list(map(lambda x: x[6], token_features))
    df["first_word_eq"] = list(map(lambda x: x[7], token_features))
    df["abs_len_diff"] = list(map(lambda x: x[8], token_features))
    df["mean_len"] = list(map(lambda x: x[9], token_features))

    print("fuzzy features..")
    df["token_set_ratio"] = df.apply(lambda x: fuzz.token_set_ratio(x["question1"], x["question2"]), axis=1)
    df["token_sort_ratio"] = df.apply(lambda x: fuzz.token_sort_ratio(x["question1"], x["question2"]), axis=1)
    df["fuzz_ratio"] = df.apply(lambda x: fuzz.QRatio(x["question1"], x["question2"]), axis=1)
    df["fuzz_partial_ratio"] = df.apply(lambda x: fuzz.partial_ratio(x["question1"], x["question2"]), axis=1)
    df["longest_substr_ratio"] = df.apply(lambda x: get_longest_substr_ratio(x["question1"], x["question2"]), axis=1)
    return df


def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    stop_words = stopwords.words('english')
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in STOP_WORDS]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())


print("Extracting features for train:")
train_df = pd.read_csv("../../input/train.csv")
train_df = extract_features(train_df)
train_df.drop(["id", "qid1", "qid2", "question1", "question2", "is_duplicate"], axis=1, inplace=True)
train_df.to_csv("../../input/nlp_features_train.csv", index=False)

print("Extracting features for test:")
test_df = pd.read_csv("../../input/test.csv")
test_df = extract_features(test_df)
test_df.drop(["test_id", "question1", "question2"], axis=1, inplace=True)
test_df.to_csv("../../input/nlp_features_test.csv", index=False)
