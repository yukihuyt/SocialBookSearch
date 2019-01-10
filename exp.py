from gensim import utils
import gensim
from gensim.models.doc2vec import LabeledSentence, TaggedDocument
from gensim.models import Doc2Vec
import pandas as pd
import nltk
import numpy as np
import pickle

import re

def load_csv(datapath):
    df = pd.read_csv(datapath)
    return df

def tokenizer(oneline):
    # string = unicode(string, "utf-8")
    # string = unidecode(string)
    oneline = re.sub(r"\[NEWLINE\]", " ", oneline)  # added for fixed microsoft sql files
    oneline = re.sub(r"[^A-Za-z0-9(),!?\'`]", " ", oneline)
    oneline = re.sub(r"\'s", " \'s", oneline)
    oneline = re.sub(r"\'ve", " \'ve", oneline)
    oneline = re.sub(r"n\'t", " n\'t", oneline)
    oneline = re.sub(r"\'re", " \'re", oneline)
    oneline = re.sub(r"\'d", " \'d", oneline)
    oneline = re.sub(r"\'ll", " \'ll", oneline)
    oneline = re.sub(r",", " ", oneline)
    oneline = re.sub(r"!", " ", oneline)
    oneline = re.sub(r"\(", " ", oneline)
    oneline = re.sub(r"\)", " ", oneline)
    oneline = re.sub(r"\?", " ", oneline)
    oneline = re.sub(r"\s{2,}", " ", oneline)
    oneline = re.sub(r" +", " ", oneline)
    oneline = oneline.strip().lower()
    new_str = nltk.word_tokenize(oneline)
    return new_str

def train_save_d2v(tg_list, modelpath, vec_size=300):
    model=Doc2Vec(tg_list,min_count=1,size=vec_size,dm=0, sample=1e-5,
                  window=15, negative=5, workers=7)
    model.train(tg_list, total_examples = model.corpus_count, epochs=10)
    model.save(modelpath)
    return model



if __name__ == "__main__":
    datapath = './data/tryin.csv'
    df = load_csv(datapath)
    clean_doc = []
    tagged_doc = []
    n_total_w = 0
    print("Number of doc before preprocessing: {}".format(len(df)))

    for i in range(len(df.index)):
        text = df.ix[i,'text']
        id = df.ix[i,'id']
        cleantext= tokenizer(text)
        n_words = len(cleantext)
        if n_words > 5:
            clean_doc.append(cleantext)
            td = TaggedDocument(cleantext, tags=[str(i)])
            tagged_doc.append(td)
            n_total_w = n_total_w + n_words

    print ("Number of doc after preprocessing: {}".format(len(tagged_doc)))
    avg_nw = int(n_total_w/len(tagged_doc))
    print("Average number of words in each doc: {}".format(avg_nw))
    print(clean_doc[:20])

    modelpath = './models/sbs_d2v00.d2v'
    # sbsd2v = train_save_d2v(tagged_doc,modelpath)
    sbsd2v = gensim.models.Word2Vec.load(modelpath)
    # sbsd2v.infer_vector(blah)


