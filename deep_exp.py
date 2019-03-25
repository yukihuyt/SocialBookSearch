import gensim
from gensim.models.doc2vec import TaggedDocument
from gensim.models import Doc2Vec
import pandas as pd
import nltk
# nltk.download('punkt')
import numpy as np
import pickle
import re
from sklearn import preprocessing
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import nnmodel as nm

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def tokenizer(oneline, sent = False, doc = False):
    oneline = re.sub(r"\[NEWLINE\]", " ", oneline)
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
    if sent:
        sent_l = nltk.sent_tokenize(oneline)
        one_doc = [nltk.word_tokenize(x) for x in sent_l]
    elif doc:
        one_doc = ' '.join(nltk.word_tokenize(oneline))
    else:
        one_doc = nltk.word_tokenize(oneline)
    return one_doc

def clean_tag_doc(df, min_words=0, sent=False, doc=False):
    print("Number of doc before cleaning: {}".format(len(df)))
    tagged_docs = []
    labels = []
    clean_docs = []
    n_total_w = 0
    count = 0
    for i in range(len(df.index)):
        text = df.loc[i, 'text']
        label = df.loc[i, 'label']
        new_label = label_trans(label)
        cleantext = tokenizer(text, sent=sent, doc=doc)
        n_words = len(cleantext)

        if n_words > min_words and new_label >= 0:
            td = TaggedDocument(cleantext, tags=[str(count)])
            tagged_docs.append(td)
            clean_docs.append(cleantext)
            n_total_w = n_total_w + n_words
            labels.append(new_label)
            count = count + 1

    print("Number of doc data after cleaning: {}".format(len(tagged_docs)))
    avg_nw = int(n_total_w / len(tagged_docs))
    print("Average number of words in each doc: {}".format(avg_nw))

    return clean_docs, tagged_docs, labels

def train_save_d2v(tg_list, modelpath, vec_size=300):
    model=Doc2Vec(tg_list,min_count=1,size=vec_size,dm=0, sample=1e-5,
                  window=15, negative=5, workers=7)
    model.train(tg_list, total_examples = model.corpus_count, epochs=10)
    model.save(modelpath)
    return model

def label_trans(label):
    if label == 'yes':
        new_label = 1
    elif label == 'no':
        new_label = 0
    elif label == 'na':
        new_label = -1
    else:
        new_label = int(label)
    return new_label

def pad_y(labels_train, labels_test):
    y_train = pd.Series(labels_train)
    y_test = pd.Series(labels_test)
    y_labels = list(y_train.value_counts().index)
    le = preprocessing.LabelEncoder()
    le.fit(y_labels)
    num_labels = len(y_labels)
    y_train = to_categorical(y_train.map(lambda x: le.transform([x])[0]), num_labels)
    y_test = to_categorical(y_test.map(lambda x: le.transform([x])[0]), num_labels)

    return y_train, y_test, num_labels

def doc_trans(docs_train, docs_test, pad_length, max_vocab=None, save=False, load=False, savepath=None, loadpath=None, onehot=False):
    if load and loadpath:
        with open(loadpath, 'rb') as handle:
            tokenizer = pickle.load(handle)
    else:
        tokenizer = Tokenizer(num_words=max_vocab)
        tokenizer.fit_on_texts(docs_train)

    vocab = tokenizer.word_index
    X_train_word_ids = tokenizer.texts_to_sequences(docs_train)
    X_test_word_ids = tokenizer.texts_to_sequences(docs_test)

    if onehot:
        x_train = tokenizer.sequences_to_matrix(X_train_word_ids, mode='binary')
        x_test = tokenizer.sequences_to_matrix(X_test_word_ids, mode='binary')
    else:
        x_train = pad_sequences(X_train_word_ids, maxlen=pad_length)
        x_test = pad_sequences(X_test_word_ids, maxlen=pad_length)

    if save and savepath:
        with open(savepath, 'wb') as handle:
            pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print('save tokenizer to disk')

    return x_train, x_test, vocab

def emb_matrix(model):
    word2idx = {"_PAD": 0}
    vocab_list = [(k, model.wv[k]) for k, v in model.wv.vocab.items()]
    embeddings_matrix = np.zeros((len(model.wv.vocab.items()) + 1, model.vector_size))
    print('Found %s word vectors.' % len(model.wv.vocab.items()))
    for i in range(len(vocab_list)):
        word = vocab_list[i][0]
        word2idx[word] = i + 1
        embeddings_matrix[i + 1] = vocab_list[i][1]
    return embeddings_matrix, vocab_list

if __name__ == "__main__":
    trainpath = './data/LT_train.csv'
    testpath = './data/LT_test.csv'
    df_train = pd.read_csv(trainpath)
    df_test = pd.read_csv(testpath)

    print ('training data: ')
    docs_train, tag_train, labels_train = clean_tag_doc(df_train, min_words=0, doc=True)
    print('test data: ')
    docs_test, tag_test, labels_test = clean_tag_doc(df_test, min_words=0, doc=True)

    modelpath = './models/sbsLT_d2v.d2v'
    #sbsd2v = train_save_d2v(tag_train,modelpath,vec_size=100)
    sbsd2v = gensim.models.Word2Vec.load(modelpath)

    tokenizer_path = './models/keras_tokenizer.pickle'
    pad_length = 300 #(2 times of average number of words in all doc, not same as d2v doc size)
    x_train, x_test, vocab = doc_trans(docs_train, docs_test, max_vocab=None, pad_length=pad_length,
                                       onehot=False, load=True, loadpath=tokenizer_path)
    y_train, y_test, num_labels = pad_y(labels_train,labels_test)

    embeddings_matrix, vocab_list = emb_matrix(model=sbsd2v)

    #clf = nm.simpleMLP(vocab,num_labels)
    clf = nm.TextCNN(pad_length, embeddings_matrix, vocab=vocab_list, num_labels=num_labels)
    #clf = nm.CNNRNN_words(embeddings_matrix, pad_length, num_labels, vec_size=300)
    nm.fit_eva(clf,x_train, y_train, x_test, y_test, savedir='./models',savename='textcnn_train', early_stop=True)

