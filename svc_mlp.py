import gensim
from gensim.models.doc2vec import LabeledSentence, TaggedDocument
from gensim.models import Doc2Vec
import pandas as pd
import nltk
# nltk.download('punkt')
import numpy as np
import re
import collections
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV

from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.linear_model import LogisticRegression

def count_vec(docs_train, docs_test, tfidf=False):
    doc_str = [' '.join(doc) for doc in docs_train + docs_test]
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(doc_str)
    tfidf_transformer = TfidfTransformer()
    doc_array = X.toarray()
    x_train = doc_array[:len(docs_train)]
    x_test = doc_array[len(docs_train):]
    if tfidf:
        X_train_tfidf = tfidf_transformer.fit_transform(x_train)
        X_test_tfidf = tfidf_transformer.transform(x_test)
        return X_train_tfidf, X_test_tfidf
    return x_train, x_test


def tokenizer(oneline):
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
    word_l = nltk.word_tokenize(oneline)
    return word_l

def clean_tag_doc(df, min_words=5):
    print("Number of doc before cleaning: {}".format(len(df)))
    tagged_docs = []
    labels = []
    clean_docs = []
    n_total_w = 0
    max_n = 0
    count = 0
    for i in range(len(df.index)):
        text = df.loc[i, 'text']
        cleantext = tokenizer(text)
        n_words = len(cleantext)
        label = df.loc[i, 'label']
        new_label = label_trans(label)
        if n_words > min_words and new_label >= 0:
            td = TaggedDocument(cleantext, tags=[str(count)])
            tagged_docs.append(td)
            clean_docs.append(cleantext)
            n_total_w = n_total_w + n_words
            if n_words > max_n:
                max_n = n_words
            labels.append(new_label)
            count = count + 1

    print("Number of doc data after cleaning: {}".format(len(tagged_docs)))
    avg_nw = int(n_total_w / len(tagged_docs))
    print("Average number of words in each doc: {}".format(avg_nw))
    print("Max number of words in all doc: {}".format(max_n))
    print("Class distribution: {}".format(collections.Counter(labels)))

    return clean_docs, tagged_docs, labels

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

def emb_data(model, docs_test, labels_train, labels_test):
    vec_size = model.vector_size
    x_train = np.zeros((len(labels_train), vec_size))
    y_train = np.zeros(len(labels_train),dtype=int)

    x_test = np.zeros((len(labels_test), vec_size))
    y_test = np.zeros(len(labels_test),dtype=int)

    for i, label in enumerate(labels_train):
        x_train[i]=model.docvecs[str(i)]
        y_train[i] = label_trans(label)

    for i, label in enumerate(labels_test):
        x_test[i] = model.infer_vector(docs_test[i])
        y_test[i] = label_trans(label)

    return x_train, y_train, x_test, y_test

def pre_eval(text_clf,x_test,y_test):
    predicted = text_clf.predict(x_test)
    test_acc = np.mean(predicted == y_test)
    print("Classification accuracy is "+ str(test_acc)+'.\n')
    print(metrics.classification_report(y_test, predicted))
    np.set_printoptions(linewidth=85)
    print(metrics.confusion_matrix(y_test, predicted))

if __name__ == "__main__":
    trainpath = './data/Reddit_train.csv'
    testpath = './data/Reddit_test.csv'
    df_train = pd.read_csv(trainpath)
    df_test = pd.read_csv(testpath)

    print ('training data: ')
    docs_train, tag_train, labels_train = clean_tag_doc(df_train, min_words=5)
    print('test data: ')
    docs_test, tag_test, labels_test = clean_tag_doc(df_test, min_words=0)

    #x_train, x_test = count_vec(docs_train, docs_test, tfidf=False)
    x_train, x_test = count_vec(docs_train, docs_test, tfidf=True)
    y_train = np.array(labels_train)
    y_test = np.array(labels_test)

    ros = RandomOverSampler(random_state=0)
    #X_resampled, y_resampled = ros.fit_resample(x_train, y_train)
    X_resampled, y_resampled = SMOTE().fit_resample(x_train, y_train)


    svc = LinearSVC(random_state=0, tol=1e-5)
    mlp = MLPClassifier(alpha=1e-5, solver='adam', random_state=1, hidden_layer_sizes=(100,))


    clf = svc
    clf.fit(X_resampled, y_resampled)
    pre_eval(clf,x_test,y_test)


