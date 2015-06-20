#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from gensim.models import word2vec
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def learn_word2vec(sentences, emb_dim=50):
    model = word2vec.Word2Vec(sentences, size=emb_dim, window=5, min_count=0)
    return model

def make_word2vec_model(emb_dim=50):
    filename = 'pretrained_word2vec_%d.model.bin' % emb_dim
    return filename

def save_word2vec_model(model, emb_dim=50):
    filename = make_word2vec_model(emb_dim=emb_dim)
    model.save_word2vec_format(filename, binary=True)

def load_word2vec_model(emb_dim):
    filename = make_word2vec_model(emb_dim=emb_dim)
    model = word2vec.Word2Vec.load_word2vec_format(filename, binary=True)
    return model

def get_vec(model, word):
    if model.vocab.get(word, None):
        index = model.vocab.get(word, None).index
        vec = model.syn0[index]
    else:
        # out of vocabはzero
        vec = np.zeros(shape=(model.syn0.shape[1],))
    return vec


def use_word2vec(sentences, index2word, emb_dim=50):
    model = learn_word2vec(sentences, emb_dim=emb_dim)
    embeddings = []
    for index,word in enumerate(index2word):
        vec = get_vec(model, word)
        embeddings.append(vec)
    embeddings = np.asarray(embeddings)
    return embeddings, model

if __name__ == '__main__':
    import stanfordSentimentTreebank as sst

    skip_unknown_words = True
    vocab, index2word, datasets, datasets_all_sentences, funcs = sst.load_stanfordSentimentTreebank_dataset(normalize=True, skip_unknown_words=skip_unknown_words)
    train_set, test_set, dev_set  = datasets
    train_set_sentences, test_set_sentences, dev_set_sentences = datasets_all_sentences
    get,sentence2ids, ids2sentence = funcs # 関数を読み込み

    scores, sentences = zip(*train_set_sentences)
    sentences = [[word for word in sentence.lower().split()] for sentence in sentences]


    emb_dim = 50
    embeddings, model = use_word2vec(sentences=sentences, index2word=index2word, emb_dim=emb_dim)
    print embeddings
    # model = learn_word2vec(sentences, emb_dim=emb_dim)

    print model.vocab.get('movie').index
    print index2word.index('movie')
    # print model.most_similar('movie')
    # print model.most_similar('good')
    # print model.most_similar('positives')
    # print model.most_similar('2002')

