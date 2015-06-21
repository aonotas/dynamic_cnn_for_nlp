#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from gensim.models import word2vec
import logging
import glove
import evaluate
import pickle
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
    model = word2vec.Word2Vec(None, size=emb_dim, window=10, min_count=0, sg=1, negative=0, hs=1)
    # model = word2vec.Word2Vec(None, size=emb_dim, window=10, min_count=0, sg=0, negative=20, hs=0)
    model.build_vocab(sentences)
    # model.syn0 = np.random.uniform(0,0.05,model.syn0.shape)
    # model.syn0 = np.random.rand(emb_dim) - 0.5 / emb_dim
    # syn0 = np.copy(model.syn0)
    # np.random.uniform(0, 0.05, emb_dim)
    # for i in xrange(emb_dim):
        # syn0[i] = np.random.uniform(0, 0.05, emb_dim)
        # syn0[i] = np.random.rand(emb_dim) + 0.5 / emb_dim
    # model.syn0 = np.copy(syn0)
    model.train(sentences)

    embeddings = []
    for index,word in enumerate(index2word):
        vec = get_vec(model, word)
        embeddings.append(vec)
    embeddings = np.asarray(embeddings)
    return embeddings, model



def use_glove(sentences, emb_dim=50):
    glove.logger.setLevel(logging.INFO)
    vocab = glove.build_vocab(sentences)
    cooccur = glove.build_cooccur(vocab, sentences, window_size=10)
    id2word = evaluate.make_id2word(vocab)


    def evaluate_word(W):
        words = ['good', 'movie', 'bad', 'worth', 'dog']
        for word in words:
            print evaluate.most_similar(W, vocab, id2word, word)


    def save_per(W,i):
        if i % 100 == 0 and i >= 100:
            filename = "log/glove_%d_iter%d.model" % (emb_dim, i)
            W = evaluate.merge_main_context(W)
            glove.save_model(W, filename)
            evaluate_word(W)

    W = glove.train_glove(vocab, cooccur, vector_size=emb_dim, iterations=3000, iter_callback=save_per)

    # Merge and normalize word vectors
    # W = evaluate.merge_main_context(W)
    # glove.save_model(W, "glove_25.model")

    # W = glove.train_glove(vocab, cooccur, vector_size=10, iterations=500)
    # # Merge and normalize word vectors
    # W = evaluate.merge_main_context(W)
    # glove.save_model(W, "glove_500.model")



    # W = glove.load_model('glove_25.model')
    
    # print ""
    # W = glove.load_model('glove_500.model')
    # e()




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
    use_glove(sentences, emb_dim=emb_dim)
    # embeddings, model = use_word2vec(sentences=sentences, index2word=index2word, emb_dim=emb_dim)
    # print embeddings


    # print model.most_similar('movie')
    # print ""
    # print model.most_similar('good')
    # print ""
    # print model.most_similar('cat')
    # print model.most_similar('2002')

