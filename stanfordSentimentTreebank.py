#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
    load the dataset 'Deeply Moving: Deep Learning for Sentiment Analysis Dataset'
    http://nlp.stanford.edu/sentiment/
'''
import csv
import numpy as np
import matplotlib.pyplot as plt

def load_splitset(csv_filename):
    dataReader = csv.reader(file(csv_filename),delimiter=',')
    fline = dataReader.next()
    train_ids = []
    test_ids  = []
    dev_ids = []
    for sentence_index, splitset_label in dataReader:
        sentence_index = int(sentence_index)
        splitset_label = int(splitset_label)
        if splitset_label == 1:
            train_ids.append(sentence_index)
        elif splitset_label == 2:
            test_ids.append(sentence_index)
        elif splitset_label == 3:
            dev_ids.append(sentence_index)
    return train_ids,test_ids,dev_ids


def load_datasets(csv_filename):
    dataReader = csv.reader(file(csv_filename),delimiter='\t')
    fline = dataReader.next() # 先頭をスキップ
    datasets = {}
    for sentence_index, sentence in dataReader:
        sentence_index = int(sentence_index)
        datasets[sentence_index] = sentence
    return datasets

def load_tree_datasets(csv_filename):
    scores = []
    for sentence_tree in open(csv_filename):
        # words, score = parse_tree(sentence_tree)
        score = sentence_tree[1]
        scores.append(score)
    return scores



def plot_histogram(datasets):
    #for plot histogram
    d = [len(datasets[i].split(' ')) for i in train_ids]
    plt.hist(d, 50, facecolor='g', alpha=0.75)
    # plt.show()

    d = [len(datasets[i].split(' ')) for i in test_ids]
    plt.hist(d, 50, facecolor='b', alpha=0.75)
    # plt.show()

    d = [len(datasets[i].split(' ')) for i in valid_ids]
    plt.hist(d, 50, facecolor='y', alpha=0.75)
    plt.show()


class Vocab(object):
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other): 
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "<" + ', '.join(vals) + ">"


def build_vocab(ids, datasets, min_count=1, unknown_word_identity=''):
    count_vocab = {}
    for i in ids:
        sentence = datasets[i]
        words = sentence.split(' ') #形態素解析

        for word in words:
            word = word.lower()
            if word in count_vocab:
                count_vocab[word].count += 1
            else:
                count_vocab[word] = Vocab(count=1)
    # print "vocab_size:", len(count_vocab.keys())

    # for unknown word or out of vocab word
    vocab = {unknown_word_identity: Vocab(count=1, index=0)}
    index2word = [unknown_word_identity]
    for word, v in count_vocab.items():
        if v.count >= min_count:
            v.index = len(vocab)
            index2word.append(word)
            vocab[word] = v
    # print "vocab_size:", len(vocab)
    return vocab,index2word


# データセットの読み込み
def load_stanfordSentimentTreebank_dataset(normalize=True, skip_unknown_words=True, datatype=5):
    csv_filename = 'data/movie/stanfordSentimentTreebank/datasetSplit.txt'
    train_ids,test_ids,dev_ids = load_splitset(csv_filename)
    csv_filename = 'data/movie/stanfordSentimentTreebank/datasetSentences.txt'
    datasets = load_datasets(csv_filename)
    vocab,index2word = build_vocab(train_ids, datasets, unknown_word_identity='')

    # 未知語をskipするか，特殊文字として扱うか
    if skip_unknown_words:
        default_word = None
    else:
        default_word = vocab['']
    def n(word):
        if normalize:
            word = word.lower()
        return word

    def get(word):
        word = n(word)
        return vocab.get(word, default_word)
    
    def sentence2ids(sentence):
        sentence_ids = []
        unknown_word_count = 0
        if isinstance(sentence, str):
            sentence = sentence.split(' ')
        for word in sentence:
            v = get(word)
            # 未知語はskip(フラグが立っていれば)
            if skip_unknown_words and v is None:
                unknown_word_count += 1
                continue
            word_index = v.index
            sentence_ids.append(word_index)
        return sentence_ids, unknown_word_count

    def ids2sentence(ids):
        sentence = []
        for i in ids:
            word = get(i)
            sentence.append(word)
        return sentence

    train_scores = [tree_sentence[1] for tree_sentence in open('data/movie/trees/train.txt')]
    test_scores  = [tree_sentence[1] for tree_sentence in open('data/movie/trees/test.txt')]
    dev_scores   = [tree_sentence[1] for tree_sentence in open('data/movie/trees/dev.txt')]

    train_set_sentences = [(int(score), n(datasets[index])) for index, score in zip(train_ids, train_scores)]
    test_set_sentences  = [(int(score), n(datasets[index])) for index, score in zip(test_ids, test_scores)]
    dev_set_sentences   = [(int(score), n(datasets[index])) for index, score in zip(dev_ids, dev_scores)]

    # fine-grained 5 class 
    # Binary 2 class class=2はスキップ
    if datatype == 2:
        def filter_class(i):
            if i in [0,1,3,4]:
                return True
            else:
                return False
        def bc(i):
            if i in [0,1]:
                # negative
                return 0
            elif i in [3,4]:
                # positive
                return 1
        train_set_sentences = [(bc(score),sentence) for score,sentence in train_set_sentences if filter_class(score)]
        test_set_sentences  = [(bc(score),sentence) for score,sentence in test_set_sentences if filter_class(score)]
        dev_set_sentences   = [(bc(score),sentence) for score,sentence in dev_set_sentences if filter_class(score)]



    train_set = [(score, sentence2ids(sentence)) for score,sentence in train_set_sentences]
    test_set  = [(score, sentence2ids(sentence)) for score,sentence in test_set_sentences]
    dev_set   = [(score, sentence2ids(sentence)) for score,sentence in dev_set_sentences]

    datasets_all = (train_set, test_set, dev_set)
    datasets_all_sentences = (train_set_sentences, test_set_sentences, dev_set_sentences)
    funcs = (get,sentence2ids,ids2sentence)
    return vocab, index2word, datasets_all, datasets_all_sentences, funcs



def parse_tree(s):
    wordindexes = []
    phrases = []
    words = []
    scores = []
    index = 0
    count = 0
    tmp = ""
    flag = False
    after_score_or_end_of_kakko = False
    for c in s:
        # print stack
        if c in " " and after_score_or_end_of_kakko:
            # 括弧を含む単語の場合エラーとなる
            after_score_or_end_of_kakko = False
            continue
        elif c in "(":
            index = len(words)
            wordindexes.append(index)
            flag = True
        elif c in ")":
            if tmp:
                words.append(tmp) 
            index = wordindexes.pop()
            phrase = words[index:len(words)]
            score = scores.pop()
            phrases.append( (phrase, score) )
            # print phrase, score
            tmp = ""
            after_score_or_end_of_kakko = True
        elif flag:
            flag = False
            scores.append(c)
            after_score_or_end_of_kakko = True
            continue
        else:
            if tmp == "":
                count = 0
            tmp += c
            count += 1

    # print phrases
    # print words
    # print scores
    # print len(scores)
    # print len(words)
    sentence_score = s[1]
    return words,sentence_score


if __name__ == '__main__':
    csv_filename = 'data/movie/stanfordSentimentTreebank/datasetSplit.txt'
    train_ids,test_ids,dev_ids = load_splitset(csv_filename)
    csv_filename = 'data/movie/stanfordSentimentTreebank/datasetSentences.txt'
    datasets = load_datasets(csv_filename)

    print "train_ids :", len(train_ids)
    print "test_ids  :", len(test_ids)
    print "dev_ids   :", len(dev_ids)
    print "= total   :", len(train_ids) + len(test_ids) + len(dev_ids)

    print "datasets  : ",len(datasets) 



    vocab,index2word = build_vocab(train_ids, datasets)

    print "vocab_size :", len(vocab)
    # A Convolutional Neural Network for Modelling Sentencesではvocabsize=15448



    def get(word):
        return vocab.get(word.lower(), vocab[''])


    # test
    vocab,index2word,get_word = load_stanfordSentimentTreebank_dataset()
    print get_word('cool').count


    s = '''(3 (2 (2 The) (2 Rock)) (4 (3 (2 is) (4 (2 destined) (2 (2 (2 (2 (2 to) (2 (2 be) (2 (2 the) (2 (2 21st) (2 (2 (2 Century) (2 's)) (2 (3 new) (2 (2 ``) (2 Conan)))))))) (2 '')) (2 and)) (3 (2 that) (3 (2 he) (3 (2 's) (3 (2 going) (3 (2 to) (4 (3 (2 make) (3 (3 (2 a) (3 splash)) (2 (2 even) (3 greater)))) (2 (2 than) (2 (2 (2 (2 (1 (2 Arnold) (2 Schwarzenegger)) (2 ,)) (2 (2 Jean-Claud) (2 (2 Van) (2 Damme)))) (2 or)) (2 (2 Steven) (2 Segal))))))))))))) (2 .)))'''


    words, score = parse_tree(s)
    print score, words
    print len('''The Rock is destined to be the 21st Century 's new `` Conan '' and that he 's going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal .'''.split(' '))


    # csv_filename = 'data/movie/trees/train.txt'
    # csv_filename = 'data/movie/trees/test.txt'
    # csv_filename = 'data/movie/trees/dev.txt'
    # scores = load_tree_datasets(csv_filename)




    # x,y = zip(*sorted([(word,v.count) for word,v in vocab.items()], key=lambda x:x[1],reverse=True))
    # x = np.arange(len(x))
    # y = np.array(y)
    # print x.shape,y.shape
    # plt.plot(x,y)
    # plt.hist(y,normed=True)
    # plt.show()
    # plot_histogram(datasets)


