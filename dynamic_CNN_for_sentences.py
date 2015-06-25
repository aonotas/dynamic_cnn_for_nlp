#!/usr/bin/env python
# -*- coding: utf-8 -*-

__doc__ = """{f}
Usage:
    {f} [--alpha=<alpha>] [--n_epoch=<n_epoch>] [--width1 <v>]  [--width2 <v>] [--feat_map_n_1 <v>] [--feat_map_n_final <v>] [--dropout_rate0 <v>] [--dropout_rate1 <v>] [--dropout_rate2 <v>] [--k_top <v>] [--activation <s>] [--learn <s>] [--skip <v>] [--pretrain <s>] [--datatype <v>] [--shuffle <v>]

Options:
    --n_epoch <n_epoch>      Number of epoch [default: 200].
    --alpha <alpha>          Learning rate [default: 0.001].
    --feat_map_n_1 <v>       Learning rate [default: 3].
    --feat_map_n_final <v>   Learning rate [default: 4].
    --width1 <v>             width [default: 10].
    --width2 <v>             width [default: 7].
    --k_top <v>              k_top [default: 4].
    --dropout_rate0 <v>      dropout rate [default: 0.2].
    --dropout_rate1 <v>      dropout rate [default: 0.4].
    --dropout_rate2 <v>      dropout rate [default: 0.5].
    --activation <s>         activation [default: tanh].
    --learn <s>              activation [default: adam].
    --pretrain <s>           pretrain [default: None]
    --skip <v>               skip_unknown_words [default: 1].
    --shuffle <v>            shuffle data [default: 0].
    --datatype <v>           datatype fine-grained or binary [default: 5].
    -h --help                Show this screen and exit.
""".format(f=__file__)

'''
    TODO:
        pretrained word embeddings
        visualise word embeddings
        visualise filters
'''

from docopt import docopt
from schema import Schema, And, Or, Use
s = Schema({
            '--n_epoch': Use(int),
            '--alpha': Use(float),
            '--width1': Use(int),
            '--width2': Use(int),
            '--feat_map_n_1': Use(int),
            '--feat_map_n_final': Use(int),
            '--feat_map_n_final': Use(int),
            '--k_top': Use(int),
            '--dropout_rate0': Use(float),
            '--dropout_rate1': Use(float),
            '--dropout_rate2': Use(float),
            '--activation': Use(str),
            '--learn': Use(str),
            '--skip': Use(int),
            '--shuffle': Use(int),
            '--pretrain': Use(str),
            '--datatype': Use(int),
})
args = docopt(__doc__)
args = s.validate(args)
print args


import sys
import math
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
theano.config.exception_verbosity='high'

from dcnn_train import WordEmbeddingLayer,DynamicConvFoldingPoolLayer #, ConvFoldingPoolLayer
from logreg import LogisticRegression
import pretrained_embedding
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score


np.random.seed(1234)
rng = np.random.RandomState(1234)
srng = RandomStreams()

def main():

    print "############# Load Datasets ##############"

    import stanfordSentimentTreebank as sst

    skip_unknown_words = bool(args.get("--skip"))
    shuffle_flag = bool(args.get("--shuffle"))
    datatype = args.get("--datatype")
    if datatype == 5:
        # Fine-grained 5-class
        n_class = 5
    elif datatype == 2:
        # Binary 2-class
        n_class = 2

    # print "skip_unknown_words",skip_unknown_words
    vocab, index2word, datasets, datasets_all_sentences, funcs = sst.load_stanfordSentimentTreebank_dataset(normalize=True, skip_unknown_words=skip_unknown_words, datatype=datatype)
    train_set, test_set, dev_set  = datasets
    train_set_sentences, test_set_sentences, dev_set_sentences = datasets_all_sentences
    get,sentence2ids, ids2sentence = funcs # 関数を読み込み
    scores, sentences = zip(*train_set_sentences)
    sentences = [[word for word in sentence.lower().split()] for sentence in sentences]
    vocab_size = len(vocab)

 
    dev_unknown_count  = sum([unknown_word_count for score,(ids,unknown_word_count) in dev_set])
    test_unknown_count = sum([unknown_word_count for score,(ids,unknown_word_count) in test_set])

    train_set = [(score, ids) for score,(ids,unknown_word_count) in train_set]
    test_set  = [(score, ids) for score,(ids,unknown_word_count) in test_set]
    dev_set   = [(score, ids) for score,(ids,unknown_word_count) in dev_set]

    print "train_size : ", len(train_set)
    print "dev_size   : ", len(dev_set)
    print "test_size  : ", len(test_set)
    print "-"*30
    print "vocab_size: ", len(vocab)
    print "dev_unknown_words  : ", dev_unknown_count
    print "test_unknown_words : ", test_unknown_count



    
    print args

    EMB_DIM = 50
    vocab_size = len(vocab)


    feat_map_n_1 = args.get("--feat_map_n_1")
    feat_map_n_final = args.get("--feat_map_n_final")

    height = 1
    width1 = args.get("--width1")
    width2 = args.get("--width2")
    k_top  = args.get("--k_top")
    n_class = n_class
    alpha   = args.get("--alpha")
    n_epoch = args.get("--n_epoch")
    dropout_rate0 = args.get("--dropout_rate0")
    dropout_rate1 = args.get("--dropout_rate1")
    dropout_rate2 = args.get("--dropout_rate2")
    activation = args.get("--activation")
    learn      = args.get("--learn")
    number_of_convolutinal_layer = 2


    pretrain = args.get('--pretrain')
    if pretrain == 'word2vec':
        print "*Using word2vec"
        embeddings_W, model = pretrained_embedding.use_word2vec(sentences=sentences, index2word=index2word, emb_dim=EMB_DIM)
        # -0.5 ~ 0.5で初期化している
    else:
        embeddings_W = np.asarray(
            rng.normal(0, 0.05, size = (vocab_size, EMB_DIM)), 
            dtype = theano.config.floatX
        )
        embeddings_W[0,:] = 0

    print np.amax(embeddings_W)
    print np.amin(embeddings_W)
    # print "*embeddings"
    print embeddings_W
    # print bool(embeddings)

    # input_x = [1, 3, 4, 5, 0, 22, 4, 5]

    print "############# Model Setting ##############"    
    x = T.imatrix('x')
    length_x = T.iscalar('length_x')
    y = T.ivector('y') # the sentence sentiment label
    embeddings = WordEmbeddingLayer(rng=rng, 
                            input=x,
                            vocab_size=vocab_size, embed_dm=EMB_DIM, embeddings=embeddings_W)


    def dropout(X, p=0.5):
        if p > 0:
            retain_prob = 1 - p
            X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            # X /= retain_prob
        return X
    # number_of_convolutinal_layer = theano.shared(number_of_convolutinal_layer)
    # dynamic_func = theano.function(inputs=[length_x], outputs=number_of_convolutinal_layer * length_x)

    # dynamic_func_test = theano.function(
    #     inputs = [length_x],
    #     outputs = dynamic_func(length_x),
    #     )
    # print dynamic_func(len([1,2,3]))

    l1 = DynamicConvFoldingPoolLayer(rng, 
                              input = dropout(embeddings.output, p=dropout_rate0), 
                              filter_shape = (feat_map_n_1, 1, height, width1),  # two feature map, height: 1, width: 2, 
                              k_top = k_top,
                              number_of_convolutinal_layer=number_of_convolutinal_layer,
                              index_of_convolitonal_layer=1,
                              length_x=length_x,
                              activation = activation
    )
    l1_no_dropout = DynamicConvFoldingPoolLayer(rng, 
                              input = embeddings.output,
                              W=l1.W * (1 - dropout_rate0),
                              b=l1.b,
                              filter_shape = (feat_map_n_1, 1, height, width1),  # two feature map, height: 1, width: 2, 
                              k_top = k_top,
                              number_of_convolutinal_layer=number_of_convolutinal_layer,
                              index_of_convolitonal_layer=1,
                              length_x=length_x,
                              activation = activation
    )


    l2 = DynamicConvFoldingPoolLayer(rng, 
                              input = dropout(l1.output, p=dropout_rate1), 
                              filter_shape = (feat_map_n_final, feat_map_n_1, height, width2),
                              # two feature map, height: 1, width: 2, 
                              k_top = k_top,
                              number_of_convolutinal_layer=number_of_convolutinal_layer,
                              index_of_convolitonal_layer=2,
                              length_x=length_x,
                              activation = activation
    )
    l2_no_dropout = DynamicConvFoldingPoolLayer(rng, 
                              input = l1_no_dropout.output,
                              W=l2.W * (1 - dropout_rate1),
                              b=l2.b,
                              filter_shape = (feat_map_n_final, feat_map_n_1, height, width2),
                              # two feature map, height: 1, width: 2, 
                              k_top = k_top,
                              number_of_convolutinal_layer=number_of_convolutinal_layer,
                              index_of_convolitonal_layer=2,
                              length_x=length_x,
                              activation = activation
    )


    # l2_output = theano.function(
    #     inputs = [x,length_x],
    #     outputs = l2.output,
    #     # on_unused_input='ignore'
    # ) 

    # TODO:
    # check the dimension
    # input: 1 x 1 x 6 x 4
    # out = l2_output(
    #     np.array([input_x], dtype = np.int32),
    #     len(input_x),
    # )


    # test = theano.function(
    #     inputs = [x],
    #     outputs = embeddings.output,
    # ) 


    # print "--input--"
    # print np.array([input_x], dtype = np.int32).shape
    # print "--input embeddings--"
    # a = np.array([input_x], dtype = np.int32)
    # print test(a).shape
    # print "-- output --"
    # print out
    # print out.shape



    # x = T.dscalar("x")
    # b = T.dscalar("b")
    # a = 1
    # f = theano.function(inputs=[x,b], outputs=b * x + a)
    # print f(2,2)


    # expected = (1, feat_map_n, EMB_DIM / 2, k)
    # assert out.shape == expected, "%r != %r" %(out.shape, expected)

    ##### Test Part Three ###############
    # LogisticRegressionLayer
    #################################

    # print "############# LogisticRegressionLayer ##############"

    l_final = LogisticRegression(
        rng, 
        input = dropout(l2.output.flatten(2), p=dropout_rate2),
        n_in = feat_map_n_final * k_top * EMB_DIM,
        # n_in = feat_map_n * k * EMB_DIM / 2, # we fold once, so divide by 2
        n_out = n_class, # five sentiment level
    )

    l_final_no_dropout = LogisticRegression(
        rng, 
        input = l2_no_dropout.output.flatten(2),
        W = l_final.W * (1 - dropout_rate2),
        b = l_final.b,
        n_in = feat_map_n_final * k_top * EMB_DIM,
        # n_in = feat_map_n * k * EMB_DIM / 2, # we fold once, so divide by 2
        n_out = n_class, # five sentiment level
    )


    print "n_in : ", feat_map_n_final * k_top * EMB_DIM
    # print "n_in = %d" %(2 * 2 * math.ceil(EMB_DIM / 2.))


    # p_y_given_x = theano.function(
    #     inputs = [x, length_x],
    #     outputs = l_final.p_y_given_x,
    #     allow_input_downcast=True,
    #     # mode = "DebugMode"
    # )

    # print "p_y_given_x = "
    # print p_y_given_x(
    #     np.array([input_x], dtype=np.int32),
    #     len(input_x)
    # )

    cost = theano.function(
        inputs = [x, length_x, y],
        outputs = l_final.nnl(y),
        allow_input_downcast=True,
        # mode = "DebugMode"
    )

    # print "cost:\n", cost(
    #     np.array([input_x], dtype = np.int32), 
    #     len(input_x),
    #     np.array([1], dtype = np.int32)
    # )



    print "############# Learning ##############"
    layers = []
    layers.append(embeddings)
    layers.append(l1)
    layers.append(l2)
    layers.append(l_final)

    cost = l_final.nnl(y)

    params = [p for layer in layers for p in layer.params]
    param_shapes = [l.param_shapes for l in layers]
    param_grads = [T.grad(cost, param) for param in params]


    def sgd(cost, params, lr=0.05):
        grads = [T.grad(cost, param) for param in params]
        updates = []
        for p, g in zip(params, grads):
            updates.append([p, p - g * lr])
        return updates

    from sgd import rmsprop, adagrad, adadelta, adam

    # updates = sgd(cost, l_final.params)

    # print param_grads
    if learn == "sgd":
        updates = sgd(cost, params, lr=0.05)
    elif learn == "adam":
        updates = adam(loss_or_grads=cost, params=params, learning_rate=alpha)
    elif learn == "adagrad":
        updates = adagrad(loss_or_grads=cost, params=params, learning_rate=alpha)
    elif learn == "adadelta":
        updates = adadelta(loss_or_grads=cost, params=params)
    elif learn == "rmsprop":
        updates = rmsprop(loss_or_grads=cost, params=params, learning_rate=alpha)


    train = theano.function(inputs=[x, length_x, y], outputs=cost, updates=updates, allow_input_downcast=True)
    # predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
    predict = theano.function(
        inputs = [x, length_x],
        outputs = T.argmax(l_final_no_dropout.p_y_given_x, axis=1),
        allow_input_downcast=True,
        # mode = "DebugMode"
    )




    def b(x_data):
        return np.array(x_data, dtype=np.int32)


    def test(test_set):
        # print "############# TEST ##############"
        y_pred = []
        test_set_y = []
        # for train_x, train_y in zip(X_data, Y_data):
        # print test_set
        # Accuracy_count = 0
        for test_y,test_x in test_set:
            test_x = b([test_x])
            p = predict(test_x, len(test_x))[0]
            y_pred.append(p)
            test_set_y.append(test_y)

            # if test_y == p:
            #     Accuracy_count += 1

            # print "*predict :",predict(train_x, len(train_x)), train_y 
        # Accuracy = float(Accuracy_count) / len(test_set)
        # print "  accuracy : %f" % Accuracy, 
        return accuracy_score(test_set_y, y_pred)
        # print classification_report(test_set_y, y_pred)

    # train_set_rand = np.ndarray(train_set)
    train_set_rand = train_set[:]
    train_cost_sum = 0.0
    for epoch in xrange(n_epoch):
        print "== epoch : %d =="  % epoch
        if shuffle_flag:
            np.random.shuffle(train_set_rand)
            # train_set_rand = np.random.permutation(train_set)
        for i,x_y_set in enumerate(train_set_rand):
            train_y, train_x = x_y_set
            train_x = b([train_x])
            train_y = b([train_y])

            train_cost = train(train_x, len(train_x) , train_y)
            train_cost_sum += train_cost
            if i % 1000 == 0 or i == len(train_set)-1:
                print "i : (%d/%d)" % (i, len(train_set)) , 
                print " (cost : %f )" % train_cost
        
        print '  cost :', train_cost_sum
        print '  train_set : %f' % test(train_set)
        print '  dev_set   : %f' % test(dev_set)
        print '  test_set  : %f' % test(test_set)





    '''
    print "x_average :",sum([len(x_y_set[1]) for i,x_y_set in enumerate(train_set)]) / float(len(train_set))

    # テストデータで過学習になる
    # python dynamic_CNN_for_sentences.py --alpha=0.01 --n_epoch=500 --activation=tanh --feat_map_n_1=1 --feat_map_n_final=1 --dropout_rate0=0.0 --dropout_rate1=0.0 --dropout_rate2=0.0 --width1=5 --width2=5

    X_data = []
    Y_data = []
    for _ in xrange(8544):
        l = np.random.randint(5, 50)
        X_data.append(np.random.randint(0, vocab_size, l))
        Y_data.append(np.random.randint(n_class))

    X_test = []
    Y_test = []
    for _ in xrange(2210):
        l = np.random.randint(5, 50)
        X_test.append(np.random.randint(0,vocab_size, l))
        Y_test.append(np.random.randint(n_class))



    def test(X_test, Y_test):
        # print "############# TEST ##############"
        y_pred = []
        # for train_x, train_y in zip(X_data, Y_data):
        for train_x, train_y in zip(X_test, Y_test):
            train_x = b([train_x])
            train_y = b([train_y])
            p = predict(train_x, len(train_x))[0]
            y_pred.append(p)
            # print "*predict :",predict(train_x, len(train_x)), train_y 
        # print classification_report(Y_data, y_pred)
        print classification_report(Y_test, y_pred)




    n_epoch = 100
    for epoch in xrange(n_epoch):
        print "=== epoch : ", epoch
        for i, (train_x, train_y) in enumerate(zip(X_data, Y_data)):
            train_x = b([train_x])
            train_y = b([train_y])
            # print "*train :", train_x, train_y
            train_cost = train(train_x, len(train_x), train_y)
            if i % 1000 == 0:
                print "i: (%d/%d)" % (i, len(X_data)), "*cost :", train_cost

        if epoch % 2 == 0:
            test(X_data, Y_data)
            test(X_test, Y_test)

    test()
    '''

if __name__ == '__main__':
    main()