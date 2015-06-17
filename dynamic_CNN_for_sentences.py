#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import math
import numpy as np
import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
theano.config.exception_verbosity='high'

from dcnn_train import WordEmbeddingLayer,DynamicConvFoldingPoolLayer #, ConvFoldingPoolLayer
from logreg import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

rng = np.random.RandomState(1234)
srng = RandomStreams()

def main():

    print "############# Load Datasets ##############"

    import stanfordSentimentTreebank as sst
    vocab, index2word, datasets, funcs = sst.load_stanfordSentimentTreebank_dataset(normalize=True, skip_unknown_words=False)
    train_set, test_set, dev_set  = datasets
    get,sentence2ids, ids2sentence = funcs # 関数を読み込み


    dev_unknown_count  = sum([unknown_word_count for score,(ids,unknown_word_count) in dev_set])
    test_unknown_count = sum([unknown_word_count for score,(ids,unknown_word_count) in test_set])

    train_set = [(score, ids) for score,(ids,unknown_word_count) in train_set]
    test_set  = [(score, ids) for score,(ids,unknown_word_count) in test_set]
    dev_set   = [(score, ids) for score,(ids,unknown_word_count) in dev_set]

    print "train_set : ", len(train_set)
    print "dev_set   : ", len(dev_set)
    print "test_set  : ", len(test_set)
    print "-"*30
    print "vocab_szie: ", len(vocab)
    print "dev_unknown_words  : ", dev_unknown_count
    print "test_unknown_words : ", test_unknown_count



    EMB_DIM = 50
    vocab_size = len(vocab)


    feat_map_n_1 = 2
    feat_map_n_final = 1

    height = 1
    width1 = 10
    width2 = 7
    k_top  = 4
    n_class = 5
    alpha   = 0.01
    n_epoch = 500
    dropout_rate1 = 0.2
    dropout_rate2 = 0.5
    number_of_convolutinal_layer = 2


    # input_x = [1, 3, 4, 5, 0, 22, 4, 5]

    print "############# Model Setting ##############"    
    x = T.imatrix('x')
    length_x = T.iscalar('length_x')
    y = T.ivector('y') # the sentence sentiment label
    embeddings = WordEmbeddingLayer(rng, 
                            x,
                            vocab_size, EMB_DIM, None)


    def dropout(X, p=0.5):
        if p > 0:
            retain_prob = 1 - p
            X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
            X /= retain_prob
        return X
    # number_of_convolutinal_layer = theano.shared(number_of_convolutinal_layer)
    # dynamic_func = theano.function(inputs=[length_x], outputs=number_of_convolutinal_layer * length_x)

    # dynamic_func_test = theano.function(
    #     inputs = [length_x],
    #     outputs = dynamic_func(length_x),
    #     )
    # print dynamic_func(len([1,2,3]))

    l1 = DynamicConvFoldingPoolLayer(rng, 
                              input = embeddings.output, 
                              filter_shape = (feat_map_n_1, 1, height, width1),  # two feature map, height: 1, width: 2, 
                              k_top = k_top,
                              number_of_convolutinal_layer=number_of_convolutinal_layer,
                              index_of_convolitonal_layer=1,
                              length_x=length_x
    )


    l2 = DynamicConvFoldingPoolLayer(rng, 
                              input = dropout(l1.output, p=dropout_rate1), 
                              filter_shape = (feat_map_n_final, feat_map_n_1, height, width2),
                              # two feature map, height: 1, width: 2, 
                              k_top = k_top,
                              number_of_convolutinal_layer=number_of_convolutinal_layer,
                              index_of_convolitonal_layer=2,
                              length_x=length_x
    )
    l2_no_dropout = DynamicConvFoldingPoolLayer(rng, 
                              input = l1.output,
                              W=l2.W,
                              b=l2.b,
                              filter_shape = (feat_map_n_final, feat_map_n_1, height, width2),
                              # two feature map, height: 1, width: 2, 
                              k_top = k_top,
                              number_of_convolutinal_layer=number_of_convolutinal_layer,
                              index_of_convolitonal_layer=2,
                              length_x=length_x
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
        n_out = n_class # five sentiment level
    )

    l_final_no_dropout = LogisticRegression(
        rng, 
        input = l2_no_dropout.output.flatten(2),
        W = l_final.W,
        b = l_final.b,
        n_in = feat_map_n_final * k_top * EMB_DIM,
        # n_in = feat_map_n * k * EMB_DIM / 2, # we fold once, so divide by 2
        n_out = n_class # five sentiment level
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

    from sgd import rmsprop,adagrad,adadelta,adam

    # updates = sgd(cost, l_final.params)

    # print param_grads

    # updates = sgd(cost, params, lr=0.05)
    updates = adam(loss_or_grads=cost, params=params, learning_rate=alpha)

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

    for epoch in xrange(n_epoch):
        print "== epoch : %d =="  % epoch
        for i,x_y_set in enumerate(train_set):
            train_y, train_x = x_y_set
            train_x = b([train_x])
            train_y = b([train_y])

            train_cost = train(train_x, len(train_x) , train_y)
            if i % 1000 == 0:
                print "i : (%d/%d)" % (i, len(train_set)) , " (cost : %f )" % train_cost
        

        print '  train_set : %f' % test(train_set)
        print '  dev_set   : %f' % test(dev_set)
        print '  test_set  : %f' % test(test_set)

        





    '''
    X_data = []
    Y_data = []
    for _ in xrange(2000):
        length = np.random.randint(5, 50)
        X_data.append(np.random.randint(0,vocab_size, length))
        Y_data.append(np.random.randint(n_class))

    X_test = []
    Y_test = []
    for _ in xrange(50):
        length = np.random.randint(5, 50)
        X_test.append(np.random.randint(0,vocab_size, length))
        Y_test.append(np.random.randint(n_class))


    def b(x_data):
        return np.array(x_data, dtype=np.int32)



    def test():
        print "############# TEST ##############"
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
            train_cost = train(train_x, len(train_x) , train_y)
            if i % 100 == 0:
                print "*cost :",train_cost

        if epoch % 5 == 0:
            test()

    test()
    '''

if __name__ == '__main__':
    main()