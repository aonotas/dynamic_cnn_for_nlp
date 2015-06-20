#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np
import sys
# import test_rbms

def log_file(filename):
    train_set = []
    dev_set   = []
    test_set  = []
    for l in open(filename):
        v = l.split(":")
        if " train_set" in l:
            train_set.append(float(v[1]))
        if " dev_set" in l:
            dev_set.append(float(v[1]))
        if " test_set" in l:
            test_set.append(float(v[1]))
    return train_set, dev_set, test_set

def main():
    argvs = sys.argv
    filename = argvs[1]
    save_filename = filename.replace(".log", ".png").replace("log/","log/png/")
    train_set, dev_set, test_set = log_file(filename=filename)
    print "train_best", max(train_set)
    print "dev_best", max(dev_set)
    print "test_best", max(test_set)

    x = range(len(train_set))
    plt.plot(x, train_set, "b", linewidth=2, alpha=0.8, label="train")
    plt.plot(x, dev_set,   "r", linewidth=2, alpha=0.8, label="dev")
    plt.plot(x, test_set,  "g", linewidth=2, alpha=0.8, label="test")
    plt.legend(loc='lower right')
    # plt.show()
    print save_filename
    plt.savefig(save_filename)
    plt.show()

if __name__ == '__main__':
    main()