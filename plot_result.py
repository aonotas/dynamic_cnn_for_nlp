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
    min_len = 1000
    for l in open(filename):
        v = l.split(":")
        train_flag = " train_set" in l
        dev_flag   = " dev_set"   in l
        test_flag  = " test_set"  in l
        if len(v) == 2 and (train_flag or dev_flag or test_flag):
            value = float(v[1])
        else:
            continue
        if train_flag:
            train_set.append(value)
        if dev_flag:
            dev_set.append(value)
        if test_flag:
            test_set.append(value)
        min_len = min(len(train_set), len(dev_set), len(test_set))
    return train_set[:min_len], dev_set[:min_len], test_set[:min_len]


def main():
    import glob


    argvs = sys.argv
    filename = argvs[1]
    filenames = [filename]
    if filename == "all":
        filenames = glob.glob('log/*.log')

    def plot(filename):
        save_filename = filename.replace(".log", ".png").replace("log/","log/png/")
        train_set, dev_set, test_set = log_file(filename=filename)
        print filename
        print "\t train_best", max(train_set)
        print "\t dev_best", max(dev_set)
        print "\t test_best", max(test_set)

        x = range(len(train_set))
        plt.ylim([0.0, 1.0])
        plt.plot(x, train_set, "b", linewidth=2, alpha=0.8, label="train")
        plt.plot(x, dev_set,   "r", linewidth=2, alpha=0.8, label="dev")
        plt.plot(x, test_set,  "g", linewidth=2, alpha=0.8, label="test")
        plt.legend(loc='lower right')
        # plt.show()
        # print save_filename

        plt.savefig(save_filename)
        plt.show()


    for filename in filenames:
        plot(filename)
        
if __name__ == '__main__':
    main()