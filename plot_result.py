#!/usr/bin/env python
# -*- coding: utf-8 -*-

from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
import numpy as np

import test_rbms

"""
Main functions: plot history of RBM training
"""

def plot_time_bbrbm():
    # Parameters of BBRBM to be loaded
    t_end = 100
    rbm1 = test_rbms._init_bbrbm(
        None, t_start=0, t_end=t_end, n_components=512, eps=0.01)
    rbm_file1 = 'bbrbm-' + rbm1.get_str() + '-t_' + str(t_end) + '.pklz.cpu'
    rbm_file2 = 'bbrbm-' + rbm1.get_str() + '-t_' + str(t_end) + '.pklz.gpu'
    rbm_files = [rbm_file1, rbm_file2]

    # Time series of reconstruction error over time
    hists = [load_rbm_hist(rbm_file) for rbm_file in rbm_files]
    tss = np.vstack([hist[0] for hist in hists]).T
    errss = np.vstack([hist[1] for hist in hists]).T

    # Plot
    fig = plt.figure(figsize=(10, 7.5))

    ax = fig.add_subplot(2, 1, 1)
    ax.plot(tss, errss)
    ax.set_ylabel('Error')
    plt.legend(['CPU', 'GPU'])
    ax.set_xlabel('Time [sec]')

    ax = fig.add_subplot(2, 1, 2)
    patches = ax.barh(
        [0.4, 1.0], tss[-1, :], height=0.4, align='center')
    plt.yticks([0.4, 1.0], ['CPU', 'GPU'])
    plt.ylim([0.0, 1.4])
    ax.set_xlabel('Time [sec]')

    xs = [p.get_xy()[0] for p in patches]
    ys = [p.get_xy()[1] for p in patches]
    ws = [p.get_width() for p in patches]
    hs = [p.get_height() for p in patches]
    ts = ['%4.1f' % t for t in tss[-1, :]]
    for x, y, w, h, t in zip(xs, ys, ws, hs, ts):
        ax.text(
            x + 0.5 * w, y + 0.5 * h, t, ha='center', va='center', 
            color='w', fontproperties=FontProperties(weight='bold'))

    plt.show()

    print('CPU: %4.1f [sec]' % tss[-1, 0])
    print('GPU: %4.1f [sec]' % tss[-1, 1])



"""
Utility functions: load result of RBM training
"""

def load_rbm_hist(rbm_file):
    data = test_rbms.load_pklz(rbm_file)

    dts = [log[0] for log in data[1]['hist']]
    errs = [log[1] for log in data[1]['hist']]
    
    buf, ts = 0., []

    for dt in dts:
        buf += dt
        ts.append(buf)

    ts = np.array(ts)

    return ts, errs

