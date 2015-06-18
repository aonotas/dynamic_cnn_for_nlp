#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cPickle
import gzip
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import sklearn.decomposition as decomp

import bbrbm
import gbrbm



"""
Main functions: test BBRBM
"""

def train_bbrbm(
    t_start, t_end, n_components=512, alpha=0.0, eps=1e-2, algorithm='pcd', 
    lmd_w=0.0, lmd_h=0.0, batch_size=50, n_particles=50):
    # Initialization of RBM
    rbm      = _init_bbrbm(
        None, t_start, t_end, n_components=n_components, alpha=alpha, eps=eps, 
        algorithm=algorithm, lmd_w=lmd_w, lmd_h=lmd_h, batch_size=batch_size, 
        n_particles=n_particles)

    # Get RBM filename
    rbm_file = 'bbrbm-' + rbm.get_str() + '-t_' + str(t_end) + '.pklz'

    # Training with parameters previouslly saved or new parameters
    if 0 < t_start:
        init_file = rbm.get_str() + '-t_' + str(t_start) + '.pklz'
        data      = load_pklz(init_file)
        pca       = data[0]
        rbm       = _init_bbrbm(data[1], t_start, t_end)

        train_rbm(rbm, rbm_file, pca)
    else:
        train_rbm(rbm, rbm_file, 0)

def plot_bbrbm_weights(
    t_end, n_components=512, alpha=0.0, eps=1e-2, algorithm='pcd', 
    lmd_w=0.0, lmd_h=0.0, batch_size=50, n_particles=50):
    # Get RBM filename
    rbm      = _init_bbrbm(
        None, -1, t_end, n_components=n_components, alpha=alpha, eps=eps, 
        algorithm=algorithm, lmd_w=lmd_w, lmd_h=lmd_h, batch_size=batch_size, 
        n_particles=n_particles)
    rbm_file = 'bbrbm-' + rbm.get_str() + '-t_' + str(t_end) + '.pklz'

    # Extract weights from RBM parameters
    def get_hidden_patterns(params_model, ixs):
        return params_model['w'][:, ixs].T

    # Plot
    plot_rbm_weights(rbm_file, get_hidden_patterns, vminmax=(-0.5, 0.5))

def plot_bbrbm_recon(
    t_end, n_components=512, alpha=0.0, eps=1e-2, algorithm='pcd', 
    lmd_w=0.0, lmd_h=0.0, batch_size=50, n_particles=50):
    # Get RBM filename
    rbm      = _init_bbrbm(
        None, -1, t_end, n_components=n_components, alpha=alpha, eps=eps, 
        algorithm=algorithm, lmd_w=lmd_w, lmd_h=lmd_h, batch_size=batch_size, 
        n_particles=n_particles)
    rbm_file = 'bbrbm-' + rbm.get_str() + '-t_' + str(t_end) + '.pklz'

    # Load RBM
    pca, params_model, params_tr = load_pklz(rbm_file)
    rbm = bbrbm.BBRBM(params_tr, params_model, 0, 0)

    # Plot
    plot_rbm_recon(pca, rbm, vminmax=(0.0, 1.0))



"""
Main functions: test GBRBM
"""

def train_gbrbm(
    t_start, t_end, n_components=512, alpha=0.0, eps=1e-6, algorithm='pcd', 
    lmd_w=0.0, lmd_h=0.0, batch_size=20, n_particles=20):
    # Initialization of RBM
    rbm      = _init_gbrbm(
        None, t_start, t_end, n_components=n_components, alpha=alpha, eps=eps, 
        algorithm=algorithm, lmd_w=lmd_w, lmd_h=lmd_h, batch_size=batch_size, 
        n_particles=n_particles)

    # Get RBM filename
    rbm_file = 'gbrbm-' + rbm.get_str() + '-t_' + str(t_end) + '.pklz'

    # Training with parameters previouslly saved or new parameters
    if 0 < t_start:
        init_file = rbm.get_str() + '-t_' + str(t_start) + '.pklz'
        data      = load_pklz(init_file)
        pca       = data[0]
        rbm       = _init_gbrbm(data[1], t_start, t_end)

        train_rbm(rbm, rbm_file, pca)
    else:
        train_rbm(rbm, rbm_file, 256)

def plot_gbrbm_weights(
    t_end, n_components=512, alpha=0.0, eps=1e-6, algorithm='pcd', lmd_w=0.0, 
    lmd_h=0.0, batch_size=20, n_particles=20):
    # Get RBM filename
    rbm      = _init_gbrbm(
        None, -1, t_end, n_components=n_components, alpha=alpha, eps=eps, 
        algorithm=algorithm, lmd_w=lmd_w, lmd_h=lmd_h, batch_size=batch_size, 
        n_particles=n_particles)
    rbm_file = 'gbrbm-' + rbm.get_str() + '-t_' + str(t_end) + '.pklz'

    # Extract weights from RBM parameters
    def get_hidden_patterns(params_model, ixs):
        return params_model['w'][:, ixs].T

    # Plot
    plot_rbm_weights(rbm_file, get_hidden_patterns, vminmax=(-0.1, 0.1))
    # plot_rbm_weights(rbm_file, get_hidden_patterns, vminmax=None)

def plot_gbrbm_recon(
    t_end, n_components=512, alpha=0.0, eps=1e-6, algorithm='pcd', lmd_w=0.0, 
    lmd_h=0.0, batch_size=20, n_particles=20):
    # Get RBM filename
    rbm      = _init_gbrbm(
        None, -1, t_end, n_components=n_components, alpha=alpha, eps=eps, 
        algorithm=algorithm, lmd_w=lmd_w, lmd_h=lmd_h, batch_size=batch_size, 
        n_particles=n_particles)
    rbm_file = 'gbrbm-' + rbm.get_str() + '-t_' + str(t_end) + '.pklz'

    # Load RBM
    pca, params_model, params_tr = load_pklz(rbm_file)
    rbm = gbrbm.GBRBM(params_tr, params_model, 0, 0)

    # Plot
    plot_rbm_recon(pca, rbm, vminmax=(0.0, 1.0))



"""
Utility functions: test RBM
"""

def train_rbm(rbm, rbm_file, pca0):
    # Load data
    xs_tr, xs_te = _load_mnist_data()

    # Transform data to subspace
    if type(pca0) == int:
        pca = MyPCA(n_components=pca0)
        pca = pca.fit(xs_tr)
    else:
        pca = pca0

    _xs_tr = pca.transform(xs_tr)
    _xs_te = pca.transform(xs_te)

    if pca.pca is not None:
        v = np.sum(pca.pca.explained_variance_ratio_)
        print('%2.3f percent of variance explained' % (v * 100.0))

    # Fit RBM
    params_tr    = rbm.params_tr
    params_model = rbm.fit(_xs_tr, _xs_te).params_model

    # Save estimated parameters
    save_pklz(rbm_file, (pca, params_model, params_tr))

def plot_rbm_weights(
    rbm_file, get_hidden_patterns, out_to_file=False, vminmax=None):
    # Load estimated parameters
    pca, params_model, _ = load_pklz(rbm_file)

    # Grids for plot
    nx, ny = 5, 5
    gs     = gridspec.GridSpec(nx, ny, wspace=0.0, hspace=0.0)

    # Indices of hidden components to be plotted
    n_components = params_model['w'].shape[1]
    ixs          = np.random.permutation(n_components)[:(nx * ny)]

    # Get weights corresponding to original data space
    ws = get_hidden_patterns(params_model, ixs)
    ws = pca.inverse_transform(ws, add_mean=False)

    # Plot weights
    _plot_patterns(ws, gs, out_to_file=out_to_file, vminmax=vminmax)

def plot_rbm_recon(pca, rbm, vminmax=None):
    # Grids for plot
    ny, nx = 4, 5
    gs     = gridspec.GridSpec(ny, nx, wspace=0.0, hspace=0.0)

     # Load and transform data
    _, xs_te0 = _load_mnist_data()
    n_data    = xs_te0.shape[0]
    ixs       = np.random.permutation(n_data)[:(nx * ny) / 2]
    xs_te0    = xs_te0[ixs, :]
    xs_te     = pca.transform(xs_te0)

    # Reconstruction and backward transformation
    xs_rec = rbm.reconstruct(xs_te)
    xs_rec = pca.inverse_transform(xs_rec)

    # Plot reconstructed data
    _plot_patterns(
        np.vstack((xs_rec, xs_te0)), gs, out_to_file=True, vminmax=vminmax)



"""
Utility function: initialization of RBMs
"""

def _init_bbrbm(
    params_model, t_start, t_end, n_components=256, alpha=0.0, eps=1e-4, 
    lmd_w=0.0, lmd_h=0.0, algorithm='pcd', batch_size=50, n_particles=50):
    # BBRBM parameters
    decay        = 1.0
    is_update    = {'w': True, 'b': True, 'c': True}
    init_w       = {'scale': 0.01, 'distribution': 'normal'}
    init_b       = {'scale': 0.01, 'distribution': 'normal'}
    init_c       = {'scale': 0.01, 'distribution': 'normal'}
    n_mod        = 1
    params_tr    = gbrbm.GBRBMTrainingParams(
        n_components=n_components, algorithm=algorithm,  
        batch_size=batch_size, n_particles=n_particles, 
        alpha=alpha, eps=eps, decay=decay, lmd_w=lmd_w, lmd_h=lmd_h, 
        is_update=is_update, init_w=init_w, init_b=init_b, init_c=init_c, 
        n_mod=n_mod)

    rbm = bbrbm.BBRBM(params_tr, params_model, t_start, t_end)

    return rbm

def _init_gbrbm(
    params_model, t_start, t_end, n_components=256, alpha=0.0, eps=1e-4, 
    lmd_w=0.0, lmd_h=0.0, algorithm='pcd', batch_size=50, n_particles=50):
    # GBRBM parameters
    n_rate       = 1
    decay        = 1.0
    is_update    = {'w': True, 'z': True, 'b': True, 'c': True}
    init_w       = {'scale': 0.001  ,      'distribution': 'normal'}
    init_z       = {'scale': np.log(0.01), 'distribution': 'const'}
    init_b       = {'scale': 0.01,         'distribution': 'normal'}
    init_c       = {'scale': 0.01,         'distribution': 'normal'}
    n_mod        = 1
    params_tr    = gbrbm.GBRBMTrainingParams(
        n_components=n_components, algorithm=algorithm,  
        n_rate=n_rate, batch_size=batch_size, n_particles=n_particles, 
        alpha=alpha, eps=eps, decay=decay, lmd_w=lmd_w, lmd_h=lmd_h, 
        is_update=is_update, init_w=init_w, init_z=init_z, init_b=init_b, 
        init_c=init_c, n_mod=n_mod)

    rbm = gbrbm.GBRBM(params_tr, params_model, t_start, t_end)

    return rbm



"""
Utility function: save and load pklz file
"""

def save_pklz(fileName, obj):
    f = gzip.GzipFile(fileName, 'wb')
    f.write(cPickle.dumps(obj))
    f.close()

def load_pklz(fileName):
    f   = gzip.GzipFile(fileName, 'rb')
    obj = cPickle.load(f)
    f.close()
    return obj



"""
Utility functions: misc
"""

def _load_mnist_data():
    # f = gzip.open(env.dataDir + '/mnist/mnist.pkl.gz', 'rb')
    f = gzip.open('./mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    xs_tr, _ = train_set
    xs_te, _ = test_set

    return xs_tr, xs_te

def _plot_patterns(ws, gs, out_to_file, vminmax=None):
    for i in xrange(len(ws)):
        ax = plt.subplot(gs[i])
        if vminmax is None:
            ax.imshow(ws[i, :].reshape((28, 28)), interpolation='none', 
                      cmap=cm.Greys_r)
        else:
            ax.imshow(ws[i, :].reshape((28, 28)), interpolation='none', 
                      cmap=cm.Greys_r, vmin=vminmax[0], vmax=vminmax[1])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    if out_to_file is True:
        plt.savefig('./result.png')
        print('Figure is saved as result.png.')
    else:
        plt.show()

class MyPCA:
    def __init__(self, n_components):
        if 0 < n_components:
            self.pca = decomp.PCA(n_components=n_components)
        else:
            self.pca = None

    def fit(self, xs):
        if self.pca is not None:
            self.ms  = np.mean(xs, axis=0)
            self.pca = self.pca.fit(xs - self.ms)
        else:
            pass

        return self

    def transform(self, xs):
        if self.pca is not None:
            return self.pca.transform(xs - self.ms)
        else:
            return xs

    def inverse_transform(self, xs, add_mean=True):
        if self.pca is not None:
            if add_mean is True:
                return self.pca.inverse_transform(xs) + self.ms
            else:
                return self.pca.inverse_transform(xs)                
        else:
            return xs

