#!/usr/bin/env python
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np

def regularize_l1(l=.01):
    def l1wrap(g, p):
        g += T.sgn(p) * l
        return g
    return l1wrap

def regularize_l2(l=.01):
    def l2wrap(g, p):
        g += p * l
        return g
    return l2wrap

