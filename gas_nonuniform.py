#!/usr/bin/env python

""" Module of bits for setting up non-uniform gas profile for Brewster"""

import numpy as np


def non_uniform_gas(press,logPt,logft,alpha):
    gas_f = np.zeros_like(press)
    
    Pt = 10.**logPt
    ft = 10.**logft
    
    for i in range(0,press.size):
        if (press[i] < Pt):
            gas_f[i]=logft+(np.log10(press[i])-logPt) / alpha
        else:
            gas_f[i]=float(logft)
    
    return gas_f


