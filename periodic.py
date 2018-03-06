# -*- coding: utf-8 -*-
"""
Created on Tue Mar  6 14:14:04 2018

@author: henri
"""

def periodic(u, n=0):
    if n == 0:
        u[0] = u[-2]
        u[-1] = u[1];
    else:
        u[:n] = u[-2*n:-n]
        u[-n:] = u[n:2*n]
    return u

#hei