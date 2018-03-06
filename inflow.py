# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 00:10:45 2018

@author: henri
"""

def inflow(u, n=0):
    if n == 0:
        u[0] = 1
    else:
        u[0:n] = 1
    return u