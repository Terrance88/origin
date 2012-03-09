# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 13:47:16 2012

@author: User
"""

import numpy as np
import scipy as sc
import scipy.signal
import matplotlib.pyplot as plt
w = np.logspace(-2, 1, 1000)

def G(w):
    s = w*1j
    kc = 1
    Gp = 3*(-2*s+1)/((10*s+1)*(5*s+1))
    return Gp

def GM(w):
    return np.arctan2(np.imag(G(w)),np.real(G(w)))-np.pi
    
w_180=sc.optimize.fsolve(GM,0.001)

#y = Tr
kc = 1
L = kc*G(w)

def phase(L):
    return np.unwrap(np.arctan2(np.imag(L),np.real(L)))
#magnitude and phase; Bode plot
plt.subplot(2,1,1)
plt.loglog(w,abs(L))
plt.subplot(2,1,2)
plt.semilogx(w,(180/np.pi)*phase(L))
plt.semilogx(w,(-180)*np.ones(len(w)))
plt.show()
#for k in [0.5, 1.5, 2.5, 3]:
#    G = 3*(-2*s+1)/((10*s+1)*(5*s+1))
#    L = k*G
#    T = L/(1+L)
#    S = 1-T
#    plt.loglog(w,abs(S))
#    plt.show()