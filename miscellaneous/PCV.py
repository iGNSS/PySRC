#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Calculate PCV models
'''
__author__ = 'hanbing'

# Standard library imports
import os
import sys
import os.path
import glob
import math

# Related third party imports
import numpy as np
import scipy.special as special

def GetPCVSphHarm(amin,amax,zmin,zmax,n,m,Anm,Bnm):
    '''
    Calculate the spherical harmonic function model for PCV.

    Ref. [1] Bernhard Hofmann-Wellenhof, Herbert Lichtenegger, Elmar Wasle. 
             GNSS – Global Navigation Satellite Systems GPS, GLONASS, Galileo, 
             and more. Chap 5.5.3, Equ (5.150).
    '''

    a=np.arange(amin,amax,0.50)
    z=np.arange(zmin,zmax,0.25)
    y=np.zeros((len(a),len(z)))
    for i in range(len(a)):
        for j in range(len(z)):
            y[i,j]=0.0
            for k in range(n+1):
                for l in range(k+1):
                    y[i,j]=y[i,j] + \
                           (Anm[k,l]*np.cos(l*np.deg2rad(a[i])) + \
                            Bnm[k,l]*np.sin(l*np.deg2rad(a[i]))) * \
                           special.lpmv(l,k,np.cos(np.deg2rad(z[j]))) * \
                           np.sqrt(math.factorial(k-l)/math.factorial(k+l)*(2*k+1)/4/np.pi)
    
    return a,z,y
