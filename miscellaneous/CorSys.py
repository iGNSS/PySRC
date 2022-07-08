#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Functions for transformation between coordinate systems
'''

__author__ = 'hanbing'

# Related third party imports
import numpy as np

def XYZ2BLH(X,Y,Z):
    '''
    XYZ -> BLH
    Base on WGS-84

    XYZ --- cartesian coordinate of the point, in metres
    BLH --- ellipsoidal latitude, longitude (in rad ) and height (in meter)
    '''

    a=6378137
    finv=298.257223563
    shift=0

    b=a-a/finv
    e2=(a*a-b*b)/(a*a)

    s=np.sqrt(X*X+Y*Y)
    Lon=np.arctan2(Y,X)
    if Lon < 0:
        Lon=2*np.pi + Lon
    zps=Z/s
    H=np.sqrt(X*X+Y*Y+Z*Z)-a
    Lat=np.arctan(zps/(1-e2*a/(a+H)))

    for i in range(10):
        N=a/np.sqrt(1-e2*np.sin(Lat)**2)
        H0=H
        Lat0=Lat
        H=s/np.cos(Lat)-N
        Lat=np.arctan(zps/(1-e2*N/(N+H)))
        if np.abs(Lat0-Lat) <= 1e-11 and np.abs(H0-H)<=1e-5:
            break
    return Lat,Lon,H

def RotENU2TRS(lat,lon):
    '''
    '''

    rotmat=np.zeros((3,3))

    sinlat=np.sin(lat)
    coslat=np.cos(lat)

    sinlon=np.sin(lon)
    coslon=np.cos(lon)

    rotmat[0,0]=-sinlon
    rotmat[0,1]=-coslon*sinlat
    rotmat[0,2]= coslon*coslat
    rotmat[1,0]= coslon
    rotmat[1,1]=-sinlon*sinlat
    rotmat[1,2]= sinlon*coslat
    rotmat[2,0]= 0
    rotmat[2,1]= coslat
    rotmat[2,2]= sinlat

    return rotmat
