#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Functions for calculating Linear Combinations of signal frequencies
of GNSS.
'''

__author__ = 'hanbing'

# Standard library imports
import math

def L3(f1,SigmaL1,SigmaP1,f2,SigmaL2,SigmaP2):
    """
    Calculate error spreading law for LC combination.

         f1 --- Frequency of L1 in MHz
    SigmaL1 --- Sigmal of L1 in cycle
    SigmaP1 --- Sigmal of P1 in meter
         f2 --- Frequency of L2 in MHz
    SigmaL2 --- Sigmal of L2 in cycle
    SigmaP2 --- Sigmal of P2 in meter

    """

    vlight=299792458
    f1=f1*1000000
    f2=f2*1000000
    #Combination coefficients in meter
    a=f1**2/(f1**2-f2**2)
    b=f2**2/(f1**2-f2**2)
    #Combination sigma of PC in meter
    SigmaPC=math.sqrt((a*SigmaP1)**2 + (b*SigmaP2)**2)
    #Combination sigma of LC in millimeter
    SigmaLC=math.sqrt((a*(vlight/f1*SigmaL1))**2 + (b*(vlight/f2*SigmaL2))**2)*1e3

    StrTmp='{:>14.6f} {:>14.6f} {:>11.2f} {:>14.5f}'.format(a,b,SigmaLC,SigmaPC)
    print(StrTmp)

def L4(f1,SigmaL1,f2,SigmaL2):
    """
    Calculate error spreading law for LG combination.

         f1 --- Frequency of L1 in MHz
    SigmaL1 --- Sigmal of L1 in cycle
         f2 --- Frequency of L2 in MHz
    SigmaL2 --- Sigmal of L2 in cycle

    """

    vlight=299792458
    f1=f1*1000000
    f2=f2*1000000
    #Combination coefficients of phase in cycle
    a= f2/(f1-f2)
    b=-f1/(f1-f2)
    #Combination sigma of LG in millimeter
    SigmaLG1=math.sqrt((vlight/f1*SigmaL1)**2 + (vlight/f2*SigmaL2)**2)*1e3
    #Combination sigma of LG in cycle
    SigmaLG2=math.sqrt((a*SigmaL1)**2 + (b*SigmaL2)**2)

    StrTmp='{:>14.6f} {:>14.6f} {:>11.2f} {:>14.5f}'.format(a,b,SigmaLG1,SigmaLG2)
    print(StrTmp)

def L6(f1,SigmaL1,SigmaP1,f2,SigmaL2,SigmaP2):
    """
    Calculate error spreading law for MW combination.

         f1 --- Frequency of L1 in MHz
    SigmaL1 --- Sigmal of L1 in cycle
    SigmaP1 --- Sigmal of P1 in meter
         f2 --- Frequency of L2 in MHz
    SigmaL2 --- Sigmal of L2 in cycle
    SigmaP2 --- Sigmal of P2 in meter

    """

    vlight=299792458
    f1=f1*1000000
    f2=f2*1000000
    #Combination coefficients of phase in meter
    a1= f1/(f1-f2)
    b1=-f2/(f1-f2)
    #Combination coefficients of code in meter
    a2=f1/(f1+f2)
    b2=f2/(f1+f2)
    #Combination sigma of MW in meter
    SigmaMW1=math.sqrt((a1*(vlight/f1*SigmaL1))**2 + (b1*(vlight/f2*SigmaL2))**2 +
                       (a2*SigmaP1)**2 + (b2*SigmaP2)**2)
    #Combination sigma of MW in cycle
    SigmaMW2=math.sqrt(SigmaL1**2 + SigmaL2**2 +
                       (SigmaP1*(f1-f2)*f1/vlight/(f1+f2))**2 +
                       (SigmaP2*(f1-f2)*f2/vlight/(f1+f2))**2)

    StrTmp='{:>14.6f} {:>14.6f} {:>14.6f} {:>14.6f} {:>14.5f} {:>14.5f}'.format(a1,b1,a2,b2,SigmaMW1,SigmaMW2)
    print(StrTmp)

def L42TEC(f1,f2,L4,lTEC):
    """
    Convert L4 (in meter) to TEC (in TECU)

         f1 --- Frequency of L1 in MHz
         f2 --- Frequency of L2 in MHz
         L4 --- LG combination in meter
       lTEC --- Convert TEC to L4
    """

    if lTEC:
        #Convert TEC to L4
        TEC=L4*40.3*1e4*(f1**2-f2**2)/(f1*f1*f2*f2)
    else:
        #Convert L4 to TEC
        TEC=L4*f1*f1*f2*f2/(40.3*1e4*(f1**2-f2**2))

    StrTmp='{:>14.5f} {:>14.5f}'.format(L4,TEC)
    print(StrTmp)


# L3(1561.098,0.01,0.6,1207.140,0.01,0.6)
# BDS B1I+B3I
L3(1561.098,0.01,0.5,1268.520,0.01,0.5)
# # BDS B3I+B2I
# L3(1268.520,0.01,0.6,1207.140,0.01,0.6)

# L4(1561.098,0.01,1207.140,0.01)
# L4(1561.098,0.01,1268.520,0.01)
# L4(1268.520,0.01,1207.140,0.01)

# L6(1561.098,0.01,0.6,1207.140,0.01,0.6)
# L6(1561.098,0.01,0.6,1268.520,0.01,0.6)
# L6(1268.520,0.01,0.6,1207.140,0.01,0.6)

# L42TEC(1561.098,1207.140,0.03*3600,True)
# L42TEC(1561.098,1268.520,0.03*3600,True)
# #B1C+B2a
# L42TEC(1575.420,1176.450,0.03*3600,True)