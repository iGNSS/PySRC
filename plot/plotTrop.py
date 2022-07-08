#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Plot some results related to trop
'''
__author__ = 'hanbing'

# Standard library imports
import os
import sys
import os.path
import glob
import datetime

# Related third party imports
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# Local application/library specific imports
from PySRC.miscellaneous import GNSSTime

def PlotTropDiff(fTrop1,fTrop2,cSta,OutFilePrefix,OutFileSuffix):
    '''
    Plot the diff of ZTD between two files for a station

    fTrop1 --- IGS SINEX_TRO file
    fTrop2 --- PANDA trop solution file
    '''

    # SINEX_TROP file
    ZTD1=[[],[]]
    with open(fTrop1,mode='rt') as fOb:
        lBeg=False
        for cLine in fOb:
            if cLine[0:14] == '+TROP/SOLUTION':
                lBeg=True
            elif cLine[0:14] == '-TROP/SOLUTION':
                break
            elif lBeg:
                cWords=cLine.split()
                if cWords[0] != cSta:
                    continue
                cStr=cWords[1].split(sep=':')
                YYYY=GNSSTime.yr2year(int(cStr[0]))
                rMJD=GNSSTime.doy2mjd(YYYY,int(cStr[1]))
                rMJD=rMJD + float(cStr[2])/86400
                ZTD1[0].append(rMJD)
                # ZTD in mm
                ZTD1[1].append(float(cWords[2]))

    # PANDA trop solution file
    ZTD2=[[],[]]
    with open(fTrop2,mode='rt') as fOb:
        for cLine in fOb:
            if len(cLine) < 10:
                continue
            cWords=cLine.split()
            if cWords[3] != cSta:
                continue
            rMJD=int(cWords[0]) + float(cWords[1])/86400
            if len(ZTD2[0]) > 0 and np.abs(rMJD-ZTD2[0][-1])*86400 < 1:
                # same epoch
                continue
            ZTD2[0].append(rMJD)
            # ZTD in mm
            ZTD2[1].append(float(cWords[7])*1e3)

    fig,axs=plt.subplots(2,1,sharex='col',squeeze=False,figsize=(8,4))
    formatterx=mpl.ticker.StrMethodFormatter('{x:7.1f}')

    # plot the trop set
    axs[0,0].plot(ZTD1[0],ZTD1[1],'.r',ms=3,label='gfz')
    axs[0,0].plot(ZTD2[0],ZTD2[1],'.b',ms=3,label='phb')
    axs[0,0].set_ylabel('[mm]',fontname='Arial',fontsize=16)
    for tl in axs[0,0].get_yticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)
    axs[0,0].legend(loc='upper right',framealpha=0.6,prop={'family':'Arial','size':14})

    # Cal the diff

    axs[1,0].set_ylabel('[mm]',fontname='Arial',fontsize=16)
    for tl in axs[1,0].get_yticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)

    axs[1,0].set_xlabel('Modified Julian Day',fontname='Arial',fontsize=16)
    for tl in axs[1,0].get_xticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)
    axs[1,0].xaxis.set_major_formatter(formatterx)

    strTmp=OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    import argparse


    # SINEX_TRO file
    fTrop1=r'Y:/MGEX/ZPD/2019/335/gmsd3350.19zpd'
    # PANDA TRO solution file
    fTrop2=r'Y:/PRO_2019001_2020366_WORK/I2_1/WORK2019335/tropO_2019335'

    OutFilePrefix=r'Z:/PRO_2019001_2020366/I2_1/TRO/'
    OutFileSuffix='Trop_GMSD.png'
    PlotTropDiff(fTrop1,fTrop2,'GMSD',OutFilePrefix,OutFileSuffix)
