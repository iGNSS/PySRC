#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Plot Helmert transformation parameters
'''
__author__ = 'hanbing'

# Standard library imports
import subprocess
import os
import sys
import os.path
import glob
import datetime

# Related third party imports
import numpy as np
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# Local application/library specific imports
from PySRC.miscellaneous import GNSSTime


def GetHel(fList, ExlMJD):
    '''
    Read Helmert parameters

        ExlMJD --- MJD list to be excluded

    Return:
          rMJD --- MJD epoch
          rEpo --- date epoch
           Hel ---
                   # 0, D
                   # 1, D rate
                   # 2, Tx
                   # 3, Tx rate
                   # 4, Ty
                   # 5, Ty rate
                   # 6, Tz
                   # 7, Tz rate
                   # 8, Rx
                   # 9, Rx rate
                   #10, Ry
                   #11, Ry rate
                   #12, Rz
                   #13, Rz rate
    '''

    nFile = len(fList)
    rEpo = []
    rMJD = []
    nSta = np.zeros((nFile, 2), dtype=np.int32)
    Hel = np.zeros((nFile, 14))
    Sig = np.zeros((nFile, 14))
    Hel[:, :] = np.nan
    Sig[:, :] = np.nan
    for i in range(nFile):
        with open(fList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:23] == 'Number of stations used':
                    nSta[i, 0] = int(cLine[36:41])
                    # number of excluded stations
                    nSta[i, 1] = int(cLine[42:47])
                elif cLine[0:7] == 'Helmert':
                    cWords = cLine[8:].split()
                    MJD = int(cWords[0])
                    rMJD.append(MJD)
                    YYYY, MO, DD = GNSSTime.mjd2dom(MJD)
                    YYYY, DOY = GNSSTime.mjd2doy(MJD)
                    rEpo.append(datetime.datetime(YYYY, MO, DD))
                    if MJD in ExlMJD:
                        continue
                    #D, ppb
                    Hel[i, 0] = float(cWords[3])
                    Sig[i, 0] = float(cWords[4])
                    # D rate
                    Hel[i, 1] = float(cWords[5])
                    Sig[i, 1] = float(cWords[6])
                    #Tx, mm
                    Hel[i, 2] = float(cWords[7])
                    Sig[i, 2] = float(cWords[8])
                    # Tx rate
                    Hel[i, 3] = float(cWords[9])
                    Sig[i, 3] = float(cWords[10])
                    # Ty
                    Hel[i, 4] = float(cWords[11])
                    Sig[i, 4] = float(cWords[12])
                    # Ty rate
                    Hel[i, 5] = float(cWords[13])
                    Sig[i, 5] = float(cWords[14])
                    # Tz
                    Hel[i, 6] = float(cWords[15])
                    Sig[i, 6] = float(cWords[16])
                    # Tz rate
                    Hel[i, 7] = float(cWords[17])
                    Sig[i, 7] = float(cWords[18])
                    #Rx, mas
                    Hel[i, 8] = float(cWords[19])
                    Sig[i, 8] = float(cWords[20])
                    # Rx rate
                    Hel[i, 9] = float(cWords[21])
                    Sig[i, 9] = float(cWords[22])
                    # Ry
                    Hel[i, 10] = float(cWords[23])
                    Sig[i, 10] = float(cWords[24])
                    # Ry rate
                    Hel[i, 11] = float(cWords[25])
                    Sig[i, 11] = float(cWords[26])
                    # Rz
                    Hel[i, 12] = float(cWords[27])
                    Sig[i, 12] = float(cWords[28])
                    # Rz rate
                    Hel[i, 13] = float(cWords[29])
                    Sig[i, 13] = float(cWords[30])
                    # Report the significant rotation (>0.02 mas)
                    if abs(Hel[i, 8]) > 0.02 or abs(Hel[i, 10]) > 0.02 or abs(Hel[i, 12]) > 0.02:
                        strTmp = '{: >5d} {: >4d} {: >03d} {: >7.2f} {: >7.2f} {: >7.2f}'.format(MJD,
                                                                                                 YYYY, DOY, Hel[i, 8], Hel[i, 10], Hel[i, 12])
                        print(strTmp)

    return rMJD, rEpo, nSta, Hel, Sig


def PlotHel1(fList, ExlMJD, lStaNum, cHel, yMax, lDate, OutFilePrefix, OutFileSuffix):
    '''
    Plot all Helmert parameters in a single figure

   fList --- List of Helmert transformation result files
  ExlMJD --- MJD list to be excluded
 lStaNum --- Whether plot also the station number
    cHel --- Specified Hel Pars to be plotted, each par takes an axis
    yMax ---
   lDate --- Use date or MJD for the x-axis
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    rMJD, rEpo, nSta, Hel, Sig = GetHel(fList, ExlMJD)
    nHel0 = 13
    cHel0 = ['D', 'dD',
             'TX', 'dTX', 'TY', 'dTY', 'TZ', 'dTZ',
             'RX', 'dRX', 'RY', 'dRY', 'RZ', 'dRZ']
    cYLab = ['Scale [ppb]', '[ppb/d]',
             'TX [mm]', '[mm/d]', 'TY [mm]', '[mm/d]', 'TZ [mm]', '[mm/d]',
             'RX [mas]', '[mas/d]', 'RY [mas]', '[mas/d]', 'RZ [mas]', '[mas/d]']

    if lDate:
        x = rEpo
    else:
        x = rMJD

    # for i in range(len(fList)):
    #     #Exclude days with less than 100 valid stations
    #     if nSta[i,0] < 100:
    #         nSta[i,:]=np.nan
    #         Hel[i,:]=np.nan

    nHel = len(cHel)
    if lStaNum:
        nAx = nHel+1
    else:
        nAx = nHel
    fig, axs = plt.subplots(nAx, 1, squeeze=False,
                            sharex='col', figsize=(8, nAx*3))
    fig.subplots_adjust(hspace=0.1)
    formattery = mpl.ticker.StrMethodFormatter('{x:5.2f}')

    for i in range(nAx):
        if lStaNum and i == 0:
            axs[0, 0].plot(x, nSta[:, 0], '^g', ms=6)

            axs[0, 0].set_ylabel('Number of sites used',
                                 fontname='Arial', fontsize=16)
            axs[0, 0].yaxis.set_major_formatter(FormatStrFormatter('%d'))
            for tl in axs[0, 0].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            ax0 = axs[0, 0].twinx()
            ax0.plot(x, nSta[:, 1], 'vr', ms=6)
            ax0.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            ax0.set_ylabel('Number of sites excl',
                           fontname='Arial', fontsize=16)
            for tl in ax0.get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)

            continue
        elif lStaNum:
            k = i-1
        else:
            k = i
        j = cHel0.index(cHel[k])
        axs[i, 0].plot(x, Hel[:, j], 'o', ms=6)
        axs[i, 0].set_ylabel(cYLab[j], fontname='Arial', fontsize=16)
        if yMax[k][0] < yMax[k][1]:
            axs[i, 0].set_ylim(bottom=yMax[k][0], top=yMax[k][1])

        axs[i, 0].yaxis.set_major_formatter(formattery)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        axs[i, 0].yaxis.set_major_formatter(formattery)
        # axs[i,0].axhline(color='darkgray',linestyle='dashed',alpha=0.5)
        axs[i, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
        Mea = np.nanmean(Hel[:, j])
        Std = np.nanstd(Hel[:, j])
        axs[i, 0].text(0.98, 0.98, '{:>7.2f} +/- {:>7.2f}'.format(Mea, Std),
                       transform=axs[i, 0].transAxes, ha='right', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
    if lDate:
        # Date
        YR = mdates.YearLocator()
        Mo = mdates.MonthLocator()
        axs[i, 0].xaxis.set_major_locator(YR)
        axs[i, 0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axs[i, 0].xaxis.set_minor_locator(Mo)
        axs[i, 0].xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    else:
        # MJD
        axs[i, 0].set_xlabel('Modified Julian Day',
                             fontname='Arial', fontsize=16)
    for tl in axs[i, 0].get_xticklabels(which='both'):
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    import argparse

    # cWhere='Local'
    # cWhere='GFZ'
    cWhere = 'HWC'
    if cWhere == 'Local':
        # Local mappings
        cWrkPre0 = r'Y:/'
        cDskPre0 = r'Z:/'
    elif cWhere == 'GFZ':
        # GFZ section cluster
        cWrkPre0 = r'/wrk/hanbing/'
        cDskPre0 = r'/dsk/hanbing/'
    elif cWhere == 'HWC':
        # Huawei cloud server
        cWrkPre0 = r'/home/hanbing/phb111/wrk/'
        cDskPre0 = r'/home/hanbing/phb111/dsk/'
    else:
        sys.exit('Unknow environment: '+cWhere)
    print('Run On '+cWhere)

    # InFilePrefix=r'D:/Code/PROJECT/WORK_Hel/'
    # InFilePrefix=r'Y:/IGS/TransToIGb14/'
    # InFilePrefix=r'Z:/PRO_2019001_2020366/D672/STA/StaHel/'
    InFilePrefix = os.path.join(cDskPre0, r'PRO_2019001_2020366/')

    # OutFilePrefix=r'D:/Code/PROJECT/WORK_Hel/'
    # OutFilePrefix=r'Z:/PRO_2019001_2020366/D672/STA/'
    OutFilePrefix = os.path.join(cDskPre0, r'PRO_2019001_2020366/D672/STA/')

    fList = glob.glob(InFilePrefix+'D672/STA/StaHel/Hel_???????')

    OutFileSuffix = 'Hel'
    PlotHel1(fList, [], False, ['D', 'TX', 'TY', 'TZ', 'RX', 'RY', 'RZ'], [[0.8, 2.0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]],
             False, OutFilePrefix, OutFileSuffix)
    # OutFileSuffix='Hel'
    # PlotHel1(fList,[],True,['D','TX','TY','TZ','RX','RY','RZ'],False,OutFilePrefix,OutFileSuffix)
