#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Plot some results from TRFs
'''
__author__ = 'hanbing'

# Standard library imports
import os
import sys
import os.path
import glob
import math
import datetime

# Related third party imports
import numpy as np
from numpy import linalg as NupLA
from scipy import linalg as SciLA
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# Local application/library specific imports
from PySRC.miscellaneous import GNSSTime
from PySRC.miscellaneous import CorSys


def PlotGeoCMotion(rYr1, rYr2, A, OutFilePrefix, OutFileSuffix):
    '''

    Plot the TRF seasonal geocenter motion model

  rYr1 --- Start year
  rYr2 --- End year
     A --- the annual and semi-annual amplitudes && phases of
           the seasonal geocenter motion
    '''

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 4))
    GeoC = ['X-comp', 'Y-comp', 'Z-comp']

    for i in range(3):
        X = [[], []]
        for t in np.linspace(rYr1, rYr2, int((rYr2 - rYr1) * 365.25)):
            d = A[i][0] * np.cos(2 * np.pi * t - np.pi / 180 * A[i][1]) + \
                A[i][2] * np.cos(4 * np.pi * t - np.pi / 180 * A[i][3])
            X[0].append(t)
            X[1].append(d)
        axs[0, 0].plot(X[0], X[1], '.-', label=GeoC[i], ms=2)

    axs[0, 0].legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                     framealpha=0.3, prop={'family': 'Arial', 'size': 14})
    axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
    axs[0, 0].set_axisbelow(True)

    axs[0, 0].set_ylabel('Seasonal Geocenter Motion [mm]', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    axs[0, 0].set_xlabel(r'Year', fontname='Arial', fontsize=16)
    axs[0, 0].xaxis.set_major_formatter('{x:7.2f}')
    for tl in axs[0, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix + OutFileSuffix + '.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotStaSeasonalSignals(fModel, lENU, lCM, rYr1, rYr2, cSta0, OutFilePrefix, OutFileSuffix):
    '''

    Plot the TRF seasonal signal model for specified stations

 fModel --- TRF seasonal signal model file, in SINEX format
   lENU --- Whether in ENU system, otherwise in XYZ system
    lCM --- Whether relative to CM, otherwise relative to CF
   rYr1 --- Start year
   rYr2 --- End year
  cSta0 --- list of specified stations, cSta0[0]='ALL' for all stations
    '''

    if lENU:
        GeoC = 'ENU'
    else:
        GeoC = 'XYZ'
    # 4-char station ID + PT code + SOLN
    Sig = [[]]
    # Two frequencies: annual and semi-annual signals
    for j in range(2):
        # Three components: X/Y/Z or E/N/U
        for k in range(3):
            # Four elements: a, sig_a, b, sig_b
            for i in range(4):
                Sig.append([])

    # Read the model for specified stations
    with open(fModel, mode='rt') as fOb:
        lBeg = False
        for cLine in fOb:
            if cLine[0:18] == '+SOLUTION/ESTIMATE':
                lBeg = True
            elif cLine[0:18] == '-SOLUTION/ESTIMATE':
                break
            elif lBeg and cLine[0:1] == ' ':
                cWords = cLine.split()
                if (cSta0[0] != 'ALL') and (cWords[2] not in cSta0):
                    continue
                # CODE+PT+SOLN
                cTmp = cLine[14:26]
                if cTmp not in Sig[0]:
                    # New station
                    Sig[0].append(cTmp)
                    # Two frequencies: annual and semi-annual signals
                    for j in range(2):
                        # Three components: X/Y/Z or E/N/U
                        for k in range(3):
                            # Four elements: a, sig_a, b, sig_b
                            for i in range(4):
                                Sig[1 + j * 12 + k * 4 + i].append(0.0)
                iSta = Sig[0].index(cTmp)
                # Frequency
                j = int(cWords[1][1:2]) - 1
                # Component
                k = GeoC.index(cWords[1][5:6])
                # Cos or Sin
                if cWords[1][2:5] == 'COS':
                    # m -> mm
                    Sig[1 + j * 12 + k * 4 + 0][iSta] = float(cWords[8]) * 1e3
                    Sig[1 + j * 12 + k * 4 + 1][iSta] = float(cWords[9]) * 1e3
                elif cWords[1][2:5] == 'SIN':
                    Sig[1 + j * 12 + k * 4 + 2][iSta] = float(cWords[8]) * 1e3
                    Sig[1 + j * 12 + k * 4 + 3][iSta] = float(cWords[9]) * 1e3
                else:
                    sys.exit('Unknown line: ' + cLine)
    nSta = len(Sig[0])

    strTmp = os.path.join(OutFilePrefix, OutFileSuffix + '.pdf')
    mpl.use('Agg')
    with PdfPages(strTmp) as pdf:
        for i in range(nSta):
            fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 4))
            for j in range(3):
                X = [[], []]
                for t in np.linspace(rYr1, rYr2, int((rYr2 - rYr1) * 365.25)):
                    d = Sig[1 + j * 4][i] * np.cos(2 * np.pi * t) + Sig[3 + j * 4][i] * np.sin(2 * np.pi * t) + \
                        Sig[13 + j * 4][i] * np.cos(4 * np.pi * t) + Sig[15 + j * 4][i] * np.sin(4 * np.pi * t)
                    X[0].append(t)
                    X[1].append(d)
                axs[0, 0].plot(X[0], X[1], '.-', label=GeoC[j:j + 1], ms=2)

            axs[0, 0].text(0.02, 0.98, Sig[0][i], transform=axs[0, 0].transAxes, ha='left', va='top',
                           fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
            axs[0, 0].legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                             framealpha=0.3, prop={'family': 'Arial', 'size': 14})
            axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
            axs[0, 0].set_axisbelow(True)
            if lCM:
                axs[0, 0].set_ylabel('Annual and semi-annual signals wrt CM [mm]', fontname='Arial', fontsize=16)
            else:
                axs[0, 0].set_ylabel('Annual and semi-annual signals wrt CF [mm]', fontname='Arial', fontsize=16)
            for tl in axs[0, 0].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)

            axs[0, 0].set_xlabel(r'Year', fontname='Arial', fontsize=16)
            axs[0, 0].xaxis.set_major_formatter('{x:7.2f}')
            for tl in axs[0, 0].get_xticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def PlotTieDis(fInput, cVec0, OutFilePrefix, OutFileSuffix):
    '''
    Plot TRF tie discrepancies.

    fInput --- the ITRF2020 tid discrepancy file
     cVec0 --- list of specified technique ties

    '''


    cTec=['GNSS', 'VLBI', 'SLR', 'DORIS']
    GeoC='ENU'
    cVec=[]
    TieDis=[]

    with open(fInput, mode='rt') as fOb:
        lBeg1=False
        lBeg2=False
        iVec=-1
        for cLine in fOb:
            cWords = cLine.split()
            if len(cWords) < 3:
                cWords.append('')
                cWords.append('')
            if (cWords[0] in cTec) and (cWords[1] in cTec) and (cWords[2] == 'Residuals'):
                if (cVec0[0] == 'ALL-ALL') or (cWords[0]+'-'+cWords[1] in cVec0) or (cWords[1]+'-'+cWords[0] in cVec0):
                    lBeg1=True
                    cVec.append(cWords[0]+'-'+cWords[1])
                    iVec=len(cVec)-1
                    TieDis.append([])
            elif lBeg1 and (not lBeg2) and (cLine[0:32] == '--------------------------------'):
                lBeg2=True
            elif lBeg1 and lBeg2 and (cLine[0:32] == '--------------------------------'):
                lBeg1=False
                lBeg2=False
                iVec=-1
            elif lBeg1 and lBeg2 and len(cLine) > 32:
                TieDis[iVec].append([])
                iTie=len(TieDis[iVec])-1
                # station vector
                TieDis[iVec][iTie].append(cWords[0]+'-'+cWords[3])
                # East, mm
                TieDis[iVec][iTie].append(float(cWords[6]))
                # North, mm
                TieDis[iVec][iTie].append(float(cWords[7]))
                # Up, mm
                TieDis[iVec][iTie].append(float(cWords[8]))
    nVec=len(cVec)

    strTmp = os.path.join(OutFilePrefix, OutFileSuffix + '.pdf')
    mpl.use('Agg')
    with PdfPages(strTmp) as pdf:
        # Number of technique tie discrepancy types
        for i in range(nVec):
            # Number of tie vectors
            nTie=len(TieDis[i])
            cTie=[]
            fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(nTie*0.6, 4))

            x=np.arange(nTie)
            # the width of bars
            w=1/(3+1)
            for j in range(3):
                y=np.zeros(nTie)
                for k in range(nTie):
                    y[k]=TieDis[i][k][1+j]
                    if j==0:
                        cTie.append(TieDis[i][k][0])
                axs[0, 0].bar(x + (j - 3 / 2) * w, y, w, align='edge', label=GeoC[j:j+1])

            axs[0, 0].text(0.02, 0.98, '{: >3d} tie vectors'.format(nTie), transform=axs[0, 0].transAxes, ha='left', va='top',
                           fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})

            axs[0, 0].legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                             prop={'family': 'Arial', 'size': 14})
            axs[0, 0].grid(which='both', axis='y', color='darkgray', linestyle='--',
                           linewidth=0.8)
            axs[0, 0].set_axisbelow(True)
            axs[0, 0].yaxis.set_major_formatter('{x: >5.1f}')
            axs[0, 0].set_ylabel(cVec[i] + ' Tie Discrepancies [mm]', fontname='Arial', fontsize=16)
            for tl in axs[0, 0].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            axs[0, 0].set_xlabel('Tie Vectors', fontname='Arial', fontsize=16)
            axs[0, 0].set_xlim(left=-1, right=nTie)
            axs[0, 0].set_xticks(x)
            axs[0, 0].set_xticklabels(cTie, rotation=90, fontdict={'fontsize': 14, 'fontname': 'monospace'})

            pdf.savefig(fig, bbox_inches='tight')
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
        sys.exit('Unknow environment: ' + cWhere)
    print('Run On ' + cWhere)

    # OutFilePrefix = os.path.join(cDskPre0, r'SISRE_Test/')
    OutFilePrefix = r'D:/Code/PROJECT/WORK_TRF/'

    # Gencenter motion model for ITRF2020
    A = [[1.23, -123.2, 0.49, 107.2],
         [3.48, 152.9, 0.22, 1.6],
         [2.76, -139.5, 1.19, 30.5]]
    # OutFileSuffix = 'ITRF2020_GeocenterMotion'
    # PlotGeoCMotion(2020, 2022, A, OutFilePrefix, OutFileSuffix)

    # fModel = r'D:/Code/PROJECT/WORK_TRF/ITRF2020-Frequencies-ENU-CM.snx'
    # OutFileSuffix = 'ITRF2020_Annual_Semi_Annual_Sig_CM_ENU'
    # PlotStaSeasonalSignals(fModel, True, True, 2020, 2022, ['ALL'], OutFilePrefix, OutFileSuffix)

    fInput = r'D:/Code/PROJECT/WORK_TRF/ITRF2020-Tie-Residuals.dat'
    OutFileSuffix = 'ITRF2020_TieDiscrepancy'
    PlotTieDis(fInput, ['ALL-ALL'], OutFilePrefix, OutFileSuffix)
