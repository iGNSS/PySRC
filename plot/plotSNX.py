#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Plot some results from SINEX files
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


def GetCRD(fList):
    '''
    Get station CRD from SINEX files
    '''

    # Get the whole site list
    cSta = []
    for i in range(len(fList)):
        lBeg = False
        with open(fList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:16] == '+SOLUTION/EPOCHS':
                    lBeg = True
                    continue
                if cLine[0:16] == '-SOLUTION/EPOCHS':
                    break
                if not lBeg:
                    continue
                if cLine[0:1] != ' ':
                    continue
                # Site Code + Point Code + Solution ID
                cTmp = cLine[1:5]+cLine[6:8]+cLine[9:13]
                if cTmp not in cSta:
                    cSta.append(cTmp)
    cSta.sort()
    # Read the CRD estimates
    CRD = []
    for i in range(len(cSta)):
        CRD.append([])
        for j in range(9):
            CRD[i].append([])

    for i in range(len(fList)):
        lBeg = False
        with open(fList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:18] == '+SOLUTION/ESTIMATE':
                    lBeg = True
                    continue
                if cLine[0:18] == '-SOLUTION/ESTIMATE':
                    break
                if not lBeg:
                    continue
                if cLine[0:1] != ' ':
                    continue
                if cLine[7:11] == 'STAX' or cLine[9:13] == 'STAX':
                    rEpo = GNSSTime.snx2mjd(cLine[27:39])
                    x = float(cLine[47:68])
                    sx = float(cLine[69:80])
                    # Site Code + Point Code + Solution ID
                    cTmp = cLine[14:18]+cLine[19:21]+cLine[22:26]
                    if cTmp not in cSta:
                        print('Not found site '+cTmp+' in '+fList[i])
                    else:
                        j = cSta.index(cTmp)
                        CRD[j][0].append(rEpo)
                        CRD[j][1].append(x)
                        CRD[j][2].append(sx)
                if cLine[7:11] == 'STAY' or cLine[9:13] == 'STAY':
                    rEpo = GNSSTime.snx2mjd(cLine[27:39])
                    x = float(cLine[47:68])
                    sx = float(cLine[69:80])
                    # Site Code + Point Code + Solution ID
                    cTmp = cLine[14:18]+cLine[19:21]+cLine[22:26]
                    if cTmp not in cSta:
                        print('Not found site '+cTmp+' in '+fList[i])
                    else:
                        j = cSta.index(cTmp)
                        CRD[j][3].append(rEpo)
                        CRD[j][4].append(x)
                        CRD[j][5].append(sx)
                if cLine[7:11] == 'STAZ' or cLine[9:13] == 'STAZ':
                    rEpo = GNSSTime.snx2mjd(cLine[27:39])
                    x = float(cLine[47:68])
                    sx = float(cLine[69:80])
                    # Site Code + Point Code + Solution ID
                    cTmp = cLine[14:18]+cLine[19:21]+cLine[22:26]
                    if cTmp not in cSta:
                        print('Not found site '+cTmp+' in '+fList[i])
                    else:
                        j = cSta.index(cTmp)
                        CRD[j][6].append(rEpo)
                        CRD[j][7].append(x)
                        CRD[j][8].append(sx)
    # Do some check
    for i in range(len(cSta)):
        nEpo = len(CRD[i][0])
        if len(CRD[i][3]) != nEpo:
            sys.exit('Different dimesion of XYZ for '+cSta[i])
        if len(CRD[i][6]) != nEpo:
            sys.exit('Different dimesion of XYZ for '+cSta[i])
        for j in range(nEpo):
            if CRD[i][0][j] != CRD[i][3][j] or CRD[i][0][j] != CRD[i][6][j]:
                sys.exit('Different epoch of XYZ for '+cSta[i])

    return cSta, CRD


def PlotCRDRes(fList, OutFilePrefix, OutFileSuffix):
    '''
    Plot Station CRD residuals wrt its mean position
    '''

    cSta, CRD = GetCRD(fList)
    nSta = len(cSta)

    strTmp = OutFilePrefix+OutFileSuffix
    with PdfPages(strTmp) as pdf:
        fig, axs = plt.subplots(nSta, 1, sharex='col',
                                squeeze=False, figsize=(12, nSta*2))
        fig.subplots_adjust(hspace=0.1)
        XYZ = np.zeros(3)
        BLH = np.zeros(3)
        for i in range(nSta):
            nEpo = len(CRD[i][0])
            dXYZ = np.zeros((nEpo, 3))
            dENU = np.zeros((nEpo, 3))

            cTmp = cSta[i]+'              '
            # STAX
            XYZ[0] = np.mean(CRD[i][1])
            dXYZ[:, 0] = CRD[i][1]-XYZ[0]
            # STAY
            XYZ[1] = np.mean(CRD[i][4])
            dXYZ[:, 1] = CRD[i][4]-XYZ[1]
            # STAZ
            XYZ[2] = np.mean(CRD[i][7])
            dXYZ[:, 2] = CRD[i][7]-XYZ[2]
            BLH[0], BLH[1], BLH[2] = CorSys.XYZ2BLH(XYZ[0], XYZ[1], XYZ[2])
            R = CorSys.RotENU2TRS(BLH[0], BLH[1])
            dENU = np.dot(dXYZ, R)

            cTmp = cTmp + \
                '  {:>15.2f} {:>15.2f} {:>15.2f}'.format(
                    XYZ[0], XYZ[1], XYZ[2])
            print(cTmp)

            # Cal rms
            RMS = np.zeros(3)
            for j in range(nEpo):
                RMS[0] = RMS[0]+dENU[j, 0]*dENU[j, 0]
                RMS[1] = RMS[1]+dENU[j, 1]*dENU[j, 1]
                RMS[2] = RMS[2]+dENU[j, 2]*dENU[j, 2]
                # #for debug
                # cTmp='{:>15.2f} {:>15.2f} {:>15.2f} {:>5.2f} {:>5.2f} {:>5.2f} {:>5.2f} {:>5.2f} {:>5.2f}'.format(XYZ[0],XYZ[1],XYZ[2],dXYZ[j,0],dXYZ[j,1],dXYZ[j,2],dENU[j,0],dENU[j,1],dENU[j,2])
                # print(cTmp)
                # #for debug
            RMS[0] = np.sqrt(RMS[0]/nEpo)
            RMS[1] = np.sqrt(RMS[1]/nEpo)
            RMS[2] = np.sqrt(RMS[2]/nEpo)

            axs[i, 0].plot(CRD[i][0], dENU[:, 0], '.r', label='dE')
            axs[i, 0].plot(CRD[i][3], dENU[:, 1], '.g', label='dN')
            axs[i, 0].plot(CRD[i][6], dENU[:, 2], '.b', label='dU')

            axs[i, 0].text(0.05, 0.95, cSta[i], transform=axs[i, 0].transAxes,
                           ha='left', va='top')
            cTmp = '{:>5.2f} {:>5.2f} {:>5.2f}'.format(RMS[0], RMS[1], RMS[2])
            axs[i, 0].text(0.95, 0.05, cTmp, transform=axs[i, 0].transAxes,
                           ha='right', va='bottom')

            axs[i, 0].legend(ncol=3, loc='upper right', framealpha=0.5)
        axs[i, 0].set_xlabel('Modified Julian Day')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def PlotParCorr11(fSNX, cSVNList, cParList, OutFilePrefix, OutFileSuffix):
    '''
    Plot correlations within specific parameters for each satellite in the
    specified satellite list. Each satellite takes an axis.

    Correlation matrix is read from the
    "+/-SOLUTION/MATRIX_ESTIMATE L CORR" block in provided SINEX file

        fSNX --- SINEX file
    cSVNList --- Satellite SVN list
    cParList --- Specific parameters list of each satellite in
                 the specified SVN list. These should be satellite-dependent
                 parameters.
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # Get the parameters table for each required satellite
    cSat = []
    cPar = []
    iPar = []
    # The total number of parameters
    nParSNX = 0
    with open(fSNX, mode='rt') as fOb:
        lBeg = False
        for cLine in fOb:
            if cLine[0:18] == '+SOLUTION/ESTIMATE':
                lBeg = True
            elif cLine[0:18] == '-SOLUTION/ESTIMATE':
                break
            elif lBeg:
                cWords = cLine.split()
                nParSNX = nParSNX+1
                if cWords[1] not in cParList:
                    continue
                if cSVNList[0] != 'ALL' and cWords[2] not in cSVNList:
                    continue
                if cWords[2] not in cSat:
                    cSat.append(cWords[2])
                    cPar.append([])
                    iPar.append([])
                i = cSat.index(cWords[2])
                cPar[i].append(cWords[1])
                iPar[i].append(int(cWords[0]))
    nSat = len(cSat)
    cSat0 = cSat.copy()
    cSat0.sort()
    if nSat == 0:
        sys.exit('No required satellite found!')

    # Get the full correlation matrix
    R = np.zeros((nParSNX, nParSNX))
    with open(fSNX, mode='rt') as fOb:
        lBeg = False
        for cLine in fOb:
            if cLine[0:32] == '+SOLUTION/MATRIX_ESTIMATE L CORR':
                lBeg = True
            elif cLine[0:32] == '-SOLUTION/MATRIX_ESTIMATE L CORR':
                break
            elif lBeg:
                cWords = cLine.split()
                iRow = int(cWords[0])
                iCol = int(cWords[1])
                for i in range(min(3, iRow-iCol+1)):
                    # lower-triangular part
                    if iRow-1 == iCol+i-1:
                        R[iRow-1, iCol+i-1] = 1
                    else:
                        R[iRow-1, iCol+i-1] = float(cWords[i+2])
                    # upper-triangular part
                    R[iCol+i-1, iRow-1] = R[iRow-1, iCol+i-1]

    fig, axs = plt.subplots(nSat, 1, squeeze=False, figsize=(8, 7*nSat))
    # fig.subplots_adjust(hspace=0.2)

    for i in range(nSat):
        iSat = cSat.index(cSat0[i])
        # Number of parameters for this satellite
        nPar0 = len(cPar[iSat])
        cPar0 = []
        # Correlation matrix of this satellite
        R0 = np.zeros((nPar0, nPar0))
        for j in range(nPar0):
            if cPar[iSat][j][0:5] == 'BOXW_':
                cPar0.append(cPar[iSat][j][5:])
            elif cPar[iSat][j] == 'Kd_BERN':
                cPar0.append('D0')
            elif cPar[iSat][j] == 'Ky_BERN':
                cPar0.append('Y0')
            elif cPar[iSat][j] == 'Kb_BERN':
                cPar0.append('B0')
            elif cPar[iSat][j] == 'Kbc1_BERN':
                cPar0.append('Bc1')
            elif cPar[iSat][j] == 'Kbs1_BERN':
                cPar0.append('Bs1')
            elif cPar[iSat][j] == 'Kdc2_BERN':
                cPar0.append('Dc2')
            elif cPar[iSat][j] == 'Kds2_BERN':
                cPar0.append('Ds2')
            elif cPar[iSat][j] == 'Kdc4_BERN':
                cPar0.append('Dc4')
            elif cPar[iSat][j] == 'Kds4_BERN':
                cPar0.append('Ds4')
            elif cPar[iSat][j] == 'GEOCX0':
                cPar0.append('Xg')
            elif cPar[iSat][j] == 'GEOCY0':
                cPar0.append('Yg')
            elif cPar[iSat][j] == 'GEOCZ0':
                cPar0.append('Zg')
            else:
                cPar0.append(cPar[iSat][j])
            for k in range(nPar0):
                R0[j, k] = np.abs(R[iPar[iSat][j]-1, iPar[iSat][k]-1])

        x = np.arange(-0.5, nPar0, 1)
        qm = axs[i, 0].pcolormesh(x, x, R0, cmap='Greys', vmin=0, vmax=1)
        axs[i, 0].set_xticks(range(len(cPar0)))
        axs[i, 0].set_xticklabels(cPar0, rotation=90)
        for tl in axs[i, 0].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(10)
        axs[i, 0].set_yticks(range(len(cPar0)))
        axs[i, 0].set_yticklabels(cPar0)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(10)
        axs[i, 0].invert_yaxis()
        # axs[i,0].xaxis.set_ticks_position('top')
        cbar = fig.colorbar(qm, ax=axs[i, 0])
        cbar.set_label(cSat0[i], loc='center', fontname='Arial', fontsize=16)
        for tl in cbar.ax.get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotParCorr12(fSNX, cSVNList, cParList1, cParList2, OutFilePrefix, OutFileSuffix):
    '''
    Plot correlations between two specific parameter sets for each satellite in the
    specified satellite list. Among the two parameter sets, one set are sat-dependent
    parameters while the other set are common parameters, like EOP, GCC et al. Each
    satellite takes an axis.

        fSNX --- SINEX file
    cSVNList --- Satellite SVN list
   cParList1 --- Specific parameters list 1, satellite-dependent ones
   cParList2 --- Specific parameters list 2, satellite-independent ones
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # Get the parameters table for each required satellite
    cSat = []
    cPar1 = []
    cPar2 = []
    iPar1 = []
    iPar2 = []
    nParSNX = 0
    with open(fSNX, mode='rt') as fOb:
        lBeg = False
        for cLine in fOb:
            if cLine[0:18] == '+SOLUTION/ESTIMATE':
                lBeg = True
            elif cLine[0:18] == '-SOLUTION/ESTIMATE':
                break
            elif lBeg:
                cWords = cLine.split()
                # The total number of parameters
                nParSNX = nParSNX+1
                if cWords[2] not in cSVNList:
                    continue
                if cWords[2] not in cSat:
                    cSat.append(cWords[2])
                    cPar1.append([])
                    iPar1.append([])
                    cPar2.append([])
                    iPar2.append([])
                i = cSat.index(cWords[2])
                # For satellite-dependent par
                if cWords[1] in cParList1:
                    cPar1[i].append(cWords[1])
                    iPar1[i].append(int(cWords[0]))
        fOb.seek(0)
        # For the commom par set
        lBeg = False
        for cLine in fOb:
            if cLine[0:18] == '+SOLUTION/ESTIMATE':
                lBeg = True
            elif cLine[0:18] == '-SOLUTION/ESTIMATE':
                break
            elif lBeg:
                cWords = cLine.split()
                if cWords[1] not in cParList2:
                    continue
                # Put in the first satellite
                cPar2[0].append(cWords[1])
                iPar2[0].append(int(cWords[0]))
        for i in range(1, len(cSat)):
            cPar2[i] = cPar2[0]
            iPar2[i] = iPar2[0]
    nSat = len(cSat)
    cSat0 = cSat.copy()
    cSat0.sort()
    if nSat == 0:
        sys.exit('No required satellite found!')

    # Get the full correlation matrix
    R = np.zeros((nParSNX, nParSNX))
    with open(fSNX, mode='rt') as fOb:
        lBeg = False
        for cLine in fOb:
            if cLine[0:32] == '+SOLUTION/MATRIX_ESTIMATE L CORR':
                lBeg = True
            elif cLine[0:32] == '-SOLUTION/MATRIX_ESTIMATE L CORR':
                break
            elif lBeg:
                cWords = cLine.split()
                iRow = int(cWords[0])
                iCol = int(cWords[1])
                for i in range(min(3, iRow-iCol+1)):
                    # lower-triangular part
                    if iRow-1 == iCol+i-1:
                        R[iRow-1, iCol+i-1] = 1
                    else:
                        R[iRow-1, iCol+i-1] = float(cWords[i+2])
                    # upper-triangular part
                    R[iCol+i-1, iRow-1] = R[iRow-1, iCol+i-1]

    fig, axs = plt.subplots(nSat, 1, squeeze=False, figsize=(3, 4*nSat))
    # fig.subplots_adjust(hspace=0.2)

    for i in range(nSat):
        iSat = cSat.index(cSat0[i])
        nPar1 = len(cPar1[iSat])
        nPar2 = len(cPar2[iSat])
        for j in range(nPar1):
            if cPar1[iSat][j][0:5] == 'BOXW_':
                cPar1[iSat][j] = cPar1[iSat][j][5:]
            elif cPar1[iSat][j] == 'Kd_BERN':
                cPar1[iSat][j] = 'D0'
            elif cPar1[iSat][j] == 'Ky_BERN':
                cPar1[iSat][j] = 'Y0'
            elif cPar1[iSat][j] == 'Kb_BERN':
                cPar1[iSat][j] = 'B0'
            elif cPar1[iSat][j] == 'Kbc1_BERN':
                cPar1[iSat][j] = 'Bc1'
            elif cPar1[iSat][j] == 'Kbs1_BERN':
                cPar1[iSat][j] = 'Bs1'
            elif cPar1[iSat][j] == 'Kdc2_BERN':
                cPar1[iSat][j] = 'Dc2'
            elif cPar1[iSat][j] == 'Kds2_BERN':
                cPar1[iSat][j] = 'Ds2'
            elif cPar1[iSat][j] == 'Kdc4_BERN':
                cPar1[iSat][j] = 'Dc2'
            elif cPar1[iSat][j] == 'Kds4_BERN':
                cPar1[iSat][j] = 'Ds2'
            elif cPar1[iSat][j] == 'GEOCX0':
                cPar1[iSat][j] = 'Xg'
            elif cPar1[iSat][j] == 'GEOCY0':
                cPar1[iSat][j] = 'Yg'
            elif cPar1[iSat][j] == 'GEOCZ0':
                cPar1[iSat][j] = 'Zg'
        for j in range(nPar2):
            if cPar2[iSat][j][0:5] == 'BOXW_':
                cPar2[iSat][j] = cPar2[iSat][j][5:]
            elif cPar2[iSat][j] == 'Kd_BERN':
                cPar2[iSat][j] = 'D0'
            elif cPar2[iSat][j] == 'Ky_BERN':
                cPar2[iSat][j] = 'Y0'
            elif cPar2[iSat][j] == 'Kb_BERN':
                cPar2[iSat][j] = 'B0'
            elif cPar2[iSat][j] == 'Kbc1_BERN':
                cPar2[iSat][j] = 'Bc1'
            elif cPar2[iSat][j] == 'Kbs1_BERN':
                cPar2[iSat][j] = 'Bs1'
            elif cPar2[iSat][j] == 'Kdc2_BERN':
                cPar2[iSat][j] = 'Dc2'
            elif cPar2[iSat][j] == 'Kds2_BERN':
                cPar2[iSat][j] = 'Ds2'
            elif cPar2[iSat][j] == 'Kdc4_BERN':
                cPar2[iSat][j] = 'Dc4'
            elif cPar2[iSat][j] == 'Kds4_BERN':
                cPar2[iSat][j] = 'Ds4'
            elif cPar2[iSat][j] == 'GEOCX0':
                cPar2[iSat][j] = 'Xg'
            elif cPar2[iSat][j] == 'GEOCY0':
                cPar2[iSat][j] = 'Yg'
            elif cPar2[iSat][j] == 'GEOCZ0':
                cPar2[iSat][j] = 'Zg'

        # Correlation matrix of this satellite
        R0 = np.zeros((nPar1, nPar2))
        for j in range(nPar1):
            for k in range(nPar2):
                R0[j, k] = np.abs(R[iPar1[iSat][j]-1, iPar2[iSat][k]-1])

        x = np.arange(-0.5, nPar2, 1)
        y = np.arange(-0.5, nPar1, 1)
        qm = axs[i, 0].pcolormesh(x, y, R0, cmap='Greys', vmin=0, vmax=1)
        axs[i, 0].set_xticks(range(nPar2))
        axs[i, 0].set_xticklabels(cPar2[iSat], fontdict={
                                  'fontsize': 14, 'fontname': 'Arial'})
        axs[i, 0].set_yticks(range(nPar1))
        axs[i, 0].set_yticklabels(cPar1[iSat], fontdict={
                                  'fontsize': 14, 'fontname': 'Arial'})
        axs[i, 0].invert_yaxis()
        # axs[i,0].xaxis.set_ticks_position('top')
        cbar = fig.colorbar(qm, ax=axs[i, 0])
        cbar.set_label(cSat0[i], loc='center', fontname='Arial', fontsize=16)
        for tl in cbar.ax.get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotParCorr13(fSNX, cSVNList, cParList, OutFilePrefix, OutFileSuffix):
    '''
    Plot correlations among all specified parameters related to specific satellites
    in a single axis.

    Correlation matrix is read from the
    "+/-SOLUTION/MATRIX_ESTIMATE L CORR" block in provided SINEX file

        fSNX --- SINEX file
    cSVNList --- SVN lists of satellites involved in the correlation matrix
    cParList --- List of specific parameters of the satellites
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # Get the parameters table
    cSat = []
    cPar = []
    iPar = []
    # The total number of parameters
    nParSNX = 0
    with open(fSNX, mode='rt') as fOb:
        lBeg = False
        for cLine in fOb:
            if cLine[0:18] == '+SOLUTION/ESTIMATE':
                lBeg = True
            elif cLine[0:18] == '-SOLUTION/ESTIMATE':
                break
            elif lBeg:
                cWords = cLine.split()
                nParSNX = nParSNX+1
                if cWords[1] not in cParList:
                    continue
                if len(cLine[38:42]) != 0 and cSVNList[0] != 'ALL' and cWords[2] not in cSVNList:
                    continue
                cPar.append(cWords[1])
                iPar.append(int(cWords[0]))
                # Satellite or station for this parameter
                cSat.append(cLine[38:42])
    nPar = len(cPar)
    if nPar == 0:
        sys.exit('No required par found!')

    # Get the full correlation matrix
    R = np.zeros((nParSNX, nParSNX))
    with open(fSNX, mode='rt') as fOb:
        lBeg = False
        for cLine in fOb:
            if cLine[0:32] == '+SOLUTION/MATRIX_ESTIMATE L CORR':
                lBeg = True
            elif cLine[0:32] == '-SOLUTION/MATRIX_ESTIMATE L CORR':
                break
            elif lBeg:
                cWords = cLine.split()
                iRow = int(cWords[0])
                iCol = int(cWords[1])
                for i in range(min(3, iRow-iCol+1)):
                    # lower-triangular part
                    if iRow-1 == iCol+i-1:
                        R[iRow-1, iCol+i-1] = 1
                    else:
                        R[iRow-1, iCol+i-1] = float(cWords[i+2])
                    # upper-triangular part
                    R[iCol+i-1, iRow-1] = R[iRow-1, iCol+i-1]

    fig, axs = plt.subplots(1, 1, squeeze=False,
                            figsize=(nPar*0.30, nPar*0.25))

    # Correlation matrix of required parameters
    R0 = np.zeros((nPar, nPar))
    cPar0 = []
    for j in range(nPar):
        if cPar[j][0:5] == 'BOXW_':
            cPar0.append(cPar[j][5:]+'-'+cSat[j])
        elif cPar[j] == 'Kd_BERN':
            cPar0.append('D0'+'-'+cSat[j])
        elif cPar[j] == 'Ky_BERN':
            cPar0.append('Y0'+'-'+cSat[j])
        elif cPar[j] == 'Kb_BERN':
            cPar0.append('B0'+'-'+cSat[j])
        elif cPar[j] == 'Kbc1_BERN':
            cPar0.append('Bc1'+'-'+cSat[j])
        elif cPar[j] == 'Kbs1_BERN':
            cPar0.append('Bs1'+'-'+cSat[j])
        elif cPar[j] == 'Kdc2_BERN':
            cPar0.append('Dc2'+'-'+cSat[j])
        elif cPar[j] == 'Kds2_BERN':
            cPar0.append('Ds2'+'-'+cSat[j])
        elif cPar[j] == 'Kdc4_BERN':
            cPar0.append('Dc4'+'-'+cSat[j])
        elif cPar[j] == 'Kds4_BERN':
            cPar0.append('Ds4'+'-'+cSat[j])
        elif cPar[j] == 'GEOCX0':
            cPar0.append('Xg'+'-'+cSat[j])
        elif cPar[j] == 'GEOCY0':
            cPar0.append('Yg'+'-'+cSat[j])
        elif cPar[j] == 'GEOCZ0':
            cPar0.append('Zg'+'-'+cSat[j])
        elif cPar[j] == 'PXSAT':
            cPar0.append('PX'+'-'+cSat[j])
        elif cPar[j] == 'PYSAT':
            cPar0.append('PY'+'-'+cSat[j])
        elif cPar[j] == 'PZSAT':
            cPar0.append('PZ'+'-'+cSat[j])
        elif cPar[j] == 'VXSAT':
            cPar0.append('VX'+'-'+cSat[j])
        elif cPar[j] == 'VYSAT':
            cPar0.append('VY'+'-'+cSat[j])
        elif cPar[j] == 'VZSAT':
            cPar0.append('VZ'+'-'+cSat[j])
        else:
            cPar0.append(cPar[j]+'-'+cSat[j])
        for k in range(nPar):
            R0[j, k] = np.abs(R[iPar[j]-1, iPar[k]-1])

    x = np.arange(-0.5, nPar, 1)
    qm = axs[0, 0].pcolormesh(x, x, R0, cmap='Greys', vmin=0, vmax=1)
    axs[0, 0].set_xticks(range(len(cPar0)))
    axs[0, 0].set_xticklabels(cPar0, rotation=90)
    for tl in axs[0, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    axs[0, 0].set_yticks(range(len(cPar0)))
    axs[0, 0].set_yticklabels(cPar0)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    axs[0, 0].invert_yaxis()
    # axs[0,0].xaxis.set_ticks_position('top')
    cbar = fig.colorbar(qm, ax=axs[0, 0])
    cbar.set_label('', loc='center', fontname='Arial', fontsize=16)
    for tl in cbar.ax.get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotParCorr20(fSNXList, cSVNList, cParList, OutFilePrefix, OutFileSuffix):
    '''
    Plot the time series of correlation coefficients between
    specific parameters related to specific satellites.

    fSNXList --- SINEX files list
    cSVNList --- the specific satellite SVN
    cParList --- Specific parameters list for satellites

    NOTE: Here we assume that the parameters list does not change
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    strTmp = os.path.join(OutFilePrefix, OutFileSuffix+'.pdf')
    with PdfPages(strTmp) as pdf:
        for cSVN in cSVNList:
            # Get the actual parameters table for this satellite
            cPar = []
            for i in range(len(fSNXList)):
                with open(fSNXList[i], mode='rt') as fOb:
                    lBeg = False
                    for cLine in fOb:
                        if cLine[0:18] == '+SOLUTION/ESTIMATE':
                            lBeg = True
                        elif cLine[0:18] == '-SOLUTION/ESTIMATE':
                            break
                        elif lBeg:
                            cWords = cLine.split()
                            if (cWords[2] != cSVN) or (cWords[1] not in cParList):
                                continue
                            cPar.append(cWords[1])
                # As the parameter list does not change,
                # we only need to read one file
                if len(cPar) > 0:
                    break
            nPar = len(cPar)
            if nPar < 2:
                print('No enought par found for '+cSVN, nPar)
                continue
            R = []
            rEpo = []
            for i in range(nPar-1):
                R.append([])
                for j in range(i+1):
                    R[i].append([])
            for i in range(len(fSNXList)):
                # Get the date from the file name
                WK = int(fSNXList[i][-9:-5])
                WKD = int(fSNXList[i][-5:-4])
                YYYY, MO, DD = GNSSTime.wkd2dom(WK, WKD)
                # Get the correlation matrix for this satellite
                # 1) Get the local index
                iPar = np.zeros(nPar, dtype=np.uint32)
                # Whether the required satellite found
                lFound = False
                with open(fSNXList[i], mode='rt') as fOb:
                    lBeg = False
                    for cLine in fOb:
                        if cLine[0:18] == '+SOLUTION/ESTIMATE':
                            lBeg = True
                        elif cLine[0:18] == '-SOLUTION/ESTIMATE':
                            break
                        elif lBeg:
                            cWords = cLine.split()
                            if (cWords[2] != cSVN) or (cWords[1] not in cPar):
                                # Not required satellite or parameters
                                continue
                            iPar[cPar.index(cWords[1])] = int(cWords[0])
                            lFound = True
                if not lFound:
                    continue
                rEpo.append(datetime.datetime(YYYY, MO, DD))
                # Extract the correlation matrix of this satellite
                with open(fSNXList[i], mode='rt') as fOb:
                    lBeg = False
                    for cLine in fOb:
                        if cLine[0:32] == '+SOLUTION/MATRIX_ESTIMATE L CORR':
                            lBeg = True
                        elif cLine[0:32] == '-SOLUTION/MATRIX_ESTIMATE L CORR':
                            break
                        elif lBeg:
                            cWords = cLine.split()
                            iRow0 = int(cWords[0])
                            if iRow0 not in iPar:
                                continue
                            iCol0 = int(cWords[1])
                            for i in range(min(3, iRow0-iCol0+1)):
                                if iCol0+i not in iPar:
                                    continue
                                if iRow0 == iCol0+i:
                                    continue
                                # Global index
                                for j in range(nPar):
                                    if iPar[j] == iRow0:
                                        iRow = j
                                    elif iPar[j] == iCol0+i:
                                        iCol = j
                                R[iRow-1][iCol].append(float(cWords[i+2]))

            fig, axs = plt.subplots(nPar-1, nPar-1, squeeze=False, sharex='col', sharey='row',
                                    figsize=((nPar-1)*2, (nPar-1)*2))
            fig.suptitle(cSVN, x=0.5, y=0.7, fontfamily='Arial',
                         fontsize=18, fontweight='bold')
            # fig.subplots_adjust(hspace=0.1)
            # fig.subplots_adjust(wspace=0.1)
            for i in range(nPar):
                if cPar[i][0:5] == 'BOXW_':
                    cPar[i] = cPar[i][5:]
                elif cPar[i] == 'Kd_BERN':
                    cPar[i] = 'D0'
                elif cPar[i] == 'Ky_BERN':
                    cPar[i] = 'Y0'
                elif cPar[i] == 'Kb_BERN':
                    cPar[i] = 'B0'
                elif cPar[i] == 'Kbc1_BERN':
                    cPar[i] = 'Bc1'
                elif cPar[i] == 'Kbs1_BERN':
                    cPar[i] = 'Bs1'
                elif cPar[i] == 'Kdc2_BERN':
                    cPar[i] = 'Dc2'
                elif cPar[i] == 'Kds2_BERN':
                    cPar[i] = 'Ds2'
                elif cPar[i] == 'Kdc4_BERN':
                    cPar[i] = 'Dc4'
                elif cPar[i] == 'Kds4_BERN':
                    cPar[i] = 'Ds4'
                elif cPar[i] == 'GEOCX0':
                    cPar[i] = 'Xg'
                elif cPar[i] == 'GEOCY0':
                    cPar[i] = 'Yg'
                elif cPar[i] == 'GEOCZ0':
                    cPar[i] = 'Zg'
                elif cPar[i] == 'PXSAT':
                    cPar[i] = 'PX'
                elif cPar[i] == 'PYSAT':
                    cPar[i] = 'PY'
                elif cPar[i] == 'PZSAT':
                    cPar[i] = 'PZ'
                elif cPar[i] == 'VXSAT':
                    cPar[i] = 'VX'
                elif cPar[i] == 'VYSAT':
                    cPar[i] = 'VY'
                elif cPar[i] == 'VZSAT':
                    cPar[i] = 'VZ'

            for i in range(nPar-1):
                axs[i, 0].set_ylabel(cPar[i+1], fontname='Arial', fontsize=16)
                for tl in axs[i, 0].get_yticklabels():
                    tl.set_fontname('Arial')
                    tl.set_fontsize(14)
                for j in range(i+1):
                    axs[i, j].plot(rEpo, R[i][j], '.', ms=4)
                    axs[i, j].axhline(color='darkgray',
                                      linestyle='dashed', alpha=0.5)
                    axs[i, j].set_ylim(bottom=-1, top=1)
                    axs[i, j].set_xticklabels([])
                    if i == (nPar-2):
                        axs[i, j].set_xlabel(
                            cPar[j], fontname='Arial', fontsize=16)
                # Close the other axies
                for j in range(i+1, nPar-1):
                    axs[i, j].set_axis_off()
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def PlotParCorr21(cSer, fPathSer, cSVN, cPar1, cPar2, MJD0, nDay, OutFilePrefix, OutFileSuffix):
    '''
    Plot the comparison of the correlation between two parameters among
    different solutions for specific satellites during a session

        cSer --- list of to-be-compared seri
    fPathSer --- path list for different solutions
        cSVN --- the specific satellite SVNs, each satellite takes an axis
       cPar1 --- Specific parameter 1, a satellite-dependent one
       cPar2 --- Specific parameter 2, a satellite-independent one
        MJD0 --- start MJD of the session
        nDay --- number of days of the session

    NOTE: Here we assume that the parameter lists of each solution during the session
          does not change.
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nSer = len(cSer)
    nSat = len(cSVN)
    rCor = np.zeros((nDay, nSat*nSer))
    rCor[:, :] = np.nan
    for iDay in range(nDay):
        MJD = MJD0+iDay
        YYYY, DOY = GNSSTime.mjd2doy(MJD)
        WK, WKD = GNSSTime.mjd2wkd(MJD)
        for iSer in range(nSer):
            fSNX = os.path.join(fPathSer[iSer], 'WORK{:4d}{:03d}'.format(YYYY, DOY),
                                'phb{:04d}{:01d}.snx'.format(WK, WKD))
            # Global index of satellite-dependent par and the satellite-independent par
            iPar1 = np.zeros(nSat, dtype=np.int64)
            iPar2 = 0
            with open(fSNX, mode='rt') as fOb:
                lBeg = False
                nParSNX = 0
                for cLine in fOb:
                    if cLine[0:18] == '+SOLUTION/ESTIMATE':
                        lBeg = True
                    elif cLine[0:18] == '-SOLUTION/ESTIMATE':
                        break
                    elif lBeg:
                        cWords = cLine.split()
                        nParSNX = nParSNX+1
                        if cWords[1] == cPar1 and cWords[2] in cSVN:
                            # par index for this satellite
                            iPar1[cSVN.index(cWords[2])] = int(cWords[0])
                        elif cWords[1] == cPar2:
                            # par index for the satellite-independent par
                            iPar2 = int(cWords[0])
                # Check for the satellite-independent par
                if iPar2 == 0:
                    sys.exit(cPar2+' not found in '+fSNX)
                fOb.seek(0)
                # Read the full correlation matrix
                R = np.zeros((nParSNX, nParSNX))
                lBeg = False
                for cLine in fOb:
                    if cLine[0:32] == '+SOLUTION/MATRIX_ESTIMATE L CORR':
                        lBeg = True
                    elif cLine[0:32] == '-SOLUTION/MATRIX_ESTIMATE L CORR':
                        break
                    elif lBeg:
                        cWords = cLine.split()
                        iRow = int(cWords[0])
                        iCol = int(cWords[1])
                        for i in range(min(3, iRow-iCol+1)):
                            # lower-triangular part
                            if iRow-1 == iCol+i-1:
                                R[iRow-1, iCol+i-1] = 1
                            else:
                                R[iRow-1, iCol+i-1] = float(cWords[i+2])
                            # upper-triangular part
                            R[iCol+i-1, iRow-1] = R[iRow-1, iCol+i-1]
            # Pick out the required correlation for each satellite
            for i in range(nSat):
                rCor[iDay, i*nSer+iSer] = np.abs(R[iPar1[i]-1, iPar2-1])

    fig, axs = plt.subplots(nSat, 1, squeeze=False,
                            sharex='col', figsize=(8, nSat*2))
    # fig.subplots_adjust(hspace=0.1)
    # fig.subplots_adjust(wspace=0.1)

    x = np.arange(nDay)
    w = 1/(nSer+1)
    for i in range(nSat):
        axs[i, 0].text(0.02, 0.98, cSVN[i], transform=axs[i, 0].transAxes, ha='left', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
        for iSer in range(nSer):
            axs[i, 0].bar(x+(iSer-nSer/2)*w, rCor[:, i*nSer+iSer],
                          w, align='edge', label=cSer[iSer])

        axs[i, 0].legend(ncol=nSer, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                         prop={'family': 'Arial', 'size': 14})
        axs[i, 0].grid(which='major', axis='y', color='darkgray', linestyle='--',
                       linewidth=0.8)
        axs[i, 0].set_axisbelow(True)
        axs[i, 0].set_ylim(bottom=0, top=1)
        axs[i, 0].set_ylabel('Correlations', fontname='Arial', fontsize=16)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
    axs[i, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    axs[i, 0].set_xlim(left=-1, right=nDay)
    axs[i, 0].set_xticks(x)
    axs[i, 0].set_xticklabels(
        '', fontdict={'fontsize': 14, 'fontname': 'Arial'})

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotParCorr22(fSNXList, cSVN, cParList, OutFilePrefix, OutFileSuffix):
    '''
    Do the same thing as `PlotParCorr20` but only for one specific satellite

    fSNXList --- SINEX files list
        cSVN --- the specific satellite SVN
    cParList --- Specific parameters list for satellites

    NOTE: Here we assume that the parameters list does not change
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # Get the actual parameters table for this satellite
    cPar = []
    for i in range(len(fSNXList)):
        with open(fSNXList[i], mode='rt') as fOb:
            lBeg = False
            for cLine in fOb:
                if cLine[0:18] == '+SOLUTION/ESTIMATE':
                    lBeg = True
                elif cLine[0:18] == '-SOLUTION/ESTIMATE':
                    break
                elif lBeg:
                    cWords = cLine.split()
                    if (cWords[2] != cSVN) or (cWords[1] not in cParList):
                        continue
                    cPar.append(cWords[1])
        # As the parameter list does not change,
        # we only need to read one file
        if len(cPar) > 0:
            break
    nPar = len(cPar)
    if nPar < 2:
        sys.exit('No enought par found for '+cSVN, nPar)

    R = []
    rEpo = []
    for i in range(nPar-1):
        R.append([])
        for j in range(i+1):
            R[i].append([])
    for i in range(len(fSNXList)):
        # Get the date from the file name
        WK = int(fSNXList[i][-9:-5])
        WKD = int(fSNXList[i][-5:-4])
        YYYY, MO, DD = GNSSTime.wkd2dom(WK, WKD)
        # Get the correlation matrix for this satellite
        # 1) Get the local index
        iPar = np.zeros(nPar, dtype=np.uint32)
        # Whether the required satellite found
        lFound = False
        with open(fSNXList[i], mode='rt') as fOb:
            lBeg = False
            for cLine in fOb:
                if cLine[0:18] == '+SOLUTION/ESTIMATE':
                    lBeg = True
                elif cLine[0:18] == '-SOLUTION/ESTIMATE':
                    break
                elif lBeg:
                    cWords = cLine.split()
                    if (cWords[2] != cSVN) or (cWords[1] not in cPar):
                        # Not required satellite or parameters
                        continue
                    iPar[cPar.index(cWords[1])] = int(cWords[0])
                    lFound = True
        if not lFound:
            continue
        rEpo.append(datetime.datetime(YYYY, MO, DD))
        # Extract the correlation matrix of this satellite
        with open(fSNXList[i], mode='rt') as fOb:
            lBeg = False
            for cLine in fOb:
                if cLine[0:32] == '+SOLUTION/MATRIX_ESTIMATE L CORR':
                    lBeg = True
                elif cLine[0:32] == '-SOLUTION/MATRIX_ESTIMATE L CORR':
                    break
                elif lBeg:
                    cWords = cLine.split()
                    iRow0 = int(cWords[0])
                    if iRow0 not in iPar:
                        continue
                    iCol0 = int(cWords[1])
                    for i in range(min(3, iRow0-iCol0+1)):
                        if iCol0+i not in iPar:
                            continue
                        if iRow0 == iCol0+i:
                            continue
                        # Global index
                        for j in range(nPar):
                            if iPar[j] == iRow0:
                                iRow = j
                            elif iPar[j] == iCol0+i:
                                iCol = j
                        R[iRow-1][iCol].append(float(cWords[i+2]))

    fig, axs = plt.subplots(nPar-1, nPar-1, squeeze=False, sharex='col', sharey='row',
                            figsize=((nPar-1)*2, (nPar-1)*2))
    fig.suptitle(cSVN, x=0.5, y=0.7, fontfamily='Arial',
                 fontsize=18, fontweight='bold')

    for i in range(nPar):
        if cPar[i][0:5] == 'BOXW_':
            cPar[i] = cPar[i][5:]
        elif cPar[i] == 'Kd_BERN':
            cPar[i] = 'D0'
        elif cPar[i] == 'Ky_BERN':
            cPar[i] = 'Y0'
        elif cPar[i] == 'Kb_BERN':
            cPar[i] = 'B0'
        elif cPar[i] == 'Kbc1_BERN':
            cPar[i] = 'Bc1'
        elif cPar[i] == 'Kbs1_BERN':
            cPar[i] = 'Bs1'
        elif cPar[i] == 'Kdc2_BERN':
            cPar[i] = 'Dc2'
        elif cPar[i] == 'Kds2_BERN':
            cPar[i] = 'Ds2'
        elif cPar[i] == 'Kdc4_BERN':
            cPar[i] = 'Dc4'
        elif cPar[i] == 'Kds4_BERN':
            cPar[i] = 'Ds4'
        elif cPar[i] == 'GEOCX0':
            cPar[i] = 'Xg'
        elif cPar[i] == 'GEOCY0':
            cPar[i] = 'Yg'
        elif cPar[i] == 'GEOCZ0':
            cPar[i] = 'Zg'
        elif cPar[i] == 'PXSAT':
            cPar[i] = 'PX'
        elif cPar[i] == 'PYSAT':
            cPar[i] = 'PY'
        elif cPar[i] == 'PZSAT':
            cPar[i] = 'PZ'
        elif cPar[i] == 'VXSAT':
            cPar[i] = 'VX'
        elif cPar[i] == 'VYSAT':
            cPar[i] = 'VY'
        elif cPar[i] == 'VZSAT':
            cPar[i] = 'VZ'

    for i in range(nPar-1):
        axs[i, 0].set_ylabel(cPar[i+1], fontname='Arial', fontsize=16)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        for j in range(i+1):
            axs[i, j].plot(rEpo, R[i][j], '.', ms=4)
            axs[i, j].axhline(color='darkgray', linestyle='dashed', alpha=0.5)
            axs[i, j].set_ylim(bottom=-1, top=1)
            axs[i, j].set_xticklabels([])
            if i == (nPar-2):
                axs[i, j].set_xlabel(cPar[j], fontname='Arial', fontsize=16)
        # Close the other axies
        for j in range(i+1, nPar-1):
            axs[i, j].set_axis_off()

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotNMatrix(fSNX, nPar, nPar1, nPar2, lExtract, OutFilePrefix, OutFileSuffix):
    '''
    Do some analysis on the Normal Matrix

    nPar --- Dimension of the Normal Matrix
    '''

    if (nPar1 > nPar) or (nPar2 > nPar) or (nPar1 > nPar2):
        sys.exit('Wrong input par numbers')

    # Get the full Normal Matrix
    N = np.zeros((nPar, nPar))
    with open(fSNX, mode='rt') as fOb:
        # with open(fSNX,mode='r',encoding='UTF-16') as fOb:
        lBeg = False
        for cLine in fOb:
            if cLine[0:34] == '+SOLUTION/NORMAL_EQUATION_MATRIX L':
                lBeg = True
            elif cLine[0:34] == '-SOLUTION/NORMAL_EQUATION_MATRIX L':
                break
            elif lBeg:
                cWords = cLine.split()
                iRow = int(cWords[0])
                iCol = int(cWords[1])
                for i in range(min(3, iRow-iCol+1)):
                    # lower-triangular part
                    N[iRow-1, iCol+i-1] = float(cWords[i+2])
                    # upper-triangular part
                    N[iCol+i-1, iRow-1] = N[iRow-1, iCol+i-1]
    # Extract the specified normal matrix
    if lExtract:
        fOut = open(OutFilePrefix + os.path.splitext(OutFileSuffix)[0], 'w')
        # Top header line
        StrTmp = '      '
        for i in range(nPar1, nPar2+1):
            StrTmp = StrTmp+'        {: >6d}        '.format(i)
        fOut.write(StrTmp + '\n')
        # The normal matrix
        for i in range(nPar1, nPar2+1):
            StrTmp = '{: >6d}'.format(i)
            for j in range(nPar1, nPar2+1):
                StrTmp = StrTmp+' {: >21.14E}'.format(N[i-1, j-1])
            fOut.write(StrTmp + '\n')

    # Analysis on specific block
    N0 = N[nPar1 - 1:nPar2, nPar1 - 1:nPar2]
    # Rank && Conditional number of the normal matrix
    print('Rank: ', NupLA.matrix_rank(N0))
    print('Cond: ', NupLA.cond(N0, p=2))
    # Determinant of the normal matrix
    print('Det:  ', SciLA.det(N0))
    # the max && min diagonal element
    Dig = np.zeros(nPar2-nPar1+1)
    for i in range(nPar2-nPar1+1):
        Dig[i] = N0[i, i]
    print('Max dig:  ', np.amax(Dig), np.argmax(Dig)+nPar1)
    print('Min dig:  ', np.amin(Dig), np.argmin(Dig)+nPar1)


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

    # InFilePrefix=r'Z:/DATA/ILRS/POS/'
    # InFilePrefix = r'Y:/PRO_2019001_2020366_WORK/C0/WORK2019???/'

    # fSNXList = glob.glob(InFilePrefix+'phb*.snx')
    # fSNXList=glob.glob(InFilePrefix+'*.v170.snx')

    # fSNX=r'D:/Code/PROJECT/WORK2022054/phb21983.snx'
    fSNX = os.path.join(
        cWrkPre0, r'PRO_2019001_2020366_WORK/D660/WORK2019335/phb20820.snx')

    # OutFilePrefix=r'D:/Code/PROJECT/WORK2022054/'
    # OutFilePrefix=r'D:/Code/PROJECT/WORK_SLR/'
    # OutFilePrefix = os.path.join(
    #     cDskPre0, r'PRO_2019001_2020366/D660/ParCorr/')
    OutFilePrefix = os.path.join(
        cWrkPre0, r'PRO_2019001_2020366_WORK/D660/WORK2019335/')

    # OutFileSuffix='StaPos.pdf'
    # PlotCRDRes(fSNXList,OutFilePrefix,OutFileSuffix)

    # cSVNList=['C01','C02','C03','C04','C05']
    # cSVNList=['C003','C016','C018','C006','C011']
    # cSVNList=['G063','G045','G052']
    # BDS-3
    # cSVNList = ['C201', 'C202', 'C203', 'C204', 'C205', 'C206', 'C207', 'C208', 'C209', 'C210',
    #             'C211', 'C212', 'C213', 'C214', 'C215', 'C216', 'C217', 'C218', 'C219', 'C220']

    # cParList=['PXSAT','PYSAT','PZSAT','VXSAT','VYSAT','VZSAT',
    #           'BOXW_SB','BOXW_SD','BOXW_SS','BOXW_Y0',
    #           'BOXW_+XR','BOXW_-XR','BOXW_+XAD','BOXW_-XAD',
    #           'BOXW_+ZR','BOXW_-ZR','BOXW_+ZAD','BOXW_-ZAD',
    #           'Kd_BERN','Ky_BERN','Kb_BERN','Kbc1_BERN','Kbs1_BERN',]
    # cParList=['PXSAT','PYSAT','PZSAT','VXSAT','VYSAT','VZSAT',
    #           'Kd_BERN','Ky_BERN','Kb_BERN','Kbc1_BERN','Kbs1_BERN',
    #           'AVPulse','CVPulse','RVPulse']
    # cParList=['Kd_BERN','Ky_BERN','Kb_BERN','Kbc1_BERN','Kbs1_BERN',
    #           'Kdc2_BERN','Kds2_BERN','Kdc4_BERN','Kds4_BERN']
    # cParList = ['PXSAT', 'PYSAT', 'PZSAT', 'VXSAT', 'VYSAT', 'VZSAT',
    #             'Kd_BERN', 'Ky_BERN', 'Kb_BERN', 'Kbc1_BERN', 'Kbs1_BERN']
    # cParList = ['PXSAT', 'PYSAT', 'PZSAT', 'VXSAT', 'VYSAT', 'VZSAT',
    #             'Kd_BERN', 'Ky_BERN', 'Kb_BERN', 'Kbc1_BERN', 'Kbs1_BERN', 'ISLBIASS']
    # cParList = ['ISLBIASS']
    # cParList = ['ISLBIASD']
    cParList = ['ISLBIASR', 'ISLBIAST']
    # cParList = ['ISLBIASR0', 'ISLBIAST0']

    # OutFileSuffix='OrbCorr_2019335'
    # PlotParCorr11(fSNX,cSVNList,cParList,OutFilePrefix,OutFileSuffix)
    # cParList2=['GEOCX0','GEOCY0','GEOCZ0']
    # PlotParCorr12(fSNX,cSVNList,cParList,cParList2,OutFilePrefix,OutFileSuffix)
    OutFileSuffix = 'ISLBiasCorr3_2019335'
    PlotParCorr13(fSNX, ['ALL'], cParList, OutFilePrefix, OutFileSuffix)

    # OutFileSuffix='OrbCorr2'
    # cSVNList=['C217','C218']
    # PlotParCorr20(fSNXList,cSVNList,cParList,OutFilePrefix,OutFileSuffix)

    # cSer = []
    # fPathSer = []
    # cSer.append('GloA')
    # fPathSer.append(os.path.join(cWrkPre0, r'PRO_2019001_2020366_WORK/C02'))
    # cSer.append('GloA+ISL')
    # fPathSer.append(os.path.join(cWrkPre0, r'PRO_2019001_2020366_WORK/D660'))
    #
    # OutFileSuffix = 'Corr_D0_ZGCC_C201'
    # PlotParCorr21(cSer, fPathSer, ['C201'], 'Kd_BERN',
    #               'GEOCZ0', 58818, 31, OutFilePrefix, OutFileSuffix)
    # OutFileSuffix = 'Corr_Bc1_ZGCC_C201'
    # PlotParCorr21(cSer, fPathSer, ['C201'], 'Kbc1_BERN',
    #               'GEOCZ0', 58818, 31, OutFilePrefix, OutFileSuffix)

    OutFileSuffix = 'TestSNX'
    # PlotNMatrix(fSNX,426,374,426,True,OutFilePrefix,OutFileSuffix)
