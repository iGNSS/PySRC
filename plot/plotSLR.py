#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
'''
__author__ = 'hanbing'

# Standard library imports
import os
import sys
import os.path
import math

# Related third party imports
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt


def GetOmCSLR1(fOmc, cSta0, cSat0, rMJD0):
    '''
    Extract SLR OmC for observations between cSta0 and cSat0 within the
    specified time period

     fOmc --- Input SLR OmC file
    cSta0 --- Specified stations, set cSta0[0]='ALL' to get all stations
    cSat0 --- ILRS names of specified satellites.
              Set cSat0[0]='ALL' to get all satellites
    rMJD0 --- Specified time periods
    '''

    cSta = []
    cSat = []
    OmC1 = [[] for i in range(7)]
    OmC2 = [[] for i in range(7)]
    with open(fOmc, mode='rt') as fOb:
        for cLine in fOb:
            if cLine[0:4] == 'BAD ' or cLine[0:4] == 'OK  ':
                cWords = cLine.split()
                if cSta0[0] != 'ALL' and cWords[2] not in cSta0:
                    continue
                if cSat0[0] != 'ALL' and cWords[4] not in cSat0:
                    continue
                # for debug
                # if cWords[2]=='7249' or cWords[2]=='1868':
                #     continue
                # for debug

                rMJD = float(cWords[6]) + float(cWords[7])/86400.0
                if rMJD < rMJD0[0] or rMJD > rMJD0[1]:
                    continue
                if cWords[2] not in cSta:
                    cSta.append(cWords[2])
                if cWords[4] not in cSat:
                    cSat.append(cWords[4])
                if cLine[0:4] == 'BAD ' or abs(float(cWords[8])/10) > 100:
                    OmC2[0].append(rMJD)
                    # Omc, mm -> cm
                    OmC2[1].append(float(cWords[8])/10)
                    # beta angle
                    OmC2[2].append(float(cWords[9]))
                    # E angle
                    OmC2[3].append(float(cWords[10]))
                    # u angle
                    OmC2[4].append(float(cWords[12]))
                    # Elevation angle
                    OmC2[5].append(float(cWords[13]))
                    # Epsilon
                    OmC2[6].append(float(cWords[11]))
                else:
                    OmC1[0].append(rMJD)
                    # Omc, mm -> cm
                    OmC1[1].append(float(cWords[8])/10)
                    # beta angle
                    OmC1[2].append(float(cWords[9]))
                    # E angle
                    OmC1[3].append(float(cWords[10]))
                    # u angle
                    OmC1[4].append(float(cWords[12]))
                    # Elevation angle
                    OmC1[5].append(float(cWords[13]))
                    # Epsilon
                    OmC1[6].append(float(cWords[11]))

    return np.array(OmC1), np.array(OmC2), cSta, cSat


def GetOmCSLR2(fOmc, rMJD0, cSatExl, lExBAD, rMax):
    '''
    Extract SLR OmC for all observations within the specified time period
    and return the OmC individually for each satellite.

     fOmc --- Input SLR OmC file
    rMJD0 --- Specified time periods
  cSatExl --- Satellites to be excluded, ILRS name
   lExBAD --- Whether exclude all the BAD records
     rMax --- If great than zero, exclude all records (No matter whether
              it is a BAD or OK record) that are great than this value in
              the absolute sense, in cm

    NOTE: Returned SLR OmCs are stored in inhomogeneous python list and
          the Satellite list are ILRS names instead of PRN
    '''

    cSat = []
    OmCGod = []
    OmCBad = []
    with open(fOmc, mode='rt') as fOb:
        for cLine in fOb:
            if cLine[0:4] == 'BAD ' or cLine[0:4] == 'OK  ':
                cWords = cLine.split()
                rMJD = float(cWords[6]) + float(cWords[7])/86400.0
                if rMJD < rMJD0[0] or rMJD > rMJD0[1]:
                    continue
                if cWords[4] in cSatExl:
                    continue
                if cWords[4] not in cSat:
                    cSat.append(cWords[4])
                    for i in range(7):
                        OmCGod.append([])
                        OmCBad.append([])
                iSat = cSat.index(cWords[4])
                if (cLine[0:4] == 'BAD ' and lExBAD) or (rMax > 0 and abs(float(cWords[8])/10) > rMax) or \
                   (cWords[8] == '9999999.9' or cWords[8] == '-9999999.9'):
                    OmCBad[iSat*7+0].append(rMJD)
                    # Omc, mm -> cm
                    OmCBad[iSat*7+1].append(float(cWords[8])/10)
                    # beta angle
                    OmCBad[iSat*7+2].append(float(cWords[9]))
                    # E angle
                    OmCBad[iSat*7+3].append(float(cWords[10]))
                    # u angle
                    OmCBad[iSat*7+4].append(float(cWords[12]))
                    # Elevation angle
                    OmCBad[iSat*7+5].append(float(cWords[13]))
                    # Epsilon
                    OmCBad[iSat*7+6].append(float(cWords[11]))
                else:
                    OmCGod[iSat*7+0].append(rMJD)
                    # Omc, mm -> cm
                    OmCGod[iSat*7+1].append(float(cWords[8])/10)
                    # beta angle
                    OmCGod[iSat*7+2].append(float(cWords[9]))
                    # E angle
                    OmCGod[iSat*7+3].append(float(cWords[10]))
                    # u angle
                    OmCGod[iSat*7+4].append(float(cWords[12]))
                    # Elevation angle
                    OmCGod[iSat*7+5].append(float(cWords[13]))
                    # Epsilon
                    OmCGod[iSat*7+6].append(float(cWords[11]))

    return OmCGod, OmCBad, cSat


def PlotMergeSLRLog(fLog, OutFilePrefix, OutFileSuffix):
    '''
    Display the data quantity for each satellite and station based on
    the output log of mergingg SLR npt files
    '''

    with open(fLog, mode='r', encoding='UTF-16') as fOb:
        # Get station list
        lBeg = False
        lEnd = False
        nSta = 0
        cSta = []
        nObsSta = []
        for cLine in fOb:
            if '+Sta data quantity' in cLine:
                lBeg = True
                continue
            if '-Sta data quantity' in cLine:
                lEnd = True
                continue
            if not lBeg:
                continue
            if lEnd:
                break
            Words = cLine.split()
            nSta = nSta+1
            cSta.append(Words[0])
            nObsSta.append(int(Words[1]))
        for i in range(0, nSta-1):
            for j in range(i+1, nSta):
                if cSta[i] <= cSta[j]:
                    continue
                cTmp = cSta[i]
                nTmp = nObsSta[i]
                cSta[i] = cSta[j]
                nObsSta[i] = nObsSta[j]
                cSta[j] = cTmp
                nObsSta[j] = nTmp
        fOb.seek(0)
        # Get satellite list
        lBeg = False
        lEnd = False
        nSat = 0
        cSat = []
        nObsSat = []
        for cLine in fOb:
            if '+Sat data quantity' in cLine:
                lBeg = True
                continue
            if '-Sat data quantity' in cLine:
                lEnd = True
                continue
            if not lBeg:
                continue
            if lEnd:
                break
            Words = cLine.split()
            nSat = nSat+1
            cSat.append(Words[0][6:10])
            nObsSat.append(int(Words[1]))
        for i in range(0, nSat-1):
            for j in range(i+1, nSat):
                if cSat[i] <= cSat[j]:
                    continue
                cTmp = cSat[i]
                nTmp = nObsSat[i]
                cSat[i] = cSat[j]
                nObsSat[i] = nObsSat[j]
                cSat[j] = cTmp
                nObsSat[j] = nTmp
        # Get the start && end day
        iDayMin = 99999
        iDayMax = 0
        nObsMin = 99999
        nObsMax = 0
        for i in range(nSat):
            fOb.seek(0)
            lBeg = False
            lEnd = False
            for cLine in fOb:
                if '+Number of obs of each day for '+cSat[i] in cLine:
                    lBeg = True
                    continue
                if '-Number of obs of each day for '+cSat[i] in cLine:
                    lEnd = True
                    continue
                if not lBeg:
                    continue
                if lEnd:
                    break
                Words = cLine.split()
                if int(Words[0]) < iDayMin:
                    iDayMin = int(Words[0])
                if int(Words[0]) > iDayMax:
                    iDayMax = int(Words[0])
                if int(Words[1]) < nObsMin:
                    nObsMin = int(Words[1])
                if int(Words[1]) > nObsMax:
                    nObsMax = int(Words[1])

    fig, axs = plt.subplots(2, 1, figsize=(15, 10))
    fig.subplots_adjust(hspace=0.2)

    axs[0].set_xlim(left=-1, right=nSta)
    x = np.arange(nSta)
    axs[0].bar(x, nObsSta)
    axs[0].set_xticks(x)
    axs[0].set_xticklabels(cSta, rotation=45)

    axs[1].set_xlim(left=-1, right=nSat)
    x = np.arange(nSat)
    axs[1].bar(x, nObsSat)
    axs[1].set_xticks(x)
    axs[1].set_xticklabels(cSat)

    strTmp = OutFilePrefix+OutFileSuffix
    with PdfPages(strTmp) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        fig, axs = plt.subplots(nSat, 1, sharex='col', figsize=(15, 1.5*nSat))
        fig.subplots_adjust(hspace=0.1)
        with open(fLog, mode='r', encoding='UTF-16') as fOb:
            for i in range(nSat):
                fOb.seek(0)
                lBeg = False
                lEnd = False
                nDay = 0
                iDay = []
                nObsDay = []
                nObs = 0
                for cLine in fOb:
                    if '+Number of obs of each day for '+cSat[i] in cLine:
                        lBeg = True
                        continue
                    if '-Number of obs of each day for '+cSat[i] in cLine:
                        lEnd = True
                        continue
                    if not lBeg:
                        continue
                    if lEnd:
                        break
                    Words = cLine.split()
                    nDay = nDay+1
                    iDay.append(int(Words[0]))
                    nObsDay.append(int(Words[1]))
                    nObs = nObs+int(Words[1])
                axs[i].set_xlim(left=iDayMin-1, right=iDayMax+1)
                axs[i].set_ylim(bottom=0, top=nObsMax+3)
                axs[i].plot(iDay, nObsDay, label=cSat[i])
                axs[i].text(0.95, 0.95, '{} #={:>7d} Ave={:>5.1f}'.format(cSat[i], nObs, nObs/nDay),
                            transform=axs[i].transAxes,
                            ha='right', va='top')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)


def PlotOmCSLR1(fOmc, rMJD0, OutFilePrefix, OutFileSuffix):
    '''
    Plot SLR OmC series for each satellite, omc VS time

    rMJD0 --- Specified time periods
    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # Get the whole satellite set
    cSat = []
    MJD1 = 99999
    MJD2 = 0
    with open(fOmc, mode='rt') as fOb:
        for cLine in fOb:
            if cLine[0:4] == 'BAD ' or cLine[0:4] == 'OK  ':
                cWords = cLine.split()
                rMJD = float(cWords[6]) + float(cWords[7])/86400.0
                if rMJD < rMJD0[0] or rMJD > rMJD0[1]:
                    continue

                # for debug
                # if cWords[2]=='7249' or cWords[2]=='1868':
                #     continue
                # for debug
                if cWords[4] not in cSat:
                    cSat.append(cWords[4])
                MJD1 = min(MJD1, rMJD)
                MJD2 = max(MJD2, rMJD)
    MJD1 = int(MJD1-10)
    MJD2 = int(MJD2+10)

    cSatSLR = ['compassg1', 'compassi3', 'compassi5', 'compassi6b', 'compassm3',
               'beidou3m1', 'beidou3m2', 'beidou3m3', 'beidou3m9', 'beidou3m10']
    cSatLab = ['compassg1 (C003)', 'compassi3 (C008)', 'compassi5 (C010)', 'compassi6b (C017)',
               'compassm3 (C012)', 'beidou3m1 (C201, CAST)', 'beidou3m2 (C202, CAST)',
               'beidou3m3 (C206, CAST)', 'beidou3m9 (C207, SECM)', 'beidou3m10 (C208, SECM)']
    cSat0 = []
    cLab0 = []
    for i in range(len(cSatSLR)):
        if cSatSLR[i] not in cSat:
            continue
        cSat0.append(cSatSLR[i])
        cLab0.append(cSatLab[i])
    nSat0 = len(cSat0)

    fig, axs = plt.subplots(nSat0, 1, sharex='col',
                            squeeze=False, figsize=(12, nSat0*3.0))
    fig.subplots_adjust(hspace=0.2)
    for i in range(nSat0):
        # Extract OmC for this satellite
        X1, X2, cSta2, cSat2 = GetOmCSLR1(fOmc, ['ALL'], [cSat0[i]], rMJD0)
        nBad = X2[1].size
        nOmC = X1[1].size
        Mea = np.mean(X1[1])
        Sig = np.std(X1[1])
        RMS = 0.0
        for j in range(nOmC):
            RMS = RMS+X1[1, j]*X1[1, j]
        RMS = np.sqrt(RMS/nOmC)
        print('{:<20s} {:>6d} {:>6d} {:>7.1f} {:>6.1f} {:>7.1f}'.format(cSat0[i],
                                                                        nOmC, nBad, Mea, Sig, RMS))

        axs[i, 0].plot(X1[0], X1[1], '.b', ms=4)
        axs[i, 0].set_xlim(left=MJD1, right=MJD2)
        if cSat0[i] == 'compassg1':
            axs[i, 0].set_ylim(-100, 100)
        elif cSat0[i] == 'compassi3' or cSat0[i] == 'compassi5' or cSat0[i] == 'compassi6b':
            axs[i, 0].set_ylim(-50, 50)
        else:
            axs[i, 0].set_ylim(-25, 25)
        axs[i, 0].axhline(color='darkgray', linestyle='dashed', alpha=0.5)
        axs[i, 0].text(0.02, 0.98, cLab0[i], transform=axs[i, 0].transAxes, ha='left', va='top',
                       family='Arial', size=14, color='darkred', weight='bold')
        axs[i, 0].text(0.98, 0.98, '{:>7.1f}+/-{:>6.1f} RMS={:>7.1f} {:>6d} used {:>6d} excl'.format(Mea, Sig, RMS, nOmC, nBad),
                       transform=axs[i, 0].transAxes, ha='right', va='top',
                       family='Arial', size=16)
        axs[i, 0].set_ylabel('[cm]', fontname='Arial', fontsize=16)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    axs[i, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotOmCSLR2(fOmc, OutFilePrefix, OutFileSuffix):
    '''
    Plot SLR OmC for each satellite, omc VS elevation angle
    '''

    # Get the whole satellite set
    cSat = []
    MJD1 = 99999
    MJD2 = 0
    with open(fOmc, mode='rt') as fOb:
        for cLine in fOb:
            if cLine[0:4] == 'BAD ' or cLine[0:4] == 'OK  ':
                cWords = cLine.split()
                # for debug
                # if cWords[2]=='7249' or cWords[2]=='1868':
                #     continue
                # for debug
                if cWords[4] not in cSat:
                    cSat.append(cWords[4])
                rMJD = float(cWords[6]) + float(cWords[7])/86400.0
                MJD1 = min(MJD1, rMJD)
                MJD2 = max(MJD2, rMJD)
    MJD1 = int(MJD1-10)
    MJD2 = int(MJD2+10)
    cSat0 = []
    cSatSLR = ['compassg1', 'compassi3', 'compassi5', 'compassi6b', 'compassm3',
               'beidou3m1', 'beidou3m2', 'beidou3m3', 'beidou3m9', 'beidou3m10']
    for i in range(len(cSatSLR)):
        if cSatSLR[i] in cSat:
            cSat0.append(cSatSLR[i])
    cSat = cSat0
    nSat = len(cSat)

    fig, axs = plt.subplots(nSat, 1, sharex='col',
                            squeeze=False, figsize=(8, nSat*1.5))
    fig.subplots_adjust(hspace=0.1)
    for i in range(nSat):
        # Extract OmC for this satellite
        X1, X2, cSta2, cSat2 = GetOmCSLR1(fOmc, ['ALL'], [cSat[i]], [0, 99999])
        nBad = X2[1].size
        nOmC = X1[1].size
        Mea = np.mean(X1[1])
        Sig = np.std(X1[1])
        RMS = 0.0
        for j in range(nOmC):
            RMS = RMS+X1[1, j]*X1[1, j]
        RMS = np.sqrt(RMS/nOmC)

        axs[i, 0].plot(X1[5], X1[1], '.b', label=cSat[i])
        axs[i, 0].set_xlim(left=0, right=90)
        axs[i, 0].text(0.05, 0.95, cSat[i], transform=axs[i, 0].transAxes,
                       ha='left', va='top')
        axs[i, 0].set_ylabel('[cm]')
    axs[i, 0].set_xlabel('Elev [deg]')

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotOmCSLR3(fOmc, OutFilePrefix, OutFileSuffix):
    '''
    Plot SLR OmC for each satellite-station pair, omc VS time
    '''

    # Get the whole satellite && station set
    cSat = []
    cSta = []
    MJD1 = 99999
    MJD2 = 0
    with open(fOmc, mode='rt') as fOb:
        for cLine in fOb:
            if cLine[0:4] == 'BAD ' or cLine[0:4] == 'OK  ':
                cWords = cLine.split()
                # for debug
                # if cWords[2]=='7249' or cWords[2]=='1868':
                #     continue
                # for debug
                if cWords[2] not in cSta:
                    cSta.append(cWords[2])
                if cWords[4] not in cSat:
                    cSat.append(cWords[4])
                rMJD = float(cWords[6]) + float(cWords[7])/86400.0
                MJD1 = min(MJD1, rMJD)
                MJD2 = max(MJD2, rMJD)
    MJD1 = int(MJD1-10)
    MJD2 = int(MJD2+10)
    cSta.sort()
    nSta = len(cSta)
    cSat0 = []
    cSatSLR = ['compassg1', 'compassi3', 'compassi5', 'compassi6b', 'compassm3',
               'beidou3m1', 'beidou3m2', 'beidou3m3', 'beidou3m9', 'beidou3m10']
    for i in range(len(cSatSLR)):
        if cSatSLR[i] in cSat:
            cSat0.append(cSatSLR[i])
    cSat = cSat0
    nSat = len(cSat)

    fig, axs = plt.subplots(nSta, nSat, sharex='col',
                            squeeze=False, figsize=(6*nSat, nSta*2))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.1)
    for i in range(nSat):
        for j in range(nSta):
            cSta1 = []
            cSta1.append(cSta[j])
            cSat1 = []
            cSat1.append(cSat[i])
            X1, X2, cSta2, cSat2 = GetOmCSLR1(fOmc, cSta1, cSat1, [0, 99999])
            nOmC = X1[1].size
            nBad = X2[1].size

            if nOmC != 0:
                axs[j, i].plot(X1[0], X1[1], '.b')
            if i == 0:
                axs[j, i].set_ylabel(cSta[j])
            if j == 0:
                axs[j, i].text(0.05, 0.95, cSat[i], transform=axs[j, i].transAxes,
                               ha='left', va='top')
            axs[j, i].text(0.95, 0.95, '#={:>6d}, nBad={:>6d}'.format(nOmC, nBad),
                           transform=axs[j, i].transAxes, ha='right', va='top')
            axs[j, i].set_xlim(left=MJD1, right=MJD2)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotOmCSLR4(fOmc, rMJD0, OutFilePrefix, OutFileSuffix):
    '''
    Plot SLR OmC for each satellite, omc VS E angle && omc VS u angle && omc VS Beta angle

    rMJD0 --- Specified time periods
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # Get the whole satellite set
    cSat = []
    with open(fOmc, mode='rt') as fOb:
        for cLine in fOb:
            if cLine[0:4] == 'BAD ' or cLine[0:4] == 'OK  ':
                cWords = cLine.split()
                rMJD = float(cWords[6]) + float(cWords[7])/86400.0
                if rMJD < rMJD0[0] or rMJD > rMJD0[1]:
                    continue

                # for debug
                # if cWords[2]=='7249' or cWords[2]=='1868':
                #     continue
                # for debug
                if cWords[4] not in cSat:
                    cSat.append(cWords[4])

    cSatSLR = ['compassg1', 'compassi3', 'compassi5', 'compassi6b', 'compassm3',
               'beidou3m1', 'beidou3m2', 'beidou3m3', 'beidou3m9', 'beidou3m10']
    cSatLab = ['compassg1 (C003)', 'compassi3 (C008)', 'compassi5 (C010)', 'compassi6b (C017)',
               'compassm3 (C012)', 'beidou3m1 (C201, CAST)', 'beidou3m2 (C202, CAST)',
               'beidou3m3 (C206, CAST)', 'beidou3m9 (C207, SECM)', 'beidou3m10 (C208, SECM)']
    cSat0 = []
    cLab0 = []
    for i in range(len(cSatSLR)):
        if cSatSLR[i] not in cSat:
            continue
        cSat0.append(cSatSLR[i])
        cLab0.append(cSatLab[i])
    nSat0 = len(cSat0)

    lEpsilon = True
    fig, axs = plt.subplots(
        nSat0, 3, sharex='col', sharey='row', squeeze=False, figsize=(12, nSat0*1.5))
    # fig.subplots_adjust(hspace=0.1); fig.subplots_adjust(wspace=0.1)

    for i in range(nSat0):
        # Extract OmC for this satellite
        X1, X2, cSta2, cSat2 = GetOmCSLR1(fOmc, ['ALL'], [cSat0[i]], rMJD0)
        nOmC = X1[1].size
        if nOmC == 0:
            continue
        Mea = np.mean(X1[1])
        Sig = np.std(X1[1])
        RMS = 0.0
        for j in range(nOmC):
            RMS = RMS+X1[1, j]*X1[1, j]
        RMS = np.sqrt(RMS/nOmC)

        #Epsilon or E
        if lEpsilon:
            axs[i, 0].plot(X1[6], X1[1], '.b', ms=4)
        else:
            axs[i, 0].plot(X1[3], X1[1], '.b', ms=4)

        if cSat0[i] == 'compassg1':
            axs[i, 0].set_ylim(-100, 100)
        elif cSat0[i] == 'compassi3' or cSat0[i] == 'compassi5' or cSat0[i] == 'compassi6b':
            axs[i, 0].set_ylim(-50, 50)
        else:
            axs[i, 0].set_ylim(-25, 25)
        axs[i, 0].axhline(color='darkgray', linestyle='dashed', alpha=0.5)
        axs[i, 0].set_xlim(left=0, right=180)
        axs[i, 0].set_xticks([30, 60, 90, 120, 150])
        axs[i, 0].text(0.01, 0.99, cLab0[i], transform=axs[i, 0].transAxes, ha='left', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkred'})
        axs[i, 0].set_ylabel('[cm]', fontname='Arial', fontsize=16)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        # u
        axs[i, 1].plot(X1[4], X1[1], '.b', ms=4)
        axs[i, 1].set_xlim(left=0, right=360)
        axs[i, 1].set_xticks([60, 120, 180, 240, 300])
        axs[i, 1].axhline(color='darkgray', linestyle='dashed', alpha=0.5)
        axs[i, 1].text(0.99, 0.99, '{:>7.1f}+/-{:>6.1f} RMS={:>7.1f}'.format(Mea, Sig, RMS),
                       transform=axs[i, 1].transAxes, ha='right', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})

        # beta
        axs[i, 2].plot(X1[2], X1[1], '.b', ms=4)
        axs[i, 2].set_xlim(left=-90, right=90)
        axs[i, 2].set_xticks([-60, -30, 0, 30, 60])
        axs[i, 2].axhline(color='darkgray', linestyle='dashed', alpha=0.5)
    if lEpsilon:
        axs[i, 0].set_xlabel(r'$\epsilon$ [deg]',
                             fontname='Arial', fontsize=16)
    else:
        axs[i, 0].set_xlabel('E [deg]', fontname='Arial', fontsize=16)
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    axs[i, 1].set_xlabel(r'$\mu$ [deg]', fontname='Arial', fontsize=16)
    for tl in axs[i, 1].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    axs[i, 2].set_xlabel(r'$\beta$ [deg]', fontname='Arial', fontsize=16)
    for tl in axs[i, 2].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotOmCSLR5(fOmc, lNegativeU, cPRN0, nCol, OutFilePrefix, OutFileSuffix):
    '''
    Plot SLR OmC for each satellite, omc VS beta angle && u angle

  lNegativeU --- Whether set the range of u angle between [-180,+180].
                 Otherwise, it is [0,360]

       cPRN0 --- PRNs of specified satellites
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    cSatSLR = ['compassg1', 'compassi3', 'compassi5', 'compassi6b', 'compassm3',
               'beidou3m1', 'beidou3m2', 'beidou3m3', 'beidou3m9', 'beidou3m10']
    cSatPRN = ['C01', 'C08', 'C10', 'C13', 'C12',
               'C19', 'C20', 'C21', 'C29', 'C30']

    # Get the whole satellite set
    cSat = []
    MJD1 = 99999
    MJD2 = 0
    with open(fOmc, mode='rt') as fOb:
        for cLine in fOb:
            if cLine[0:4] == 'BAD ' or cLine[0:4] == 'OK  ':
                cWords = cLine.split()
                # for debug
                # if cWords[2]=='7249' or cWords[2]=='1868':
                #     continue
                # for debug
                if cPRN0[0] != 'ALL' and cWords[3] not in cPRN0:
                    continue
                if cWords[4] not in cSat:
                    cSat.append(cWords[4])
                rMJD = float(cWords[6]) + float(cWords[7])/86400.0
                MJD1 = min(MJD1, rMJD)
                MJD2 = max(MJD2, rMJD)
    MJD1 = int(MJD1-10)
    MJD2 = int(MJD2+10)
    nSat = len(cSat)
    cPRN = []
    for i in range(nSat):
        if cSat[i] not in cSatSLR:
            continue
        cPRN.append(cSatPRN[cSatSLR.index(cSat[i])])

    # Cal the number of row based on specified number of col
    nRow = math.ceil(nSat/nCol)
    fig, axs = plt.subplots(nRow, nCol, sharex='col', sharey='row',
                            squeeze=False, figsize=(nCol*5, nRow*4))
    # fig.subplots_adjust(hspace=0.1)
    # fig.subplots_adjust(wspace=0.05)

    for i in range(nSat):
        # Extract OmC for this satellite
        X1, X2, cSta2, cSat2 = GetOmCSLR1(fOmc, ['ALL'], [cSat[i]], [0, 99999])
        nBad = X2[1].size
        nOmC = X1[1].size
        Mea = np.mean(X1[1])
        Sig = np.std(X1[1])
        RMS = 0.0
        for j in range(nOmC):
            RMS = RMS+X1[1, j]*X1[1, j]
            # Convert from [0,360] to [-180,180]
            if lNegativeU and X1[4, j] >= 180:
                X1[4, j] = X1[4, j]-360
        RMS = np.sqrt(RMS/nOmC)
        # Cal the axis position, row-wise
        iRow = math.ceil((i+1)/nCol)-1
        iCol = i-iRow*nCol

        qm = axs[iRow, iCol].scatter(
            X1[4], X1[2], s=60, c=X1[1], marker='.', cmap='viridis', label=cSat[i])

        # axs[iRow,iCol].text(0.98,0.98,'{: >7.1f}+/-{: >6.1f} RMS={: >7.1f}'.format(Mea,Sig,RMS),
        #               transform=axs[iRow,iCol].transAxes,ha='right',va='top',
        #               fontdict={'fontsize':14,'fontname':'Arial','fontweight':'bold'})
        axs[iRow, iCol].text(0.02, 0.98, cPRN[i], transform=axs[iRow, iCol].transAxes, ha='left', va='top',
                             fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
        axs[iRow, iCol].grid(which='major', axis='both',
                             c='darkgray', ls='--', lw=0.4, alpha=0.5)
        axs[iRow, iCol].set_axisbelow(True)

        cb = fig.colorbar(qm, ax=axs[iRow, iCol])
        cb.set_label('[cm]', fontname='Arial', fontsize=16)
        cb.ax.yaxis.set_major_formatter('{x: >3.0f}')
        for tl in cb.ax.get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        if iCol == 0:
            axs[iRow, iCol].set_ylabel(
                r'$\beta$ [deg]', fontname='Arial', fontsize=16)
            for tl in axs[iRow, iCol].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
        if iRow == (nRow-1):
            axs[iRow, iCol].set_xlabel(
                r'$\mu$ [deg]', fontname='Arial', fontsize=16)
            if lNegativeU:
                axs[iRow, iCol].set_xlim(left=-180, right=180)
                axs[iRow, iCol].set_xticks([-60, -120, 0, 60, 120])
            else:
                axs[iRow, iCol].set_xlim(left=0, right=360)
                axs[iRow, iCol].set_xticks([60, 120, 180, 240, 300])
            for tl in axs[iRow, iCol].get_xticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotOmCSLR6(fOmc, OutFilePrefix, OutFileSuffix):
    '''
    Plot SLR OmC for each station, omc VS time
    '''

    # Get the whole station set
    cSta = []
    MJD1 = 99999
    MJD2 = 0
    with open(fOmc, mode='rt') as fOb:
        for cLine in fOb:
            if cLine[0:4] == 'BAD ' or cLine[0:4] == 'OK  ':
                cWords = cLine.split()
                # for debug
                # if cWords[2]=='7249' or cWords[2]=='1868':
                #     continue
                # for debug
                if cWords[2] not in cSta:
                    cSta.append(cWords[2])
                rMJD = float(cWords[6]) + float(cWords[7])/86400.0
                MJD1 = min(MJD1, rMJD)
                MJD2 = max(MJD2, rMJD)
    MJD1 = int(MJD1-10)
    MJD2 = int(MJD2+10)
    cSta.sort()
    nSta = len(cSta)

    fig, axs = plt.subplots(nSta, 1, sharex='col',
                            squeeze=False, figsize=(8, nSta*1.5))
    fig.subplots_adjust(hspace=0.1)
    for i in range(nSta):
        # Extract OmC for this station
        cSta1 = []
        cSta1.append(cSta[i])
        cSat1 = []
        cSat1.append('ALL')
        X1, X2, cSta2, cSat2 = GetOmCSLR1(fOmc, cSta1, cSat1, [0, 99999])
        nBad = X2[1].size
        nOmC = X1[1].size
        Mea = np.mean(X1[1])
        Sig = np.std(X1[1])
        RMS = 0.0
        for j in range(nOmC):
            RMS = RMS+X1[1, j]*X1[1, j]
        RMS = np.sqrt(RMS/nOmC)
        print('{:<20s} {:>6d} {:>6d} {:>7.1f} {:>6.1f} {:>7.1f}'.format(cSta[i],
                                                                        nOmC, nBad, Mea, Sig, RMS))

        axs[i, 0].plot(X1[0], X1[1], '.b', label=cSta[i])
        axs[i, 0].text(0.05, 0.95, cSta[i], transform=axs[i, 0].transAxes,
                       ha='left', va='top')
        axs[i, 0].text(0.95, 0.95, 'Mea={:>7.1f}, STD={:>6.1f}, RMS={:>7.1f}, #={:>6d}, nBad={:>6d}'.format(Mea, Sig, RMS, nOmC, nBad),
                       transform=axs[i, 0].transAxes, ha='right', va='top')
        axs[i, 0].set_xlim(left=MJD1, right=MJD2)
        axs[i, 0].set_ylabel('[cm]')
    print(' ')
    axs[i, 0].set_xlabel('Modified Julian Day')

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotOmCSLR7(fOmc, OutFilePrefix, OutFileSuffix):
    '''
    Plot SLR OmC for each station, omc VS elevation angle
    '''

    # Get the whole station set
    cSta = []
    MJD1 = 99999
    MJD2 = 0
    with open(fOmc, mode='rt') as fOb:
        for cLine in fOb:
            if cLine[0:4] == 'BAD ' or cLine[0:4] == 'OK  ':
                cWords = cLine.split()
                # for debug
                # if cWords[2]=='7249' or cWords[2]=='1868':
                #     continue
                # for debug
                if cWords[2] not in cSta:
                    cSta.append(cWords[2])
                rMJD = float(cWords[6]) + float(cWords[7])/86400.0
                MJD1 = min(MJD1, rMJD)
                MJD2 = max(MJD2, rMJD)
    MJD1 = int(MJD1-10)
    MJD2 = int(MJD2+10)
    cSta.sort()
    nSta = len(cSta)

    fig, axs = plt.subplots(nSta, 1, sharex='col',
                            squeeze=False, figsize=(8, nSta*1.5))
    fig.subplots_adjust(hspace=0.1)
    for i in range(nSta):
        # Extract OmC for this station
        cSta1 = []
        cSta1.append(cSta[i])
        cSat1 = []
        cSat1.append('ALL')
        X1, X2, cSta2, cSat2 = GetOmCSLR1(fOmc, cSta1, cSat1, [0, 99999])
        nBad = X2[1].size
        nOmC = X1[1].size
        Mea = np.mean(X1[1])
        Sig = np.std(X1[1])
        RMS = 0.0
        for j in range(nOmC):
            RMS = RMS+X1[1, j]*X1[1, j]
        RMS = np.sqrt(RMS/nOmC)

        axs[i, 0].plot(X1[5], X1[1], '.b', label=cSta[i])
        axs[i, 0].text(0.05, 0.95, cSta[i], transform=axs[i, 0].transAxes,
                       ha='left', va='top')
        axs[i, 0].set_xlim(left=0, right=90)
        axs[i, 0].set_ylabel('[cm]')
    axs[i, 0].set_xlabel('Elev [deg]')

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotOmCSLR8(fOmc, OutFilePrefix, OutFileSuffix):
    '''
    Plot SLR OmC for all stations, omc VS elevation angle
    '''

    fig, axs = plt.subplots(1, 1, figsize=(8, 3))

    # Extract OmC for all stations
    cSta1 = []
    cSta1.append('ALL')
    cSat1 = []
    cSat1.append('ALL')
    X1, X2, cSta2, cSat2 = GetOmCSLR1(fOmc, cSta1, cSat1, [0, 99999])
    axs.plot(X1[5], X1[1], '.b')
    axs.text(0.05, 0.95, 'ALL-ALL', transform=axs.transAxes,
             ha='left', va='top')
    axs.set_xlim(left=0, right=90)
    axs.set_ylabel('[cm]')
    axs.set_xlabel('Elev [deg]')

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotOmCSLR9(fOmc, OutFilePrefix, OutFileSuffix):
    '''
    Plot SLR OmC for each station-satellite pair, omc VS time && omc VS elevation angle
    '''

    # Get the whole satellite && station set
    cSat = []
    cSta = []
    MJD1 = 99999
    MJD2 = 0
    with open(fOmc, mode='rt') as fOb:
        for cLine in fOb:
            if cLine[0:4] == 'BAD ' or cLine[0:4] == 'OK  ':
                cWords = cLine.split()
                # for debug
                # if cWords[2]=='7249' or cWords[2]=='1868':
                #     continue
                # for debug
                if cWords[2] not in cSta:
                    cSta.append(cWords[2])
                if cWords[4] not in cSat:
                    cSat.append(cWords[4])
                rMJD = float(cWords[6]) + float(cWords[7])/86400.0
                MJD1 = min(MJD1, rMJD)
                MJD2 = max(MJD2, rMJD)
    MJD1 = int(MJD1-10)
    MJD2 = int(MJD2+10)
    cSta.sort()
    nSta = len(cSta)
    cSat0 = []
    cSatSLR = ['compassg1', 'compassi3', 'compassi5', 'compassi6b', 'compassm3',
               'beidou3m1', 'beidou3m2', 'beidou3m3', 'beidou3m9', 'beidou3m10']
    for i in range(len(cSatSLR)):
        if cSatSLR[i] in cSat:
            cSat0.append(cSatSLR[i])
    cSat = cSat0
    nSat = len(cSat)

    nPair = 0
    cPair = []
    for i in range(nSta):
        for j in range(nSat):
            cSta1 = []
            cSta1.append(cSta[i])
            cSat1 = []
            cSat1.append(cSat[j])
            X1, X2, cSta2, cSat2 = GetOmCSLR1(fOmc, cSta1, cSat1, [0, 99999])
            if X1[1].size > 0:
                nPair = nPair+1
                cPair.append(cSta[i]+'-'+cSat[j])

    fig, axs = plt.subplots(nPair, 2, sharex='col',
                            sharey='row', squeeze=False, figsize=(8, nPair*1.5))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.1)
    for i in range(nPair):
        cWords = cPair[i].split('-')
        cSta1 = []
        cSta1.append(cWords[0])
        cSat1 = []
        cSat1.append(cWords[1])
        X1, X2, cSta2, cSat2 = GetOmCSLR1(fOmc, cSta1, cSat1, [0, 99999])
        nBad = X2[1].size
        nOmC = X1[1].size

        axs[i, 0].plot(X1[0], X1[1], '.b')
        axs[i, 0].text(0.05, 0.95, cPair[i], transform=axs[i, 0].transAxes,
                       ha='left', va='top')
        axs[i, 0].text(0.95, 0.95, '#={:>6d}, nBad={:>6d}'.format(nOmC, nBad),
                       transform=axs[i, 0].transAxes, ha='right', va='top')
        axs[i, 0].set_xlim(left=MJD1, right=MJD2)
        axs[i, 0].set_ylabel('[cm]')

        axs[i, 1].plot(X1[5], X1[1], '.b')
        axs[i, 1].set_xlim(left=0, right=90)
    axs[i, 0].set_xlabel('Modified Julian Day')
    axs[i, 1].set_xlabel('Elev [deg]')

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotOmCSLR10(cSer, fOmcSer, lCommonSatOnly, cSatExl, iComp, yMax, lExBAD, rMax,
                 OutFilePrefix, OutFileSuffix):
    '''
    Plot SLR OmC Mean, STD and RMS for each satellite from different solutions

    lCommonSatOnly --- Whether show only the common satellites among differen
                       solutions
           cSatExl --- ILRS name list of satellites to be excluded
             iComp --- the component to be plotted
                       # 0, mean
                       # 1, std
                       # 2, RMS
                       # 3, all components in one figure
              yMax --- max of y-axis for Mean, STD and RMS in cm
            lExBAD --- Whether exclude all the BAD records
              rMax --- If great than zero, exclude all records (No matter whether
                       it is a BAD or OK record) that are great than this value in
                       the absolute sense, in cm

    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nSer = len(cSer)
    cSat = []
    RMS = []
    for iSer in range(nSer):
        for i in range(5):
            # For Mean, STD, RMS, nGod, nBad
            RMS.append([])

    for iSer in range(nSer):
        OmC1, OmC2, cSatSer = GetOmCSLR2(
            fOmcSer[iSer], [0, 99999], cSatExl, lExBAD, rMax)
        nSatSer = len(cSatSer)
        for i in range(nSatSer):
            if cSatSer[i] not in cSat:
                cSat.append(cSatSer[i])
                for j in range(nSer):
                    for k in range(5):
                        RMS[j*5+k].append(np.nan)
            iSat = cSat.index(cSatSer[i])
            # Mean, in cm
            RMS[iSer*5][iSat] = np.mean(OmC1[i*7+1])
            # STD, in cm
            RMS[iSer*5+1][iSat] = np.std(OmC1[i*7+1])
            # RMS
            nOmC = len(OmC1[i*7+1])
            xTmp = 0.0
            for j in range(nOmC):
                xTmp = xTmp + OmC1[i*7+1][j]*OmC1[i*7+1][j]
            RMS[iSer*5+2][iSat] = np.sqrt(xTmp/nOmC)
            # Number of good poins
            RMS[iSer*5+3][iSat] = nOmC
            # Number of bad poins
            RMS[iSer*5+4][iSat] = len(OmC2[i*7+1])
    nSat = len(cSat)
    cPRN = cSat.copy()
    # Report to the terminal
    strTmp = '{: <15s}'.format('Satellite')
    for j in range(nSer):
        strTmp = strTmp+' {: >15s} {: >15s} {: >15s} {: >15s} {: >15s}'.format(cSer[j]+'_Mea',
                                                                               cSer[j]+'_STD', cSer[j]+'_RMS', cSer[j]+'_God', cSer[j]+'_Bad')
    print(strTmp)
    for i in range(nSat):
        strTmp = '{: <15s}'.format(cSat[i])
        for j in range(nSer):
            strTmp = strTmp+' {: >15.2f} {: >15.2f} {: >15.2f} {: >15.0f} {: >15.0f}'.format(RMS[j*5][i],
                                                                                             RMS[j*5+1][i], RMS[j*5+2][i], RMS[j*5+3][i], RMS[j*5+4][i])
        print(strTmp)
    # Overall mean
    strTmp = '{: <15s}'.format('Mean')
    for j in range(nSer):
        strTmp = strTmp+' {: >15.2f} {: >15.2f} {: >15.2f} {: >15.0f} {: >15.0f}'.format(np.mean(np.fabs(RMS[j*5])),
                                                                                         np.mean(RMS[j*5+1]), np.mean(RMS[j*5+2]), np.mean(RMS[j*5+3]), np.mean(RMS[j*5+4]))
    print(strTmp)

    # Convert from ILRS name to PRN
    for i in range(nSat):
        if cPRN[i] == 'beidou3m2':
            cPRN[i] = 'C20'
        elif cPRN[i] == 'beidou3m3':
            cPRN[i] = 'C21'
        elif cPRN[i] == 'beidou3m9':
            cPRN[i] = 'C29'
        elif cPRN[i] == 'beidou3m10':
            cPRN[i] = 'C30'
        elif cPRN[i] == 'compassm3':
            cPRN[i] = 'C11'
        elif cPRN[i] == 'compassi6b':
            cPRN[i] = 'C13'
        elif cPRN[i] == 'compassi3':
            cPRN[i] = 'C08'
        elif cPRN[i] == 'compassi5':
            cPRN[i] = 'C10'
        elif cPRN[i] == 'compassg1':
            cPRN[i] = 'C01'

    # Satellite list that would be plotted
    nSat0 = 0
    iSat0 = []
    cSat0 = []
    if lCommonSatOnly:
        # Search for the common satellite list
        for i in range(nSat):
            lCommon = True
            for j in range(nSer):
                if np.isnan(RMS[j*5+2][i]):
                    lCommon = False
                    break
            if not lCommon:
                continue
            nSat0 = nSat0+1
            iSat0.append(i)
            cSat0.append(cPRN[i])
    else:
        for i in range(nSat):
            nSat0 = nSat0+1
            iSat0.append(i)
            cSat0.append(cPRN[i])
    if nSat0 == 0:
        sys.exit('No satellites would be plotted!')

    # Number of Cols for legend
    if nSer <= 5:
        nColLG = nSer
    else:
        nColLG = 5

    x = np.arange(nSat0)
    yLab = ['Mean [cm]', 'STD [cm]', 'RMS [cm]']
    if iComp != 3:
        # Only one of Mean, STD and RMS
        fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(nSat0*1.5, 2))

        axs[0, 0].set_xlim(left=-1, right=nSat0)
        if iComp == 0:
            # For Mean
            axs[0, 0].set_ylim(bottom=-yMax[iComp], top=yMax[iComp])
        elif iComp == 1:
            # For STD
            axs[0, 0].set_ylim(top=yMax[iComp])
        else:
            # For RMS
            axs[0, 0].set_ylim(top=yMax[iComp])

        # the width of the bars
        w = 1/(nSer+1)
        for i in range(nSer):
            y = []
            for k in range(nSat0):
                y.append(RMS[i*5+iComp][iSat0[k]])
            axs[0, 0].bar(x+(i-nSer/2)*w, y, w, align='edge', label=cSer[i])

        axs[0, 0].set_ylabel(yLab[iComp], fontname='Arial', fontsize=16)
        for tl in axs[0, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        axs[0, 0].grid(which='both', axis='y', color='darkgray', linestyle='--',
                       linewidth=0.8)
        axs[0, 0].set_axisbelow(True)
        axs[0, 0].legend(ncol=nColLG, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                         prop={'family': 'Arial', 'size': 14}, borderaxespad=0.1, framealpha=0.1,
                         columnspacing=1.0, handlelength=1.0, handletextpad=0.4)
        axs[0, 0].set_xlabel('Satellite PRNs', fontname='Arial', fontsize=16)
        axs[0, 0].set_xticks(x)
        axs[0, 0].set_xticklabels(
            cSat0, fontdict={'fontsize': 14, 'fontname': 'Arial'})
    else:
        fig, axs = plt.subplots(3, 1, squeeze=False,
                                sharex='col', figsize=(nSat0*1.5, 6))
        for j in range(3):
            axs[j, 0].set_xlim(left=-1, right=nSat0)
            if j == 0:
                # For Mean
                axs[j, 0].set_ylim(bottom=-yMax[j], top=yMax[j])
            elif j == 1:
                # For STD
                axs[j, 0].set_ylim(top=yMax[j])
            else:
                # For RMS
                axs[j, 0].set_ylim(top=yMax[j])

            # the width of the bars
            w = 1/(nSer+1)
            for i in range(nSer):
                y = []
                for k in range(nSat0):
                    y.append(RMS[i*5+j][iSat0[k]])
                axs[j, 0].bar(x+(i-nSer/2)*w, y, w, align='edge',
                              linewidth=0, label=cSer[i])

            axs[j, 0].set_ylabel(yLab[j], fontname='Arial', fontsize=16)
            axs[j, 0].yaxis.set_major_formatter('{x: >4.1f}')
            for tl in axs[j, 0].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            axs[j, 0].grid(which='both', axis='y', color='darkgray', linestyle='--',
                           linewidth=0.8)
            axs[j, 0].set_axisbelow(True)
        axs[0, 0].legend(ncol=nColLG, loc='lower center', bbox_to_anchor=(0.5, 1.0),
                         prop={'family': 'Arial', 'size': 14}, borderaxespad=0.1, framealpha=0.1,
                         columnspacing=1.0, handlelength=1.0, handletextpad=0.4)
        axs[j, 0].set_xlabel('Satellite PRNs', fontname='Arial', fontsize=16)
        axs[j, 0].set_xticks(x)
        axs[j, 0].set_xticklabels(
            cSat0, fontdict={'fontsize': 14, 'fontname': 'Arial'})

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotOmCSLR11(cSer, fOmcSer, cPRN0, lByCol, OutFilePrefix, OutFileSuffix):
    '''
    Plot SLR OmC series for specified satellites from different solutions

      cPRN0 --- PRNs of specified satellites


    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    cSatSLR = ['compassg1', 'compassi3', 'compassi5', 'compassi6b', 'compassm3',
               'beidou3m1', 'beidou3m2', 'beidou3m3', 'beidou3m9', 'beidou3m10']
    cSatPRN = ['C01', 'C08', 'C10', 'C13', 'C12',
               'C19', 'C20', 'C21', 'C29', 'C30']

    cSat = []
    cPRN = []
    for iSat in range(len(cPRN0)):
        if cPRN0[iSat] not in cSatPRN:
            continue
        cSat.append(cSatSLR[cSatPRN.index(cPRN0[iSat])])
        cPRN.append(cPRN0[iSat])
    nSat = len(cSat)
    nSer = len(cSer)

    if lByCol:
        fig, axs = plt.subplots(nSat, nSer, sharex='col', sharey='row',
                                squeeze=False, figsize=(3.5*nSer, nSat*2))
    else:
        fig, axs = plt.subplots(nSer, nSat, sharex='col', sharey='row',
                                squeeze=False, figsize=(3.5*nSat, nSer*2))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)

    for iSer in range(nSer):
        for iSat in range(nSat):
            OmC1, OmC2, cSta1, cSat1 = GetOmCSLR1(
                fOmcSer[iSer], ['ALL'], [cSat[iSat]], [0, 99999])
            if lByCol:
                iRow = iSat
                iCol = iSer
            else:
                iRow = iSer
                iCol = iSat
            axs[iRow, iCol].axhline(y=0, c='darkgray', ls='dashed', lw=0.4)
            axs[iRow, iCol].plot(OmC1[6], OmC1[1], '.b', ms=4)
            axs[iRow, iCol].text(0.02, 0.98, cPRN[iSat]+' '+cSer[iSer],
                                 transform=axs[iRow, iCol].transAxes, ha='left', va='top',
                                 fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
            axs[iRow, iCol].grid(which='major', axis='y',
                                 c='darkgray', ls='--', lw=0.8, alpha=0.5)
            # y-label, only for the first column
            if iCol == 0:
                axs[iRow, iCol].set_ylim(-25, 25)
                axs[iRow, iCol].set_ylabel(
                    '[cm]', fontname='Arial', fontsize=16)
                axs[iRow, iCol].yaxis.set_major_formatter('{x: >3.0f}')
                for tl in axs[iRow, iCol].get_yticklabels():
                    tl.set_fontname('Arial')
                    tl.set_fontsize(14)
            # x-lable, only for the last row
            if (lByCol and iRow == (nSat-1)) or (not lByCol and iRow == (nSer-1)):
                axs[iRow, iCol].set_xlim(left=0, right=180)
                axs[iRow, iCol].set_xticks([30, 60, 90, 120, 150])
                axs[iRow, iCol].set_xlabel(
                    r'$\epsilon$ [deg]', fontname='Arial', fontsize=16)
                for tl in axs[iRow, iCol].get_xticklabels():
                    tl.set_fontname('Arial')
                    tl.set_fontsize(14)
    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
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

    cSer = []
    fOmcSer = []

    # OutFilePrefix='D:/Code/PROJECT/WORK_SLR/'
    # OutFileSuffix='Log.pdf'

    fOmc = os.path.join(
        cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_BE4')
    # OutFilePrefix='D:/Code/PROJECT/WORK_BDSOBS/'
    OutFilePrefix = os.path.join(cDskPre0, r'PRO_2019001_2020366/I448/ORB/')

    # OutFileSuffix='SLRRes1'
    # PlotOmCSLR1(fOmc,[58818,58849],OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='SLRRes4'
    # PlotOmCSLR4(fOmc,[0,99999],OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='SLRRes5'
    # PlotOmCSLR5(fOmc,True,['C20','C21','C29','C30'],2,OutFilePrefix,OutFileSuffix)

    # cSer.append('None')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_J600'))
    # cSer.append('Model 1')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_J601'))
    # cSer.append('Model 2')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_J602'))
    # cSer.append('Model 3')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_J603'))
    cSer.append('Reg')
    fOmcSer.append(os.path.join(
        cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_I121'))
    cSer.append('Reg+ISL')
    fOmcSer.append(os.path.join(
        cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_I448'))
    # cSer.append('1 cm')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_J640'))
    # cSer.append('2 cm')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_J641'))
    # cSer.append('4 cm')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_J642'))
    # cSer.append('8 cm')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_J646'))
    # cSer.append('12 cm')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_J647'))
    # cSer.append('15 cm')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_J648'))
    # cSer.append('18 cm')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_J651'))
    # cSer.append('20 cm')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_J653'))
    # cSer.append('No ISL')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_C01'))
    # cSer.append('ISL Rng')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_I648'))
    # cSer.append('ISL Clk')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_J642'))
    # cSer.append('ISL Rng+Clk')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_D650'))
    # cSer.append('E1P5')
    # fOmcSer.append(os.path.join(cDskPre0,r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_I1_c'))
    # cSer.append('BW+E1P5')
    # fOmcSer.append(os.path.join(cDskPre0,r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_I1_d_YAN'))
    # cSer.append('E2P7')
    # fOmcSer.append(os.path.join(cDskPre0,r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_E1'))
    # cSer.append('BW+E2P7')
    # fOmcSer.append(os.path.join(cDskPre0,r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_BE1_YAN'))
    # cSer.append('E2P9')
    # fOmcSer.append(os.path.join(cDskPre0,r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_E2'))
    # cSer.append('BW+E2P9')
    # fOmcSer.append(os.path.join(cDskPre0,r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_BE2_YAN'))
    # cSer.append('SatE')
    # fOmcSer.append(os.path.join(cDskPre0,r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_E4'))
    # cSer.append('BW+SatE')
    # fOmcSer.append(os.path.join(cDskPre0,r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_BE4_YAN'))

    # cSer.append('None')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_I610'))
    # cSer.append('Model1')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_I611'))
    # cSer.append('Model2')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_I612'))
    # cSer.append('Model3')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_I613'))
    # cSer.append('A-Priori')
    # fOmcSer.append(os.path.join(
    #     cDskPre0, r'GNSS/PROJECT/SLRCheck/slr_2019335_2019365_I614'))

    OutFileSuffix = 'SLRResComp_RMS'
    PlotOmCSLR10(cSer, fOmcSer, True, [], 2, [10, 10, 40], False, -100,
                 OutFilePrefix, OutFileSuffix)
    # OutFileSuffix='SLRRes11'
    # PlotOmCSLR11(cSer,fOmcSer,['C20','C21','C29','C30'],True,OutFilePrefix,OutFileSuffix)
