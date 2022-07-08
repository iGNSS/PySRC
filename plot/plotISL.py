#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
'''
__author__ = 'hanbing'

# Standard library imports
import os
import sys
import glob
import datetime
import math

# Related third party imports
import numpy as np
from numpy.core.fromnumeric import transpose
import numpy.ma as ma
from numpy import dtype, linalg as NupLA, matmul
from scipy import linalg as SciLA
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.ticker import FormatStrFormatter

from astropy import stats
import allantools

# Local application/library specific imports
from PySRC.miscellaneous import GNSSTime


def GetLinkTab(fISLList):
    '''
    Get the number of observations for each Inter-Satellite link

    Input :

    Return:
    '''
    nFile = len(fISLList)
    cSat = []
    for i in range(nFile):
        with open(fISLList[i], mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine) < 30:
                    continue
                cWords = cLine.split()
                if cWords[3][0:3] not in cSat:
                    cSat.append(cWords[3][0:3])
                if cWords[3][4:7] not in cSat:
                    cSat.append(cWords[3][4:7])
    cSat.sort()
    nSat = len(cSat)
    nObs = np.zeros((nSat, nSat), dtype=np.uint32)
    for i in range(nFile):
        with open(fISLList[i], mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine) < 30:
                    continue
                cWords = cLine.split()
                k = cSat.index(cWords[3][0:3])
                j = cSat.index(cWords[3][4:7])
                nObs[k, j] = nObs[k, j]+1

    return cSat, nObs


def MergeLink(cSat, nObs):
    '''
    Merge the links that have same ends without distinguishing
    trans or receive end
    '''

    for i in range(len(cSat)):
        for j in range(0, i):
            if nObs[i, j] == 0:
                continue
            nObs[j, i] = nObs[j, i] + nObs[i, j]
            nObs[i, j] = 0
    return nObs


def GetLinkClk(fISLList, cLink, lMerge):
    '''
    Extract ISL relative clock observations (in meters) for a specified link

    cLink --- the specified ISL link
   lMerge --- Whether ignore the difference between trans && recv end
    '''

    nFile = len(fISLList)
    rEpo = []
    Clk = []
    for i in range(nFile):
        with open(fISLList[i], mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine) < 30:
                    continue
                cWords = cLine.split()
                # Exclude C76
                if cWords[3][0:3] == 'C76' or cWords[3][4:7] == 'C76':
                    continue
                cTmp1 = cWords[3][0:3]+'-'+cWords[3][4:7]
                cTmp2 = cWords[3][4:7]+'-'+cWords[3][0:3]
                if cLink == cTmp1 or (lMerge and cLink == cTmp2):
                    rEpo.append(int(cWords[0])+float(cWords[1])/86400.0)
                    #Clock in meters
                    Clk.append(float(cWords[5])*1E-9*299792458.0)
    x = np.array(rEpo)
    y = np.array(Clk)
    return x, y


def CalNEQ(fISLList, lRng, lClk, iMod, lCons):
    '''
    Do some experiments on the ISL design && norm matrix

    lRng --- Whether use range obs
    lClk --- Whether use clock obs
    iMod --- Model Number in different cases
   lCons --- Whether apply constraint on the hardware delay of
             a reference satellite when only clock obs used
    '''
    if not lRng and not lClk:
        sys.exit('No observation type specified')

    nFile = len(fISLList)
    # Get the number of satellite and observations
    cSat = []
    nObs = 0
    cLink = []
    for i in range(nFile):
        with open(fISLList[i], mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine) < 30:
                    continue
                if lRng and lClk:
                    # Both range and clock observations used
                    nObs = nObs+2
                elif (lRng and not lClk) or (not lRng and lClk):
                    # Only range or clock observations used
                    nObs = nObs+1
                cWords = cLine.split()
                if cWords[3][0:3] not in cSat:
                    cSat.append(cWords[3][0:3])
                if cWords[3][4:7] not in cSat:
                    cSat.append(cWords[3][4:7])
                if cWords[3][0:3] < cWords[3][4:7]:
                    cTmp = cWords[3][0:3]+'-'+cWords[3][4:7]
                else:
                    cTmp = cWords[3][4:7]+'-'+cWords[3][0:3]
                if cTmp not in cLink:
                    cLink.append(cTmp)

    cSat.sort()
    nSat = len(cSat)
    nLink = len(cLink)
    print('{:9d} Obs, {:3d} Sat, {:4d} Link'.format(nObs, nSat, nLink))

    # Get the design matrix for relative observations
    if lRng and lClk:
        # Use Rng && Clk obs simultaneously
        # recv, tran delay for each satellite
        A = np.zeros((nObs, 2*nSat))
    elif (lRng and not lClk):
        # Use only Rng obs
        if iMod == 0:
            # tran+recv delay for each satellite
            A = np.zeros((nObs, nSat))
        elif iMod == 1:
            # tran+recv delay for each satellite, 0-order link mode
            A = np.zeros((nObs, nSat+nLink*1))
        elif iMod == 2:
            # tran+recv delay for each satellite, 1-order link mode
            A = np.zeros((nObs, nSat+nLink*2))
        elif iMod == 3:
            # tran+recv delay for each satellite, 0- and 1-order link mode
            A = np.zeros((nObs, nSat+nLink*3))
    else:
        # Use only Clk obs
        if iMod == 0:
            # tran-recv delay for each satellite
            A = np.zeros((nObs, nSat))

    iObs = 0
    for i in range(nFile):
        with open(fISLList[i], mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine) < 30:
                    continue
                cWords = cLine.split()
                iSat1 = cSat.index(cWords[3][0:3])
                iSat2 = cSat.index(cWords[3][4:7])
                if cWords[3][0:3] < cWords[3][4:7]:
                    cTmp = cWords[3][0:3]+'-'+cWords[3][4:7]
                else:
                    cTmp = cWords[3][4:7]+'-'+cWords[3][0:3]
                iLink = cLink.index(cTmp)
                if (lRng and lClk):
                    # Both range && clock
                    if iMod == 1:
                        # Estimate the tran && recv delay for each satellite

                        # Relative range observation
                        # Tran && Recv delay of the first satellite
                        A[iObs, 2*iSat1] = 0.5
                        A[iObs, 2*iSat1+1] = 0.5
                        # Tran && Recv delay of the second satellite
                        A[iObs, 2*iSat2] = 0.5
                        A[iObs, 2*iSat2+1] = 0.5
                        # Relative clock observations
                        # Tran && Recv delay of the first satellite
                        A[iObs+1, 2*iSat1] = 0.5
                        A[iObs+1, 2*iSat1+1] = -0.5
                        # Tran && Recv delay of the second satellite
                        A[iObs+1, 2*iSat2] = -0.5
                        A[iObs+1, 2*iSat2+1] = 0.5
                    else:
                        # Estimate the sum && diff of the tran and recv delay for each satellite

                        # Relative range observation
                        # Sum of Tran && Recv delay
                        A[iObs, 2*iSat1] = 0.5
                        A[iObs, 2*iSat2] = 0.5
                        # Relative clock observations
                        # Diff of Tran && Recv delay
                        A[iObs+1, 2*iSat1+1] = 0.5
                        A[iObs+1, 2*iSat2+1] = -0.5
                    iObs = iObs+2
                elif (lRng and not lClk):
                    # Only range, sum of Tran && Recv delay
                    if iMod == 0:
                        A[iObs, iSat1] = 0.5
                        A[iObs, iSat2] = 0.5
                    elif iMod == 1:
                        A[iObs, iSat1] = 0.5
                        A[iObs, iSat2] = 0.5
                        A[iObs, iLink] = 1.0
                    iObs = iObs+1
                elif (not lRng and lClk):
                    # Only clock
                    if iMod == 0:
                        # diff of Tran && Recv delay
                        A[iObs, iSat1] = 0.5
                        A[iObs, iSat2] = -0.5
                    iObs = iObs+1
    # Normal matrx
    N = np.matmul(np.transpose(A), A)
    if (not lRng and lClk):
        if lCons:
            # Apply constraints on reference sat
            iSat = cSat.index('C19')
            N[iSat, iSat] = N[iSat, iSat] + 1e9

    # Rank && Conditional number of the normal matrix
    print('Rank: ', NupLA.matrix_rank(N))
    print('Cond: ', NupLA.cond(N))
    # Determinant of the normal matrix
    print('Det:  ', SciLA.det(N))


def PlotSatLinkNum0(fISLList, cPRN0, OutFilePrefix, OutFileSuffix):
    '''
    Plot link numbers for each satellite, distingushing intra-plane
    and inter-plane links.

    cPRN0 --- Satellite filter

    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # BDS-3 Satellites in the same orbit. (NOTE: As PRNs are used, this table should be
    # updated from time to time)
    nPlane = 7
    cPlane = [['C27', 'C28', 'C29', 'C30', 'C34', 'C35', 'C43', 'C44'],
              ['C19', 'C20', 'C21', 'C22', 'C32', 'C33', 'C41', 'C42'],
              ['C23', 'C24', 'C36', 'C37', 'C45', 'C46', 'C25', 'C26'],
              ['C59', 'C60', 'C61'], ['C38'], ['C39'], ['C40']]

    nFile = len(fISLList)
    cSat = []
    cLink = []
    for i in range(nFile):
        with open(fISLList[i], mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine) < 30:
                    continue
                cWords = cLine.split()
                if cPRN0[0] == 'EXCL':
                    # Excluded satellites
                    if cWords[3][0:3] in cPRN0[1:] or cWords[3][4:7] in cPRN0[1:]:
                        continue
                if cWords[3][0:3] not in cSat:
                    cSat.append(cWords[3][0:3])
                    # intra-/inter-plane link list for this satellite
                    cLink.append([])
                    cLink.append([])
                iSat = cSat.index(cWords[3][0:3])
                iOrb = -1
                # Determine the orbit plane for this satellite
                for j in range(nPlane):
                    if cSat[iSat] not in cPlane[j]:
                        continue
                    iOrb = j
                    break
                if iOrb == -1:
                    sys.exit('Unkown orbit plane for satellite '+cSat[iSat])
                if cWords[3][0:3]+'-'+cWords[3][4:7] not in cLink[2*iSat] and \
                   cWords[3][0:3]+'-'+cWords[3][4:7] not in cLink[2*iSat+1] and \
                   cWords[3][4:7]+'-'+cWords[3][0:3] not in cLink[2*iSat] and \
                   cWords[3][4:7]+'-'+cWords[3][0:3] not in cLink[2*iSat+1]:
                    # New link for this satellite
                    if cWords[3][4:7] in cPlane[iOrb]:
                        # intra-plane link
                        cLink[2*iSat].append(cWords[3])
                    else:
                        # inter-plane link
                        cLink[2*iSat+1].append(cWords[3])
                # Same procedure for the other satellite
                if cWords[3][4:7] not in cSat:
                    cSat.append(cWords[3][4:7])
                    cLink.append([])
                    cLink.append([])
                iSat = cSat.index(cWords[3][4:7])
                iOrb = -1
                for j in range(nPlane):
                    if cSat[iSat] not in cPlane[j]:
                        continue
                    iOrb = j
                    break
                if iOrb == -1:
                    sys.exit('Unkown orbit plane for satellite '+cSat[iSat])
                if cWords[3][0:3]+'-'+cWords[3][4:7] not in cLink[2*iSat] and \
                   cWords[3][0:3]+'-'+cWords[3][4:7] not in cLink[2*iSat+1] and \
                   cWords[3][4:7]+'-'+cWords[3][0:3] not in cLink[2*iSat] and \
                   cWords[3][4:7]+'-'+cWords[3][0:3] not in cLink[2*iSat+1]:
                    if cWords[3][0:3] in cPlane[iOrb]:
                        cLink[2*iSat].append(cWords[3])
                    else:
                        cLink[2*iSat+1].append(cWords[3])
    nSat = len(cSat)
    cSat0 = cSat.copy()
    cSat0.sort()

    # Report to the terminal
    print('{: <3s} {: >5s} {: >5s} {: >5s}'.format(
        'PRN', 'Intra', 'Inter', 'All'))
    nLink = np.zeros((nSat, 2), dtype=np.int32)
    for i in range(nSat):
        j = cSat.index(cSat0[i])
        nLink[i, 0] = len(cLink[2*j])
        nLink[i, 1] = len(cLink[2*j+1])
        print('{: <3s} {: >5d} {: >5d} {: >5d}'.format(cSat0[i], nLink[i, 0], nLink[i, 1],
              nLink[i, 0]+nLink[i, 1]))

    x = np.arange(nSat)
    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(nSat*0.6, 4))

    axs[0, 0].set_xlim(left=-1, right=nSat)

    # the width of the bars
    w = 1/(1+1)
    # number of intra-plane links
    axs[0, 0].bar(x+(0-1/2)*w, nLink[:, 0], w,
                  align='edge', label='Intra-Plane')
    # number of inter-plane links
    axs[0, 0].bar(x+(0-1/2)*w, nLink[:, 1], w, align='edge',
                  label='Inter-Plane', bottom=nLink[:, 0])

    axs[0, 0].legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                     prop={'family': 'Arial', 'size': 14}, framealpha=0.3)
    axs[0, 0].grid(which='both', axis='y', color='darkgray',
                   linestyle='--', linewidth=0.4)
    axs[0, 0].set_axisbelow(True)
    axs[0, 0].set_ylabel('Number of links', fontname='Arial', fontsize=16)
    axs[0, 0].yaxis.get_major_locator().set_params(integer=True)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(
        cSat0, fontdict={'fontsize': 14, 'fontname': 'Arial'})
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotSatLinkNum1(fISLList, cPRN0, OutFilePrefix, OutFileSuffix):
    '''
    Similar as PlotSatLinkNum1, but plot for each file

    cPRN0 --- Satellite filter

    '''

    # BDS-3 Satellites in the same orbit. (NOTE: As PRNs are used, this table should be
    # updated from time to time)
    nPlane = 7
    cPlane = [['C27', 'C28', 'C29', 'C30', 'C34', 'C35', 'C43', 'C44'],
              ['C19', 'C20', 'C21', 'C22', 'C32', 'C33', 'C41', 'C42'],
              ['C23', 'C24', 'C36', 'C37', 'C45', 'C46', 'C25', 'C26'],
              ['C59', 'C60', 'C61'], ['C38'], ['C39'], ['C40']]

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    strTmp = os.path.join(OutFilePrefix, OutFileSuffix+'.pdf')
    with PdfPages(strTmp) as pdf:
        for fISL in fISLList:
            cSat = []
            cLink = []
            with open(fISL, mode='rt') as fOb:
                for cLine in fOb:
                    if len(cLine) < 30:
                        continue
                    cWords = cLine.split()
                    if cPRN0[0] == 'EXCL':
                        # Excluded satellites
                        if cWords[3][0:3] in cPRN0[1:] or cWords[3][4:7] in cPRN0[1:]:
                            continue
                    if cWords[3][0:3] not in cSat:
                        cSat.append(cWords[3][0:3])
                        # intra-/inter-plane link list for this satellite
                        cLink.append([])
                        cLink.append([])
                    iSat = cSat.index(cWords[3][0:3])
                    iOrb = -1
                    # Determine the orbit plane for this satellite
                    for j in range(nPlane):
                        if cSat[iSat] not in cPlane[j]:
                            continue
                        iOrb = j
                        break
                    if iOrb == -1:
                        sys.exit(
                            'Unkown orbit plane for satellite '+cSat[iSat])
                    if cWords[3][0:3]+'-'+cWords[3][4:7] not in cLink[2*iSat] and \
                            cWords[3][0:3]+'-'+cWords[3][4:7] not in cLink[2*iSat+1] and \
                            cWords[3][4:7]+'-'+cWords[3][0:3] not in cLink[2*iSat] and \
                            cWords[3][4:7]+'-'+cWords[3][0:3] not in cLink[2*iSat+1]:
                        # New link for this satellite
                        if cWords[3][4:7] in cPlane[iOrb]:
                            # intra-plane link
                            cLink[2*iSat].append(cWords[3])
                        else:
                            # inter-plane link
                            cLink[2*iSat+1].append(cWords[3])
                    # Same procedure for the other satellite
                    if cWords[3][4:7] not in cSat:
                        cSat.append(cWords[3][4:7])
                        cLink.append([])
                        cLink.append([])
                    iSat = cSat.index(cWords[3][4:7])
                    iOrb = -1
                    for j in range(nPlane):
                        if cSat[iSat] not in cPlane[j]:
                            continue
                        iOrb = j
                        break
                    if iOrb == -1:
                        sys.exit(
                            'Unkown orbit plane for satellite '+cSat[iSat])
                    if cWords[3][0:3]+'-'+cWords[3][4:7] not in cLink[2*iSat] and \
                            cWords[3][0:3]+'-'+cWords[3][4:7] not in cLink[2*iSat+1] and \
                            cWords[3][4:7]+'-'+cWords[3][0:3] not in cLink[2*iSat] and \
                            cWords[3][4:7]+'-'+cWords[3][0:3] not in cLink[2*iSat+1]:
                        if cWords[3][0:3] in cPlane[iOrb]:
                            cLink[2*iSat].append(cWords[3])
                        else:
                            cLink[2*iSat+1].append(cWords[3])
            nSat = len(cSat)
            cSat0 = cSat.copy()
            cSat0.sort()
            nLink = np.zeros((nSat, 2), dtype=np.int32)
            for i in range(nSat):
                j = cSat.index(cSat0[i])
                nLink[i, 0] = len(cLink[2*j])
                nLink[i, 1] = len(cLink[2*j+1])
            x = np.arange(nSat)
            fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(nSat*0.6, 4))
            axs[0, 0].text(0.50, 1.00, fISL, transform=axs[0, 0].transAxes, ha='center', va='bottom',
                           fontdict={'fontsize': 10, 'fontname': 'Arial', 'fontweight': 'bold'})

            axs[0, 0].set_xlim(left=-1, right=nSat)

            # the width of the bars
            w = 1/(1+1)
            # number of intra-plane links
            axs[0, 0].bar(x+(0-1/2)*w, nLink[:, 0], w,
                          align='edge', label='Intra-Plane')
            # number of inter-plane links
            axs[0, 0].bar(x+(0-1/2)*w, nLink[:, 1], w, align='edge',
                          label='Inter-Plane', bottom=nLink[:, 0])

            axs[0, 0].legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                             prop={'family': 'Arial', 'size': 14}, framealpha=0.3)
            axs[0, 0].grid(which='both', axis='y',
                           color='darkgray', linestyle='--', linewidth=0.4)
            axs[0, 0].set_axisbelow(True)
            axs[0, 0].set_ylabel(
                'Number of links', fontname='Arial', fontsize=16)
            axs[0, 0].yaxis.get_major_locator().set_params(integer=True)
            axs[0, 0].set_xticks(x)
            axs[0, 0].set_xticklabels(
                cSat0, fontdict={'fontsize': 14, 'fontname': 'Arial'})
            for tl in axs[0, 0].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def PlotLinkTab0(fISLList, lMerge, OutFilePrefix, OutFileSuffix):
    '''
    Plot the number of observations for each link
    One pdf page for each file
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    strTmp = os.path.join(OutFilePrefix, OutFileSuffix+'.pdf')
    with PdfPages(strTmp) as pdf:
        for fISL in fISLList:
            # Get the date from the file name
            YYYY = int(fISL[-7:-3])
            DOY = int(fISL[-3:])
            cSat, nObs = GetLinkTab([fISL])
            if lMerge:
                # Merge links that have same ends
                MergeLink(cSat, nObs)
            nSat = len(cSat)

            fig, axs = plt.subplots(
                2, 1, squeeze=False, figsize=(0.8*nSat, 1.0*nSat))
            fig.suptitle('{:4d}{:03d}'.format(YYYY, DOY),
                         fontfamily='Arial', fontsize=18, fontweight='bold')
            fig.subplots_adjust(hspace=0)

            k = 0
            for i in range(nSat):
                for j in range(nSat):
                    if nObs[i, j] == 0:
                        continue
                    k = k+1
            print('Number of links having obs: {:>4d} in '.format(k)+fISL)

            x = np.arange(-0.5, nSat, 1)
            y = np.arange(-0.5, nSat, 1)
            qm = axs[0, 0].pcolormesh(x, y, ma.masked_equal(nObs, 0),
                                      vmin=np.min(ma.masked_equal(nObs, 0)),
                                      vmax=np.max(ma.masked_equal(nObs, 0)),
                                      shading='flat')
            axs[0, 0].xaxis.set_major_locator(MultipleLocator(1))
            axs[0, 0].yaxis.set_major_locator(MultipleLocator(1))
            axs[0, 0].set_xticks(range(nSat))
            axs[0, 0].set_xticklabels(
                cSat, fontdict={'fontsize': 14, 'fontname': 'Arial'})
            axs[0, 0].set_yticks(range(nSat))
            axs[0, 0].set_yticklabels(
                cSat, fontdict={'fontsize': 14, 'fontname': 'Arial'})
            axs[0, 0].invert_yaxis()
            axs[0, 0].xaxis.set_ticks_position('top')
            cb = fig.colorbar(qm, ax=axs[0, 0], location='bottom', pad=0.05,
                              aspect=30, anchor=(0.5, 0.0), panchor=(0.5, 0.0))
            for tl in cb.ax.get_xticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            cellT = []
            for i in range(len(cSat)):
                cellT.append(['%6d' % x for x in nObs[i, :]])
            axs[1, 0].axis('off')
            LinkTab = axs[1, 0].table(cellText=cellT,
                                      rowLabels=cSat, rowLoc='center',
                                      colLabels=cSat, colLoc='center',
                                      loc='upper left')
            # for tl in axs[1,0].get_yticklabels():
            #     tl.set_fontname('Arial'); tl.set_fontsize(14); tl.set_fontweight('bold')
            LinkTab.AXESPAD = 0
            LinkTab.scale(1.0, 1.5)
            LinkTab.auto_set_font_size(False)
            for iRow in range(nSat+1):
                for iCol in range(-1, nSat):
                    if iRow == 0 and iCol == -1:
                        continue
                    cell = LinkTab[iRow, iCol]
                    if (iRow == 0) or iCol == -1:
                        # Row-/Col-header labels
                        cell.set_text_props(
                            family='Arial', size=14, weight='bold')
                    else:
                        cell.set_text_props(
                            family='Arial', size=10, weight='normal')
                    # Set the frame
                    if iRow == 0 or iCol == -1:
                        cell.visible_edges = ''
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def PlotLinkTab1(fISLList, lMerge, OutFilePrefix, OutFileSuffix):
    '''
    Plot the number of observations for each link
    One figure for all files!
    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    cSat, nObs = GetLinkTab(fISLList)
    if lMerge:
        # Merge links that have same ends
        MergeLink(cSat, nObs)
    nSat = len(cSat)

    fig, axs = plt.subplots(2, 1, squeeze=False, figsize=(0.8*nSat, 1.0*nSat))
    fig.subplots_adjust(hspace=0)

    k = 0
    for i in range(nSat):
        for j in range(nSat):
            if nObs[i, j] == 0:
                continue
            k = k+1
    print('Number of links having obs: {:>4d}'.format(k))

    x = np.arange(-0.5, nSat, 1)
    y = np.arange(-0.5, nSat, 1)
    qm = axs[0, 0].pcolormesh(x, y, ma.masked_equal(nObs, 0),
                              vmin=np.min(ma.masked_equal(nObs, 0)),
                              vmax=np.max(ma.masked_equal(nObs, 0)),
                              shading='flat')
    axs[0, 0].xaxis.set_major_locator(MultipleLocator(1))
    axs[0, 0].yaxis.set_major_locator(MultipleLocator(1))
    axs[0, 0].set_xticks(range(nSat))
    axs[0, 0].set_xticklabels(
        cSat, fontdict={'fontsize': 14, 'fontname': 'Arial'})
    axs[0, 0].set_yticks(range(nSat))
    axs[0, 0].set_yticklabels(
        cSat, fontdict={'fontsize': 14, 'fontname': 'Arial'})
    axs[0, 0].invert_yaxis()
    axs[0, 0].xaxis.set_ticks_position('top')

    cb = fig.colorbar(qm, ax=axs[0, 0], orientation='horizontal', pad=0.05,
                      aspect=30, anchor=(0, 0), panchor=(0, 0))
    for tl in cb.ax.get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    cellT = []
    for i in range(len(cSat)):
        cellT.append(['%6d' % x for x in nObs[i, :]])

    axs[1, 0].axis('off')
    LinkTab = axs[1, 0].table(cellText=cellT,
                              rowLabels=cSat, rowLoc='center',
                              colLabels=cSat, colLoc='center',
                              loc='upper left')
    # for tl in axs[1,0].get_yticklabels():
    #     tl.set_fontname('Arial'); tl.set_fontsize(14); tl.set_fontweight('bold')
    LinkTab.AXESPAD = 0
    LinkTab.scale(1.0, 1.5)
    LinkTab.auto_set_font_size(False)
    for iRow in range(nSat+1):
        for iCol in range(-1, nSat):
            if iRow == 0 and iCol == -1:
                continue
            cell = LinkTab[iRow, iCol]
            if (iRow == 0) or iCol == -1:
                # Row-/Col-header labels
                cell.set_text_props(family='Arial', size=14, weight='bold')
            else:
                cell.set_text_props(family='Arial', size=10, weight='normal')
            # Set the frame
            if iRow == 0 or iCol == -1:
                cell.visible_edges = ''

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotLinkClk(fISLList, cLink0, lMerge, lFit, nDeg, rWin, rSlip, nCol, OutFilePrefix, OutFileSuffix):
    '''
    Plot relative clock series of specific ISL links one by one

    cLink0 --- Specific links list
    lMerge --- Whether ignore the difference between trans && recv end
      lFit --- Whether de-trend the series with polynomial fitting
      nDeg --- Degree of the fitting polynomial
      rWin --- Length of the window for fitting, in fractional day
     rSlip --- Length of the slip for moving window, in fractional day
      nCol --- Number of columns for the figure
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nFile = len(fISLList)

    # Find all links
    cLink = []
    for i in range(nFile):
        with open(fISLList[i], mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine) < 30:
                    continue
                cWords = cLine.split()
                # Exclude links related to C76
                if cWords[3][0:3] == 'C76' or cWords[3][4:7] == 'C76':
                    continue
                cTmp1 = cWords[3][0:3]+'-'+cWords[3][4:7]
                cTmp2 = cWords[3][4:7]+'-'+cWords[3][0:3]
                if cLink0[0] != 'ALL-ALL':
                    # Not all links
                    if cTmp1 not in cLink0:
                        if lMerge:
                            if cTmp2 not in cLink0:
                                if (cWords[3][0:3]+'-ALL' not in cLink0) and \
                                   ('ALL-'+cWords[3][0:3] not in cLink0) and \
                                   ('ALL-'+cWords[3][4:7] not in cLink0) and \
                                   (cWords[3][4:7]+'-ALL' not in cLink0):
                                    continue
                        else:
                            if (cWords[3][0:3]+'-ALL' not in cLink0) and \
                               ('ALL-'+cWords[3][4:7] not in cLink0):
                                continue
                if cTmp1 in cLink:
                    continue
                elif lMerge:
                    if cTmp2 in cLink:
                        continue
                cLink.append(cTmp1)
    cLink.sort()
    nLink = len(cLink)

    fig, axs = plt.subplots(nLink, 1, sharex='col',
                            squeeze=False, figsize=(8, nLink*2))
    # fig.subplots_adjust(hspace=0.1)
    formatterx = mpl.ticker.StrMethodFormatter('{x:7.1f}')

    for i in range(nLink):
        rEpo, Clk = GetLinkClk(fISLList, cLink[i], lMerge)
        if lFit:
            # Piece-Wise polynomial fitting
            ind = np.argsort(rEpo)
            lFirst = True
            j = 0
            t0 = rEpo[ind[0]]-rSlip
            while j < rEpo.size:
                # Slip the start point of the window
                if (rEpo[ind[j]]-t0) < rSlip:
                    j = j+1
                    continue
                # Inithialize the window
                t0 = rEpo[ind[j]]
                k = j
                rEpo0 = []
                Clk0 = []
                while k < rEpo.size:
                    if rEpo[ind[k]] <= (t0+rWin):
                        rEpo0.append(rEpo[ind[k]])
                        Clk0.append(Clk[ind[k]])
                        k = k+1
                    else:
                        break
                nP0 = len(rEpo0)
                # Report the number of available point within each window
                if nP0 < (nDeg+1):
                    MJD1 = int(t0)
                    SOD1 = (t0-MJD1)*86400
                    MJD2 = int(t0+rWin)
                    SOD2 = (t0+rWin-MJD2)*86400
                    strTmp = '#' + \
                        cLink[i]+' {:5d} {:8.2f} - {:5d} {:8.2f} {:5d}'.format(
                            MJD1, SOD1, MJD2, SOD2, nP0)
                    print(strTmp)
                else:
                    MJD1 = int(t0)
                    SOD1 = (t0-MJD1)*86400
                    MJD2 = int(t0+rWin)
                    SOD2 = (t0+rWin-MJD2)*86400
                    strTmp = ' ' + \
                        cLink[i]+' {:5d} {:8.2f} - {:5d} {:8.2f} {:5d}'.format(
                            MJD1, SOD1, MJD2, SOD2, nP0)
                    print(strTmp)

                    x = np.array(rEpo0)
                    y = np.array(Clk0)
                    c = np.polynomial.polynomial.polyfit(x, y, nDeg)
                    y_fit = np.polynomial.polynomial.polyval(x, c)
                    if lFirst:
                        lFirst = False
                        # m -> cm
                        tClk = x
                        VClk = (y-y_fit)*1e2
                    else:
                        # m -> cm
                        tClk = np.append(tClk, x)
                        VClk = np.append(VClk, (y-y_fit)*1e2)
            # Report possible blunders and cal the RMS (in cm)
            nP = tClk.size
            nGood = 0
            xtmp = 0.0
            for j in range(nP):
                # More than 10 meters
                if np.abs(VClk[j]) >= 10*1e2:
                    MJD = int(tClk[j])
                    SOD = (tClk[j]-MJD)*86400
                    strTmp = 'Blunder ' + \
                        cLink[i] + \
                        ' {:5d} {:8.2f} {:12.3f}'.format(MJD, SOD, VClk[j])
                    print(strTmp)
                    VClk[j] = np.nan
                else:
                    nGood = nGood+1
                    xtmp = xtmp+VClk[j]*VClk[j]
            if nGood > 0:
                RMS = np.sqrt(xtmp/nGood)
                # Cal the std && rms, in cm
                Mea = np.nanmean(VClk)
                Sig = np.nanstd(VClk)
                axs[i, 0].text(0.98, 0.98, '{:>6.1f}+/-{:>5.1f}, RMS={:>6.1f}, #={:>6d}'.format(Mea, Sig, RMS, nP),
                               transform=axs[i, 0].transAxes, ha='right', va='top',
                               fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
            rEpo = tClk
            Clk = VClk

        axs[i, 0].plot(rEpo, Clk, '.r', ms=4)
        if not lFit:
            axs[i, 0].set_ylabel(
                'Rel. clock [m]', fontname='Arial', fontsize=16)
        else:
            axs[i, 0].set_ylabel(
                'Rel. clock [cm]', fontname='Arial', fontsize=16)
            axs[i, 0].set_ylim(bottom=-10, top=10)
            axs[i, 0].axhline(color='darkgray', linestyle='dashed', alpha=0.5)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        axs[i, 0].text(0.05, 0.95, cLink[i], transform=axs[i, 0].transAxes, ha='left', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})

    axs[i, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    axs[i, 0].xaxis.set_major_formatter(formatterx)
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)

    # Return the link list
    return cLink


def PlotSatClk1(fISLList, cSat0, lFit, nDeg, rWin, rSlip, nCol, OutFilePrefix, OutFileSuffix):
    '''
    Similar to PlotLinkClk, but plot satellite by satellite instead of link by link
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    if lFit:
        fOut = open(OutFilePrefix+os.path.splitext(OutFileSuffix)[0], 'w')
    nFile = len(fISLList)

    # Find all links and satellites
    cLink = []
    cSat = []
    for i in range(nFile):
        with open(fISLList[i], mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine) < 30:
                    continue
                cWords = cLine.split()
                # Exclude C76
                if cWords[3][0:3] == 'C76' or cWords[3][4:7] == 'C76':
                    continue
                if cSat0[0] != 'ALL':
                    if cSat0[0] == 'EXL':
                        # Exclude links related to the specified satellites
                        if (cWords[3][0:3] in cSat0[1:]) or (cWords[3][4:7] in cSat0[1:]):
                            continue
                    else:
                        # Specified satellite list
                        if (cWords[3][0:3] not in cSat0) and (cWords[3][4:7] not in cSat0):
                            # Both satellites are not required
                            continue
                # At least one of the two satellites is required. Only add the required
                # satellites
                if cWords[3][0:3] not in cSat:
                    if cSat0[0] == 'ALL' or cWords[3][0:3] in cSat0:
                        cSat.append(cWords[3][0:3])
                if cWords[3][4:7] not in cSat:
                    if cSat0[0] == 'ALL' or cWords[3][4:7] in cSat0:
                        cSat.append(cWords[3][4:7])
                cTmp1 = cWords[3][0:3]+'-'+cWords[3][4:7]
                cTmp2 = cWords[3][4:7]+'-'+cWords[3][0:3]
                if (cTmp1 in cLink) or (cTmp2 in cLink):
                    continue
                cLink.append(cTmp1)
    cLink.sort()
    nLink = len(cLink)
    cSat.sort()
    nSat = len(cSat)

    t = []
    f = []
    for i in range(nSat):
        t.append([])
        f.append([])
    # Number of valid and invalid points for each satellite
    nPoint = np.zeros((nSat, 2), dtype=np.int32)

    for i in range(nLink):
        rEpo, Clk = GetLinkClk(fISLList, cLink[i], True)
        if rEpo.size == 0:
            continue
        nBad = 0
        nGod = rEpo.size
        if lFit:
            # Piece-Wise polynomial fitting
            ind = np.argsort(rEpo)
            lFirst = True
            j = 0
            t0 = rEpo[ind[0]]-rSlip
            while j < rEpo.size:
                # Slip the start point of the window
                if (rEpo[ind[j]]-t0) < rSlip:
                    j = j+1
                    continue
                # Inithialize the window
                t0 = rEpo[ind[j]]
                k = j
                rEpo0 = []
                Clk0 = []
                while k < rEpo.size:
                    if rEpo[ind[k]] <= (t0+rWin):
                        rEpo0.append(rEpo[ind[k]])
                        Clk0.append(Clk[ind[k]])
                        k = k+1
                    else:
                        break
                nP0 = len(rEpo0)
                if nP0 < (nDeg+1):
                    # No enough points within this window
                    MJD1 = int(t0)
                    SOD1 = (t0-MJD1)*86400
                    MJD2 = int(t0+rWin)
                    SOD2 = (t0+rWin-MJD2)*86400
                    strTmp = '#'+cLink[i]+' {: >5d} {: >8.2f} - {: >5d} {: >8.2f} {: >5d}\n'.format(MJD1,
                                                                                                    SOD1, MJD2, SOD2, nP0)
                    fOut.write(strTmp)
                else:
                    MJD1 = int(t0)
                    SOD1 = (t0-MJD1)*86400
                    MJD2 = int(t0+rWin)
                    SOD2 = (t0+rWin-MJD2)*86400
                    strTmp = ' '+cLink[i]+' {: >5d} {: >8.2f} - {: >5d} {: >8.2f} {: >5d}\n'.format(MJD1,
                                                                                                    SOD1, MJD2, SOD2, nP0)
                    fOut.write(strTmp)

                    x = np.array(rEpo0)
                    y = np.array(Clk0)
                    c = np.polynomial.polynomial.polyfit(x, y, nDeg)
                    y_fit = np.polynomial.polynomial.polyval(x, c)
                    # Accumulate the fitting residuals
                    if lFirst:
                        lFirst = False
                        # m -> cm
                        tClk = x
                        VClk = (y-y_fit)*1e2
                    else:
                        # m -> cm
                        tClk = np.append(tClk, x)
                        VClk = np.append(VClk, (y-y_fit)*1e2)
            # Do sigma clipping
            Ma = stats.sigma_clip(VClk, sigma=5, maxiters=5, masked=True)
            # Report possible blunders
            for j in range(tClk.size):
                if Ma.mask[j]:
                    # Clipped point
                    VClk[j] = np.nan
                    nBad = nBad+1
                    nGod = nGod-1
                    MJD = int(tClk[j])
                    SOD = (tClk[j]-MJD)*86400
                    strTmp = 'Blunder ' + \
                        cLink[i] + \
                        ' {: >5d} {: >8.2f} {: >12.3f}\n'.format(
                            MJD, SOD, VClk[j])
                    fOut.write(strTmp)
            rEpo = tClk
            Clk = VClk
        # append to related satellites
        if cLink[i][0:3] in cSat:
            iSat1 = cSat.index(cLink[i][0:3])
            for j in range(rEpo.size):
                t[iSat1].append(rEpo[j])
                f[iSat1].append(Clk[j])
            nPoint[iSat1][0] = nPoint[iSat1][0]+nGod
            nPoint[iSat1][1] = nPoint[iSat1][1]+nBad
        if cLink[i][4:7] in cSat:
            iSat2 = cSat.index(cLink[i][4:7])
            for j in range(rEpo.size):
                t[iSat2].append(rEpo[j])
                f[iSat2].append(Clk[j])
            nPoint[iSat2][0] = nPoint[iSat2][0]+nGod
            nPoint[iSat2][1] = nPoint[iSat2][1]+nBad
    if lFit:
        fOut.close()

    # Cal the number of row based on specified number of col
    nRow = math.ceil(nSat/nCol)

    fig, axs = plt.subplots(nRow, nCol, sharex='col', sharey='row',
                            squeeze=False, figsize=(nCol*7, nRow*2))
    # fig.subplots_adjust(hspace=0.15)
    fig.subplots_adjust(wspace=0.05)
    # Report to the terminal
    if lFit:
        print('{: <3s} {: >9s} {: >8s} {: >9s} {: >6s} {: >6s}'.format('PRN',
              'Mean', 'STD', 'RMS', 'God', 'Bad'))

    for i in range(nSat):
        # Cal the axis position, row-wise
        iRow = math.ceil((i+1)/nCol)-1
        iCol = i-iRow*nCol

        nP = len(t[i])
        if lFit:
            Mea = np.nanmean(f[i])
            Sig = np.nanstd(f[i])
            RMS = 0.0
            nGood = 0
            for j in range(nP):
                if np.isnan(f[i][j]):
                    continue
                nGood = nGood+1
                RMS = RMS + f[i][j]*f[i][j]
            if nGood > 0:
                RMS = np.sqrt(RMS/nGood)
                strTmp = 'RMS={: >6.1f}'.format(RMS)
                axs[iRow, iCol].text(0.98, 0.98, strTmp, transform=axs[iRow, iCol].transAxes,
                                     ha='right', va='top',
                                     fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
            axs[iRow, iCol].set_ylim(bottom=-20, top=20)
            axs[iRow, iCol].axhline(
                color='darkgray', linestyle='dashed', alpha=0.5)
            print('{: <3s} {: >9.2f} {: >8.2f} {: >9.2f} {: >6d} {: >6d}'.format(cSat[i],
                                                                                 Mea, Sig, RMS, nPoint[i][0], nPoint[i][1]))
        axs[iRow, iCol].plot(t[i], f[i], '.r', ms=2)
        axs[iRow, iCol].grid(which='major', axis='y',
                             c='darkgray', ls='--', lw=0.4)
        axs[iRow, iCol].set_axisbelow(True)
        if iCol == 0:
            if lFit:
                axs[iRow, 0].set_ylabel(
                    'Residuals [cm]', fontname='Arial', fontsize=16)
            else:
                axs[iRow, 0].set_ylabel(
                    'Clock [m]', fontname='Arial', fontsize=16)
            for tl in axs[iRow, 0].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
        axs[iRow, iCol].text(0.02, 0.98, cSat[i], transform=axs[iRow, iCol].transAxes, ha='left', va='top',
                             fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
        if iRow == (nRow-1):
            axs[nRow-1, iCol].set_xlabel('Modified Julian Day',
                                         fontname='Arial', fontsize=16)
            axs[nRow-1, iCol].xaxis.set_major_formatter('{x: >7.1f}')
            for tl in axs[nRow-1, iCol].get_xticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotLinkObsInterval(fISLList, lMerge, OutFilePrefix, OutFileSuffix):
    '''
    Plot the time interval between consecutive observations
    '''
    nFile = len(fISLList)

    # Find all links
    cLink = []
    for i in range(nFile):
        with open(fISLList[i], mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine) < 30:
                    continue
                cWords = cLine.split()
                cTmp1 = cWords[3][0:3]+'-'+cWords[3][4:7]
                if cTmp1 in cLink:
                    continue
                cTmp2 = cWords[3][4:7]+'-'+cWords[3][0:3]
                if lMerge and cTmp2 in cLink:
                    continue
                cLink.append(cTmp1)
    # Sort the links
    cLink.sort()
    nLink = len(cLink)

    fig, axs = plt.subplots(nLink, 1, sharex='col',
                            squeeze=False, figsize=(8, nLink*2))
    fig.subplots_adjust(hspace=0.1)

    for i in range(nLink):
        rEpo, Clk = GetLinkClk(fISLList, cLink[i], lMerge)
        # Time interval between consecutive observations
        x = np.arange(1, len(rEpo))
        y = np.diff(np.sort(rEpo), n=1)*86400
        axs[i, 0].plot(x, y, '.b')
        axs[i, 0].set_ylabel('Interval [sec]')
        axs[i, 0].text(0.95, 0.95, cLink[i]+' Mean={:>7.1f}'.format(np.mean(y)),
                       transform=axs[i, 0].transAxes, ha='right', va='top')
        rInt = []
        nInt = []
        # Calculate how many points for each sampling interval
        for j in range(len(x)):
            lFound = False
            for k in range(len(rInt)):
                if abs(y[j]-rInt[k]) < 0.5:
                    # Old interval
                    lFound = True
                    nInt[k] = nInt[k]+1
                    break
            if not lFound:
                # New interval
                rInt.append(y[j])
                nInt.append(1)
        # Account for each sampling interval
        pInt = np.divide(nInt[:], len(x))*100
        iInt = np.argsort(pInt)
        # Print out the top 5 sampling interval
        print(cLink[i], ' {:>7.1f}({:>6.2f}%) {:>7.1f}({:>6.2f}%) \
                         {:>7.1f}({:>6.2f}%) {:>7.1f}({:>6.2f}%) \
                         {:>7.1f}({:>6.2f}%)'.format(rInt[iInt[-1]], pInt[iInt[-1]],
                                                     rInt[iInt[-2]
                                                          ], pInt[iInt[-2]],
                                                     rInt[iInt[-3]
                                                          ], pInt[iInt[-3]],
                                                     rInt[iInt[-4]
                                                          ], pInt[iInt[-4]],
                                                     rInt[iInt[-5]], pInt[iInt[-5]]))

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotSatObsNum0(fISLList, cPRN0, lObsNum, OutFilePrefix, OutFileSuffix):
    '''
    Plot obs number for each satellite, distingushing intra-plane
    and inter-plane link obs.

    lObsNum --- Number or percnetage
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # BDS-3 Satellites in the same orbit. (NOTE: As PRNs are used, this table should be
    # updated from time to time)
    nPlane = 7
    cPlane = [['C27', 'C28', 'C29', 'C30', 'C34', 'C35', 'C43', 'C44'],
              ['C19', 'C20', 'C21', 'C22', 'C32', 'C33', 'C41', 'C42'],
              ['C23', 'C24', 'C36', 'C37', 'C45', 'C46', 'C25', 'C26'],
              ['C59', 'C60', 'C61'], ['C38'], ['C39'], ['C40']]

    nFile = len(fISLList)
    cSat = []
    nObs = []
    for i in range(nFile):
        with open(fISLList[i], mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine) < 30:
                    continue
                cWords = cLine.split()
                if cPRN0[0] == 'EXCL':
                    # Excluded satellites
                    if cWords[3][0:3] in cPRN0[1:] or cWords[3][4:7] in cPRN0[1:]:
                        continue
                # The first satellite
                if cWords[3][0:3] not in cSat:
                    cSat.append(cWords[3][0:3])
                    # same-orbit && diff-orbit link obs for this satellite
                    nObs.append(0)
                    nObs.append(0)
                iSat = cSat.index(cWords[3][0:3])
                iOrb = -1
                # Determine the orbit plane for the first satellite
                for j in range(nPlane):
                    if cSat[iSat] not in cPlane[j]:
                        continue
                    iOrb = j
                    break
                if iOrb == -1:
                    sys.exit('Unkown orbit plane for satellite '+cSat[iSat])
                # Judge whether the second satellite is in the same orbit
                if cWords[3][4:7] in cPlane[iOrb]:
                    # same-orbit link
                    nObs[2*iSat] = nObs[2*iSat]+1
                else:
                    # diff-orbit link
                    nObs[2*iSat+1] = nObs[2*iSat+1]+1
                # The second satellite
                if cWords[3][4:7] not in cSat:
                    cSat.append(cWords[3][4:7])
                    nObs.append(0)
                    nObs.append(0)
                iSat = cSat.index(cWords[3][4:7])
                iOrb = -1
                for j in range(nPlane):
                    if cSat[iSat] not in cPlane[j]:
                        continue
                    iOrb = j
                    break
                if iOrb == -1:
                    sys.exit('Unkown orbit plane for satellite '+cSat[iSat])
                if cWords[3][0:3] in cPlane[iOrb]:
                    # same-orbit link
                    nObs[2*iSat] = nObs[2*iSat]+1
                else:
                    # diff-orbit link
                    nObs[2*iSat+1] = nObs[2*iSat+1]+1
    nSat = len(cSat)
    cSat0 = cSat.copy()
    cSat0.sort()

    # Report to the terminal
    print('{: <4s} {: >6s} {: >6s}'.format('PRN', 'Intra', 'Inter'))
    rObs = np.zeros((nSat, 2))
    for i in range(nSat):
        j = cSat.index(cSat0[i])
        if lObsNum:
            # Plot the number of obs, in 1k
            rObs[i, 0] = nObs[2*j]*1e-3
            rObs[i, 1] = nObs[2*j+1]*1e-3
        else:
            # Plot the percentage of obs
            rObs[i, 0] = nObs[2*j]/(nObs[2*j] + nObs[2*j+1])*100
            rObs[i, 1] = 100 - rObs[i, 0]
        print('{: <4s} {:>6.2f} {:>6.2f}'.format(
            cSat0[i], rObs[i, 0], rObs[i, 1]))

    x = np.arange(nSat)
    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(nSat*0.6, 4))

    axs[0, 0].set_xlim(left=-1, right=nSat)

    # the width of the bars
    w = 1/(1+1)
    # number of same-orbit links
    axs[0, 0].bar(x+(0-1/2)*w, rObs[:, 0], w,
                  align='edge', label='Intra-Plane')
    # number of diff-orbit links
    axs[0, 0].bar(x+(0-1/2)*w, rObs[:, 1], w, align='edge',
                  label='Inter-Plane', bottom=rObs[:, 0])

    axs[0, 0].legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                     prop={'family': 'Arial', 'size': 14}, framealpha=0.3)
    axs[0, 0].grid(which='both', axis='y', color='darkgray',
                   linestyle='--', linewidth=0.4)
    axs[0, 0].set_axisbelow(True)
    if lObsNum:
        axs[0, 0].set_ylabel('Number [k]', fontname='Arial', fontsize=16)
    else:
        axs[0, 0].set_ylabel('Percentage [%]', fontname='Arial', fontsize=16)
    axs[0, 0].yaxis.get_major_locator().set_params(integer=True)
    axs[0, 0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(
        cSat0, fontdict={'fontsize': 14, 'fontname': 'Arial'})
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotSatObsNum1(fISLList, OutFilePrefix, OutFileSuffix):
    '''
    Plot number of observations at each epoch for all satellites
    and print duplicated observations, i.e. more than one observations
    are tagged as the same epoch
    '''

    nFile = len(fISLList)
    cSat = []
    for i in range(nFile):
        with open(fISLList[i], mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine) < 30:
                    continue
                cWords = cLine.split()
                if cWords[3][0:3] not in cSat:
                    cSat.append(cWords[3][0:3])
                if cWords[3][4:7] not in cSat:
                    cSat.append(cWords[3][4:7])
    cSat.sort()
    nSat = len(cSat)

    fig, axs = plt.subplots(nSat, 1, sharex='col',
                            squeeze=False, figsize=(8, nSat*2))
    fig.subplots_adjust(hspace=0.1)

    for k in range(nSat):
        # Read all observation epochs for this satellite
        rEpo = []
        nObs = []

        for i in range(nFile):
            with open(fISLList[i], mode='rt') as fOb:
                for cLine in fOb:
                    if len(cLine) < 30:
                        continue
                    cWords = cLine.split()
                    t = int(cWords[0])+float(cWords[1])/86400.0
                    RelRng = float(cWords[4])
                    RelClk = float(cWords[5])
                    if cWords[3][0:3] == cSat[k]:
                        if len(rEpo) == 0:
                            # the first epoch
                            rEpo.append(t)
                            cSat1 = []
                            cSat1.append(cWords[3][4:7])
                            nObs.append(1)
                        else:
                            if abs(t-rEpo[-1])*86400 < 0.5:
                                # Same as the last epoch (all records are sorted along time)
                                if cWords[3][4:7] not in cSat1:
                                    cSat1.append(cWords[3][4:7])
                                    nObs[-1] = nObs[-1]+1
                                else:
                                    # To same satellite at the same epoch
                                    print(cLine, end='')
                            else:
                                # New Epoch
                                rEpo.append(t)
                                cSat1 = []
                                cSat1.append(cWords[3][4:7])
                                nObs.append(1)
                    if cWords[3][4:7] == cSat[k]:
                        if len(rEpo) == 0:
                            rEpo.append(t)
                            cSat1 = []
                            cSat1.append(cWords[3][0:3])
                            nObs.append(1)
                        else:
                            if abs(t-rEpo[-1])*86400 < 0.5:
                                # Old epoch
                                if cWords[3][0:3] not in cSat1:
                                    cSat1.append(cWords[3][0:3])
                                    nObs[-1] = nObs[-1]+1
                                else:
                                    print(cLine, end='')
                            else:
                                # New Epoch
                                rEpo.append(t)
                                cSat1 = []
                                cSat1.append(cWords[3][0:3])
                                nObs.append(1)
        axs[k, 0].plot(rEpo, nObs, '.b')
        axs[k, 0].set_ylabel('# of obs')
        axs[k, 0].text(0.05, 0.95, cSat[k], transform=axs[k,
                       0].transAxes, ha='left', va='top')
    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotSatObsNum2(fISLList, cSat0, iPlot, OutFilePrefix, OutFileSuffix):
    '''
    Plot number of observations for specified satellites in each file:
    1) Except for cSat0=['ALL'], only links whose both involved satellites are
       in the list would be counted.
    2) As every link involving two satellites, therefore each link is
       accounted twice for different satellites respectively.

    iPlot --- Which info to plot
              # 0, Number of links for each satellite in each file
              # 1, Percent of intra-plane links for each satellite in each file

    Additionally, percentage of same-orbit links of each satellite is presented as well
    '''

    # BDS-3 Satellites in the same orbit. (NOTE: As PRNs are used, this table should be
    # updated from time to time)
    nPlane = 7
    cPRN = [['C27', 'C28', 'C29', 'C30', 'C34', 'C35', 'C43', 'C44'],
            ['C19', 'C20', 'C21', 'C22', 'C32', 'C33', 'C41', 'C42'],
            ['C23', 'C24', 'C36', 'C37', 'C45', 'C46', 'C25', 'C26'],
            ['C59', 'C60', 'C61'], ['C38'], ['C39'], ['C40']]

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nFile = len(fISLList)
    rEpo = []
    cSat = []
    nObs = []
    for i in range(nFile):
        # Get the date of this file from the file name
        YYYY = int(os.path.basename(fISLList[i])[-7:-3])
        DOY = int(os.path.basename(fISLList[i])[-3:])
        MO, DD = GNSSTime.doy2dom(YYYY, DOY)
        # rEpo.append(datetime.datetime(YYYY,MO,DD))
        rEpo.append(GNSSTime.doy2mjd(YYYY, DOY))
        with open(fISLList[i], mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine) < 30:
                    continue
                cWords = cLine.split()
                if cSat0[0] != 'ALL':
                    if cSat0[0] == 'EXL':
                        # Excluded satellite list
                        if (cWords[3][0:3] in cSat0) or (cWords[3][4:7] in cSat0):
                            continue
                    elif cSat0[0] == 'INC':
                        # Included satellite list
                        if (cWords[3][0:3] not in cSat0) or (cWords[3][4:7] not in cSat0):
                            continue
                if cWords[3][0:3] not in cSat:
                    cSat.append(cWords[3][0:3])
                    # same-orbit && diff-orbit links for this satellite
                    nObs.append([])
                    nObs.append([])
                    for j in range(nFile):
                        nObs[(len(cSat)-1)*2].append(0)
                        nObs[(len(cSat)-1)*2+1].append(0)
                iSat = cSat.index(cWords[3][0:3])
                # Determine the orbit plane for the first satellite
                iOrb = -1
                for j in range(nPlane):
                    if cSat[iSat] not in cPRN[j]:
                        continue
                    iOrb = j
                    break
                if iOrb == -1:
                    sys.exit('Unkown satellite '+cSat[iSat])
                # Judge whether the second satellite is in the same orbit
                if cWords[3][4:7] in cPRN[iOrb]:
                    # same-orbit link
                    nObs[2*iSat][i] = nObs[2*iSat][i]+1
                else:
                    # diff-orbit link
                    nObs[2*iSat+1][i] = nObs[2*iSat+1][i]+1

                if cWords[3][4:7] not in cSat:
                    cSat.append(cWords[3][4:7])
                    # same-orbit && diff-orbit link for this satellite
                    nObs.append([])
                    nObs.append([])
                    for j in range(nFile):
                        nObs[(len(cSat)-1)*2].append(0)
                        nObs[(len(cSat)-1)*2+1].append(0)
                iSat = cSat.index(cWords[3][4:7])
                iOrb = -1
                for j in range(nPlane):
                    if cSat[iSat] not in cPRN[j]:
                        continue
                    iOrb = j
                    break
                if iOrb == -1:
                    sys.exit('Unkown satellite '+cSat[iSat])
                if cWords[3][0:3] in cPRN[iOrb]:
                    # same-orbit link
                    nObs[2*iSat][i] = nObs[2*iSat][i]+1
                else:
                    # diff-orbit link
                    nObs[2*iSat+1][i] = nObs[2*iSat+1][i]+1
    nSat = len(cSat)
    cSat1 = cSat.copy()
    cSat1.sort()

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 5))

    if iPlot == 0:
        # Total number of observations for each satellite
        axs[0, 0].set_ylabel('Number of obs', fontname='Arial', fontsize=16)
    else:
        # Percentage of same-orbit links
        axs[0, 0].set_ylabel(
            'Percent of intra-plane links [%]', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    axs[0, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
    axs[0, 0].set_axisbelow(True)

    axs[0, 0].set_prop_cycle(color=['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r',
                                    'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g',
                                    'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'],
                             marker=['.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                     '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                     '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X'])
    y = np.zeros((nFile, nSat))
    for j in range(nSat):
        k = cSat.index(cSat1[j])
        for i in range(nFile):
            if (nObs[2*k][i]+nObs[2*k+1][i]) < 1:
                y[i, j] = np.nan
            else:
                if iPlot == 0:
                    y[i, j] = nObs[2*k][i]+nObs[2*k+1][i]
                else:
                    y[i, j] = nObs[2*k][i]/(nObs[2*k][i]+nObs[2*k+1][i])*100
        # Report the mean for each satellite
        strTmp = '{: <3s} {: >10.2f} {: >10.2f}'.format(
            cSat[k], np.nanmean(y[:, j]), np.nanmedian(y[:, j]))
        print(strTmp)
        axs[0, 0].plot(rEpo, y[:, j], ls='--', lw=1, label=cSat[k])

    axs[0, 0].legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5), framealpha=0.6,
                     labelspacing=0.1, borderpad=0.1, prop={'family': 'Arial', 'size': 14})

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotSatObsNum3(fISLList, cSat0, OutFilePrefix, OutFileSuffix):
    '''
    Plot number of observations for specified satellites in each file:
    1) Except for cSat0=['ALL'], only links whose both involved satellites are
       in the list would be counted.
    2) As every link involving two satellites, therefore each link is
       accounted twice for different satellites respectively.

    NOTE: Basically the same as PlotSatObsNum2 but using a broken x-axis
    '''
    nFile = len(fISLList)
    x = []
    cSat = []
    nObs = []
    for i in range(nFile):
        # Get the date of this file from the file name
        YYYY = int(os.path.basename(fISLList[i])[-7:-3])
        DOY = int(os.path.basename(fISLList[i])[-3:])
        MO, DD = GNSSTime.doy2dom(YYYY, DOY)
        x.append(datetime.datetime(YYYY, MO, DD))
        with open(fISLList[i], mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine) < 30:
                    continue
                cWords = cLine.split()
                # Exclude C76
                if cWords[3][0:3] == 'C76' or cWords[3][4:7] == 'C76':
                    continue
                if cSat0[0] != 'ALL' and (cWords[3][0:3] not in cSat0 or cWords[3][4:7] not in cSat0):
                    continue
                if cWords[3][0:3] not in cSat:
                    cSat.append(cWords[3][0:3])
                    nObs.append([])
                    for j in range(nFile):
                        nObs[len(cSat)-1].append(0)
                if cWords[3][4:7] not in cSat:
                    cSat.append(cWords[3][4:7])
                    nObs.append([])
                    for j in range(nFile):
                        nObs[len(cSat)-1].append(0)
                j = cSat.index(cWords[3][0:3])
                nObs[j][i] = nObs[j][i] + 1
                k = cSat.index(cWords[3][4:7])
                nObs[k][i] = nObs[k][i] + 1
    nSat = len(cSat)
    cSat1 = cSat.copy()
    cSat1.sort()
    for j in range(nSat):
        for i in range(nFile):
            if nObs[j][i] < 1:
                nObs[j][i] = np.nan

    fig, axs = plt.subplots(1, 2, sharey='row', squeeze=False, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.1)
    axs[0, 0].set_xlim(datetime.datetime(2018, 12, 31),
                       datetime.datetime(2019, 1, 8))
    axs[0, 0].spines['right'].set_visible(False)
    axs[0, 0].yaxis.tick_left()
    axs[0, 0].set_ylabel('Number of obs', fontname='Arial', fontsize=16)
    axs[0, 0].set_xlabel('Date of 2019', fontname='Arial',
                         fontsize=16, loc='right')
    axs[0, 0].tick_params(axis='x', which='major', labelsize=14, pad=15)
    axs[0, 0].tick_params(axis='x', which='minor', labelsize=12)
    axs[0, 0].tick_params(axis='y', which='major', labelsize=14)

    axs[0, 1].set_xlim(datetime.datetime(2019, 11, 30),
                       datetime.datetime(2020, 1, 1))
    axs[0, 1].spines['left'].set_visible(False)
    axs[0, 1].tick_params(axis='x', which='major', labelsize=14, pad=15)
    axs[0, 1].tick_params(axis='x', which='minor', labelsize=12)
    axs[0, 1].tick_params(axis='y', which='both', left=False)

    axs[0, 0].set_prop_cycle(color=['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r',
                                    'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g',
                                    'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'],
                             marker=['.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                     '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                     '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X'])
    axs[0, 1].set_prop_cycle(color=['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r',
                                    'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g',
                                    'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'],
                             marker=['.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                     '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                     '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X'])
    for i in range(nSat):
        # Plot for each satellite
        j = cSat.index(cSat1[i])
        axs[0, 0].plot(x, nObs[j], label=cSat[j])
        axs[0, 1].plot(x, nObs[j], label=cSat[j])

    d = 0.015
    axs[0, 0].plot((1-d, 1+d), (1-d, 1+d), '-k',
                   transform=axs[0, 0].transAxes, clip_on=False)
    axs[0, 0].plot((1-d, 1+d), (-d, +d), '-k',
                   transform=axs[0, 0].transAxes, clip_on=False)
    MO = mdates.MonthLocator()
    DD = mdates.DayLocator(interval=1)
    axs[0, 0].xaxis.set_major_locator(MO)
    axs[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axs[0, 0].xaxis.set_minor_locator(DD)
    axs[0, 0].xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
    for tl in axs[0, 0].get_xticklabels(which='both'):
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    axs[0, 1].plot((-d, +d), (1-d, 1+d), '-k',
                   transform=axs[0, 1].transAxes, clip_on=False)
    axs[0, 1].plot((-d, +d), (-d, +d), '-k',
                   transform=axs[0, 1].transAxes, clip_on=False)
    MO = mdates.MonthLocator()
    DD = mdates.DayLocator(interval=4)
    axs[0, 1].xaxis.set_major_locator(MO)
    axs[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axs[0, 1].xaxis.set_minor_locator(DD)
    axs[0, 1].xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
    axs[0, 1].legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5), framealpha=0.6,
                     labelspacing=0.1, borderpad=0.1, prop={'family': 'Arial', 'size': 14})
    for tl in axs[0, 1].get_xticklabels(which='both'):
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotSatCN(fISLList, cSat0, OutFilePrefix, OutFileSuffix):
    '''
    Plot the Carrier-to-Noise ratio series for specified satellites

    cSat0 --- Specified satellite list (only referred to receipt satellites)

    NOTE: This is only for the one-way ISL observations (X73B)
    '''
    nFile = len(fISLList)
    x = []
    cSat = []
    for i in range(nFile):
        with open(fISLList[i], mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine) < 30:
                    continue
                cWords = cLine.split()
                # # Exclude C76
                # if cWords[3][0:3] == 'C76' or cWords[3][4:7] == 'C76':
                #     continue
                if cSat0[0] != 'ALL' and cWords[3][0:3] not in cSat0:
                    continue
                if cWords[3][0:3] not in cSat:
                    cSat.append(cWords[3][0:3])
                    x.append([])
                    x.append([])
                j = cSat.index(cWords[3][0:3])
                # epoch in MJD
                x[j*2].append(float(cWords[0])+float(cWords[1])/86400)
                # CNR
                x[j*2+1].append(float(cWords[17]))
    nSat = len(cSat)
    cSat1 = cSat.copy()
    cSat1.sort()

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 5))
    axs[0, 0].set_prop_cycle(color=['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r',
                                    'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g',
                                    'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'],
                             marker=['.', 'v', '^', '<', '>', '*', 'x', 'd', 'X', 'o', 's',
                                     '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X', 'o', 's',
                                     '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X', 'o', 's'])

    for i in range(nSat):
        # Plot for each satellite
        j = cSat.index(cSat1[i])
        axs[0, 0].plot(x[j*2], x[j*2+1], ms=4, ls='', label=cSat[j])
    axs[0, 0].legend(ncol=8, loc='lower center', bbox_to_anchor=(0.5, 1.0),
                     prop={'family': 'Arial', 'size': 14})
    axs[0, 0].grid(which='both', axis='y', color='darkgray', ls='--', lw=0.8)
    axs[0, 0].set_axisbelow(True)

    axs[0, 0].xaxis.set_major_formatter('{x: >7.1f}')
    axs[0, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    axs[0, 0].yaxis.set_major_formatter('{x: >4.1f}')
    axs[0, 0].set_ylabel('C/N', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotRelCorr(fISLList, cLink0, lMerge, OutFilePrefix, OutFileSuffix):
    '''
    Plot the relativistic corrections for specified links

    cLink0 --- Specific links list
    lMerge --- Whether ignore the difference between trans && recv end

    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nFile = len(fISLList)

    cLink = []
    RelCorr = []
    for i in range(nFile):
        with open(fISLList[i], mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine) < 30:
                    continue
                cWords = cLine.split()
                # # Exclude links related to C76
                # if cWords[3][0:3] == 'C76' or cWords[3][4:7] == 'C76':
                #     continue
                cTmp1 = cWords[3][0:3]+'-'+cWords[3][4:7]
                cTmp2 = cWords[3][4:7]+'-'+cWords[3][0:3]
                # Check if this link should be excluded
                if (cLink0[0] != 'ALL-ALL') and (cTmp1 not in cLink0):
                    if lMerge:
                        if (cTmp2 not in cLink0) and \
                           (cWords[3][0:3]+'-ALL' not in cLink0) and \
                           ('ALL-'+cWords[3][0:3] not in cLink0) and \
                           ('ALL-'+cWords[3][4:7] not in cLink0) and \
                           (cWords[3][4:7]+'-ALL' not in cLink0):
                            continue
                    else:
                        if (cWords[3][0:3]+'-ALL' not in cLink0) and \
                           ('ALL-'+cWords[3][4:7] not in cLink0):
                            continue
                if cTmp1 in cLink:
                    iLink = cLink.index(cTmp1)
                elif lMerge and (cTmp2 in cLink):
                    iLink = cLink.index(cTmp2)
                else:
                    # New link
                    cLink.append(cTmp1)
                    # Epoch, in MJD
                    RelCorr.append([])
                    # Relativistic coorection 1, for clock
                    RelCorr.append([])
                    # Relativistic coorection 2, space-time bending
                    RelCorr.append([])
                    iLink = len(cLink)-1
                RelCorr[iLink*3].append(float(cWords[0]) +
                                        float(cWords[1])/86400.0)
                # m -> cm
                RelCorr[iLink*3+1].append(float(cWords[14])*1e2)
                # m -> cm
                RelCorr[iLink*3+2].append(float(cWords[15])*1e2)
    nLink = len(cLink)

    fig, axs = plt.subplots(
        2, 1, sharex='col', squeeze=False, figsize=(12, 4*2))

    for i in range(nLink):
        axs[0, 0].plot(RelCorr[i*3], RelCorr[i*3+1], '.', ms=2, label=cLink[i])
        axs[1, 0].plot(RelCorr[i*3], RelCorr[i*3+2], '.', ms=2, label=cLink[i])

    axs[0, 0].set_ylabel('dRel1 [cm]', fontname='Arial', fontsize=16)
    axs[0, 0].yaxis.set_major_formatter('{x:6.1f}')
    # axs[0, 0].set_ylim(bottom=-10, top=10)
    axs[0, 0].grid(which='both', axis='y', color='darkgray', ls='--', lw=0.8)
    axs[0, 0].set_axisbelow(True)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    axs[1, 0].set_ylabel('dRel2 [cm]', fontname='Arial', fontsize=16)
    axs[1, 0].yaxis.set_major_formatter('{x:5.1f}')
    axs[1, 0].set_ylim(bottom=0, top=10)
    axs[1, 0].grid(which='both', axis='y', color='darkgray', ls='--', lw=0.8)
    axs[1, 0].set_axisbelow(True)
    for tl in axs[1, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    axs[1, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    axs[1, 0].xaxis.set_major_formatter('{x:7.1f}')
    for tl in axs[1, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotDupObs(fDupList, OutFilePrefix, OutFileSuffix):
    '''
    Plot duplicated ISL obs record
    '''

    cLink = []
    xDif = []
    nFile = len(fDupList)
    for i in range(nFile):
        with open(fDupList[i], mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine) < 30:
                    continue
                cWords = cLine.split()
                if cWords[3][0:3]+'-'+cWords[3][4:7] not in cLink and \
                   cWords[3][4:7]+'-'+cWords[3][0:3] not in cLink:
                    # New link
                    cLink.append(cWords[3][0:3]+'-'+cWords[3][4:7])
                    xDif.append([])
                    xDif.append([])
                    xDif.append([])
                    iLink = len(cLink)-1
                elif cWords[3][0:3]+'-'+cWords[3][4:7] in cLink:
                    iLink = cLink.index(cWords[3][0:3]+'-'+cWords[3][4:7])
                else:
                    iLink = cLink.index(cWords[3][4:7]+'-'+cWords[3][0:3])
                # Epoch
                xDif[iLink*3].append(int(cWords[0])+float(cWords[1])/86400.0)
                #range in cm
                xDif[iLink*3+1].append(float(cWords[6])*1e-9*299792458*1e2)
                #clock in cm
                xDif[iLink*3+2].append(float(cWords[7])*1e-9*299792458*1e2)
    cLink0 = cLink.copy()
    cLink0.sort()
    nLink = len(cLink)

    fig, axs = plt.subplots(3, 2, squeeze=False, figsize=(12, 10))

    axs[0, 0].set_ylim(-100, 100)
    axs[0, 1].set_ylim(-100, 100)
    axs[0, 0].set_ylabel('Range [cm]')
    axs[0, 0].set_xlim(58483, 58492)
    axs[0, 1].set_xlim(58817, 58850)

    axs[1, 0].set_ylim(-100, 100)
    axs[1, 1].set_ylim(-100, 100)
    axs[1, 0].set_ylabel('Clock [cm]')
    axs[1, 0].set_xlim(58483, 58492)
    axs[1, 1].set_xlim(58817, 58850)

    # Number of dif for each link
    nDif = np.zeros(nLink, dtype=np.int32)
    nDifAll = 0
    xDifAll = [[], []]
    nExclude = [0, 0]
    for i in range(nLink):
        j = cLink.index(cLink0[i])

        nDif[i] = len(xDif[j*3])
        nDifAll = nDifAll + nDif[i]
        for k in range(nDif[i]):
            xDifAll[0].append(xDif[j*3+1][k])
            xDifAll[1].append(xDif[j*3+2][k])
            if abs(xDif[j*3+1][k]) > 100:
                nExclude[0] = nExclude[0]+1
            if abs(xDif[j*3+2][k]) > 100:
                nExclude[1] = nExclude[1]+1
        # Get the mean, std && rms
        Mea1 = np.mean(xDif[j*3+1])
        Sig1 = np.std(xDif[j*3+1])
        Mea2 = np.mean(xDif[j*3+2])
        Sig2 = np.std(xDif[j*3+2])

        # range
        axs[0, 0].plot(xDif[j*3], xDif[j*3+1], '.', label=cLink[j])
        axs[0, 1].plot(xDif[j*3], xDif[j*3+1], '.', label=cLink[j])
        # clock
        axs[1, 0].plot(xDif[j*3], xDif[j*3+2], '.', label=cLink[j])
        axs[1, 1].plot(xDif[j*3], xDif[j*3+2], '.', label=cLink[j])

    # Number of duplicated obs for each link
    axs[2, 0].plot(nDif, 'o')
    axs[2, 0].set_ylabel('Number of dup.')
    axs[2, 0].set_xlabel('Link')

    axs[2, 0].text(0.95, 0.95, '#={:>7d}'.format(int(nDifAll)),
                   transform=axs[2, 0].transAxes, ha='right', va='top')

    clabel = ['range ({:>6d} excl.)'.format(nExclude[0]),
              'clock ({:>6d} excl.)'.format(nExclude[1])]
    axs[2, 1].hist(xDifAll, label=clabel, bins=20, range=(-100, 100))
    axs[2, 1].legend()
    axs[2, 1].set_xlabel('[cm]')

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotDerivErr(fISLList, cLink0, OutFilePrefix, OutFileSuffix):
    '''
    Plot the derivation error series based on the Error Propagation Law
    for  derivation of dual one-way ISL links

    fISLList --- List of ISL files (simulated ones, which contains the vectors of
                 line-of-sight)
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # Time Difference between derviation epoch and actual epoch, in sec
    dt1 = 10
    dt2 = 10
    # Sigma for clock drift, 1E-13 s/s
    SigC = 1e-12*299792458
    # Co-variance matrix for velocity
    V1 = np.zeros((3, 3))
    V2 = np.zeros((3, 3))
    # 0.1 mm/s (i.e. Var=1e-8)
    V1[0, 0] = 1e-8
    V1[1, 1] = 1e-8
    V1[2, 2] = 1e-8
    V2[0, 0] = 1e-8
    V2[1, 1] = 1e-8
    V2[2, 2] = 1e-8

    cLink = []
    Err = []
    for i in range(len(fISLList)):
        with open(fISLList[i], mode='rt') as fOb:
            for cLine in fOb:
                cWords = cLine.split()
                if cLink0[0] != 'ALL-ALL' and cWords[3] not in cLink0:
                    continue
                # Epoch
                rMJD = int(cWords[0]) + float(cWords[1])/86400.0
                if cWords[3] not in cLink:
                    cLink.append(cWords[3])
                    Err.append([])
                    Err.append([])
                j = cLink.index(cWords[3])
                Err[j*2].append(rMJD)
                # Vector of line-of-sight
                E = np.zeros((1, 3))
                E[0, 0] = float(cWords[9])
                E[0, 1] = float(cWords[10])
                E[0, 2] = float(cWords[11])
                OrbErr = dt2*np.matmul(np.matmul(E, V1), np.transpose(E))*dt2 + \
                    dt1*np.matmul(np.matmul(E, V1), np.transpose(E))*dt1
                rErr = OrbErr[0, 0] + dt2*SigC*SigC*dt2 + dt1*SigC*SigC*dt1
                # Sigma in mm
                Err[j*2+1].append(np.sqrt(rErr)*1e3)
    nLink = len(cLink)
    cLink1 = cLink.copy()
    cLink1.sort()

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 4))

    for i in range(nLink):
        j = cLink.index(cLink1[i])
        axs[0, 0].plot(Err[j*2], Err[j*2+1], '.', ms=2, label=cLink[j])
    axs[0, 0].grid(which='both', axis='y', c='darkgray', ls='--', lw=0.4)
    axs[0, 0].set_axisbelow(True)

    axs[0, 0].set_ylabel('Derivation errors [mm]',
                         fontname='Arial', fontsize=16)
    # axs[0,0].set_ylim(bottom=0,top=0.1)
    axs[0, 0].ticklabel_format(
        axis='y', style='sci', useOffset=False, useMathText=True)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    axs[0, 0].xaxis.set_major_formatter('{x:7.1f}')
    axs[0, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotAtmISLNum(fISLList, OutFilePrefix, OutFileSuffix):
    '''
    Plot the daily percentage of ISL links affected by Trop && Iono
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nFile = len(fISLList)
    rMJD = np.zeros(nFile, dtype=np.int32)
    nObs = np.zeros((nFile, 3), dtype=np.int32)
    # Percentage
    Per = np.zeros((nFile, 2))
    for i in range(nFile):
        # Epoch according to the file name
        YYYY = int(os.path.basename(fISLList[i])[-7:-3])
        DOY = int(os.path.basename(fISLList[i])[-3:])
        rMJD[i] = GNSSTime.doy2mjd(YYYY, DOY)
        with open(fISLList[i], mode='rt') as fOb:
            for cLine in fOb:
                # Total number of obs
                nObs[i, 0] = nObs[i, 0] + 1
                cWords = cLine.split()
                if cWords[6] == 'T':
                    # Trop
                    nObs[i, 1] = nObs[i, 1] + 1
                if cWords[7] == 'T':
                    # Iono
                    nObs[i, 2] = nObs[i, 2] + 1
        # Percentage of Iono
        Per[i, 0] = nObs[i, 2]/nObs[i, 0]*100
        # Percentage of Trop
        Per[i, 1] = nObs[i, 1]/nObs[i, 0]*100
    # Print the mean value
    print('{: >8s} {: >8s}'.format('MeaIono', 'MeaTrop'))
    strTmp = '{: >8.3f} {: >8.3f}'.format(
        np.nanmean(Per[:, 0]), np.nanmean(Per[:, 1]))
    print(strTmp)

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 4))

    axs[0, 0].set_xlim(left=rMJD[0]-1, right=rMJD[nFile-1]+1)
    axs[0, 0].set_xticks(rMJD)
    axs[0, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    # the width of the bars
    w = 1/(2+1)
    axs[0, 0].bar(rMJD+(0-2/2)*w, Per[:, 0], w, align='edge', label='Ion')
    axs[0, 0].bar(rMJD+(1-2/2)*w, Per[:, 1], w, align='edge', label='Tro')

    axs[0, 0].legend(ncol=2, loc='upper center', prop={
                     'family': 'Arial', 'size': 14}, framealpha=0.3)
    axs[0, 0].grid(b=True, which='both', axis='y', color='darkgray', linestyle='--',
                   linewidth=0.8)
    axs[0, 0].set_axisbelow(True)
    axs[0, 0].set_ylabel('Percentage [%]', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotIonDelay(OutFilePrefix, OutFileSuffix):
    '''
    Plot the signal delay caused by the first order Ionosphere effect for ISL
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # Assuming a high value of Slant Total Electron Content, in 1e16 m-2
    S = 300
    # Radio frequencies in GHz for Ka-Band i.e 26.5 ~ 40 GHz
    f = np.linspace(26.5, 40, num=1000)
    # Delay in cm
    d = np.zeros(f.size)
    for i in range(f.size):
        d[i] = 40.308*S/(f[i]*f[i])

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(6, 4))
    formatterx = mpl.ticker.StrMethodFormatter('{x:4.1f}')

    axs[0, 0].plot(f, d, '.r', ms=2)
    axs[0, 0].grid(which='both', axis='y', c='darkgray', ls='--', lw=0.8)
    axs[0, 0].set_axisbelow(True)

    axs[0, 0].set_xlabel('Frequencies [GHz]', fontname='Arial', fontsize=16)
    axs[0, 0].xaxis.set_major_formatter(formatterx)
    for tl in axs[0, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    axs[0, 0].set_ylabel('Ionospheric Delay [cm]',
                         fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_yticklabels():
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

    # InFilePrefix = r'E:/ISL/'
    InFilePrefix=r'D:/Code/PROJECT/ISL_Sim/'
    # InFilePrefix = os.path.join(cDskPre0, r'DATA/ISL/')

    fISLList = glob.glob(InFilePrefix+'ISLE_2019001')

    # OutFilePrefix = os.path.join(cDskPre0, r'DATA/ISL/')
    OutFilePrefix=r'D:/Code/PROJECT/ISL_Sim/'

    # CalNEQ(fISLList,False,True,0,False)
    # CalNEQ(fISLList,False,True,0,True)

    # OutFileSuffix='SatLinkNum0_2019335'
    # PlotSatLinkNum0(fISLList,['EXCL','C76'],OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='SatLinkNum0_2019001_2019365'
    # PlotSatLinkNum1(fISLList,['EXCL','C76'],OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='SatObsNum0_2019335'
    # PlotSatObsNum0(fISLList,['EXCL','C76'],False,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='LinkTab_2019001_2019365'
    # PlotLinkTab0(fISLList,False,OutFilePrefix,OutFileSuffix)

    # for i in range(len(fISLList)):
    #     fList=[fISLList[i]]
    #     OutFileSuffix='LinkClk'+fISLList[i][-8:]+'.png'
    #     PlotLinkClk(fList,['ALL-ALL'],True,False,1,0.0417,0.0417,OutFilePrefix,OutFileSuffix)

    # OutFileSuffix='LinkClk_2019351.pdf'
    # PlotLinkClk(fISLList,['ALL-ALL'],True,False,1,0.0417,0.0417,OutFilePrefix,OutFileSuffix)

    # # MEO plane-A
    # cSat=['C27','C28','C29','C30','C34','C35']
    # # MEO plane-B
    # cSat=['C19','C20','C21','C22','C32','C33']
    # MEO plane-C
    cSat = ['C23', 'C24', 'C25', 'C26', 'C36', 'C37']
    # # IGSO
    # cSat=['C38','C39']
    # OutFileSuffix = 'SatClk_2019342_Fit'
    # PlotSatClk1(fISLList, ['ALL'], True, 1, 0.0417,
    #             0.0417, 2, OutFilePrefix, OutFileSuffix)

    # OutFileSuffix='LinkObsInterval.png'
    # PlotLinkObsInterval(fISLList,True,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='SatObsNum1.png'
    # PlotSatObsNum1(fISLList,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='SatObsNum_2019335_2019365_Num'
    # PlotSatObsNum2(fISLList,['EXL','C40','C45','C46','C59','C76'],0,OutFilePrefix,OutFileSuffix)

    fDupList = ['D:/Code/PROJECT/WORK_ISL/DupRec']
    OutFileSuffix = 'DupRec.png'
    # PlotDupObs(fDupList,OutFilePrefix,OutFileSuffix)

    OutFileSuffix='DerivErr2_1'
    # PlotDerivErr(fISLList,['ALL-ALL'],OutFilePrefix,OutFileSuffix)

    # OutFileSuffix='AtmObsNum'
    # PlotAtmISLNum(fISLList,OutFilePrefix,OutFileSuffix)

    # OutFileSuffix='IonDelay'
    # PlotIonDelay(OutFilePrefix,OutFileSuffix)

    # OutFileSuffix = 'ISLO_2022054'
    # PlotSatCN(fISLList, ['ALL'], OutFilePrefix, OutFileSuffix)

    OutFileSuffix = 'RelCorr_2019001'
    PlotRelCorr(fISLList, ['ALL-ALL'], False, OutFilePrefix, OutFileSuffix)