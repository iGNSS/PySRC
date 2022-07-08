#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
'''
__author__ = 'hanbing'

# Standard library imports
import os
import sys
import os.path
import glob

# Related third party imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# Local application/library specific imports
from PySRC.miscellaneous import PCV
from PySRC.miscellaneous import CorSys
from PySRC.miscellaneous import GNSSTime
from PySRC.miscellaneous import ReadCtrl


def PlotOrbPar0(cSer, fPathSer, MJD0, nDay, OutFilePrefix, OutFileSuffix):
    '''
    Plot the comparison of observations number for orbit parameters
    between different solutions

    fPathSer --- path list for different solutions
        MJD0 --- start mjd
        nDay --- number of days
    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nSer = len(cSer)
    cSat = []
    nObs = []
    for iDay in range(nDay):
        # Initialisation for this day
        nObs.append([])
        for i in range(len(cSat)):
            for j in range(nSer):
                nObs[iDay].append(0)
        MJD = MJD0+iDay
        YYYY, DOY = GNSSTime.mjd2doy(MJD)
        for iSer in range(nSer):
            fPar = os.path.join(
                fPathSer[iSer], 'par_{:4d}{:03d}'.format(YYYY, DOY))
            if not os.path.isfile(fPar):
                print(fPar+' does not exist!')
                continue
            with open(fPar, mode='rt') as fOb:
                for cLine in fOb:
                    cWords = cLine.split()
                    if cWords[1] != 'PXSAT':
                        continue
                    if cWords[5] not in cSat:
                        cSat.append(cWords[5])
                        # Append the new satellite for each day until now
                        for i in range(iDay+1):
                            for j in range(nSer):
                                nObs[i].append(0)
                    iSat = cSat.index(cWords[5])
                    nObs[iDay][iSat*nSer+iSer] = int(cWords[14])
    nSat = len(cSat)
    cSat1 = cSat.copy()
    cSat1.sort()

    x = np.arange(nSat)
    fig, axs = plt.subplots(nDay, 1, sharex='col',
                            squeeze=False, figsize=(nSat*0.5, nDay*3))
    # the width of the bars
    w = 1/(nSer+1)

    for iDay in range(nDay):
        MJD = MJD0+iDay
        YYYY, DOY = GNSSTime.mjd2doy(MJD)
        axs[iDay, 0].set_xlim(left=-1, right=nSat)
        axs[iDay, 0].text(0.02, 0.98, '{:4d} {:03d}'.format(YYYY, DOY),
                          transform=axs[iDay, 0].transAxes, ha='left', va='top',
                          fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
        for iSer in range(nSer):
            y = np.zeros(nSat)
            for iSat in range(nSat):
                y[iSat] = nObs[iDay][cSat.index(cSat1[iSat])*nSer+iSer]
            axs[iDay, 0].bar(x+(iSer-nSer/2)*w, y, w,
                             align='edge', label=cSer[iSer])
        axs[iDay, 0].legend(ncol=nSer, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                            prop={'family': 'Arial', 'size': 14})
        axs[iDay, 0].set_ylabel('# of obs', fontname='Arial', fontsize=16)
        for tl in axs[iDay, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
    axs[iDay, 0].set_xlabel('Satellite PRNs', fontname='Arial', fontsize=16)
    axs[iDay, 0].set_xticks(x)
    axs[iDay, 0].set_xticklabels(
        cSat1, fontdict={'fontsize': 14, 'fontname': 'Arial'})
    for tl in axs[iDay, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotOrbPar1(fParList, cSat0, OutFilePrefix, OutFileSuffix):
    '''
    Plot the
    '''


def PlotClkPar1(fPar, ClkList, lAPriori, OutFilePrefix, OutFileSuffix):
    '''
    Plot clock parameters, including cal && est clock rate, their difference
    and clock offset estimation

    lAPriori --- Whether add the a-priori values, otherwise only the estimations
    '''

    ClkName = []
    Clk = []
    with open(fPar, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cWords[1][0:6] != 'SATCLK' and cWords[1][0:6] != 'RECCLK':
                continue
            if int(cWords[2]) != 0:
                # Receiver clock
                if (ClkList[0] != 'ALL') and (cWords[3] not in ClkList):
                    continue
                elif cWords[3] not in ClkName:
                    # New clock
                    ClkName.append(cWords[3])
                    # Epoch and clock offset, clock rate
                    Clk.append([])
                    Clk.append([])
                    Clk.append([])
                iClk = ClkName.index(cWords[3])
            else:
                # Satellite clock
                if (ClkList[0] != 'ALL') and (cWords[5] not in ClkList):
                    continue
                elif cWords[5] not in ClkName:
                    # New clock
                    ClkName.append(cWords[5])
                    # Epoch and clock offset, clock rate
                    Clk.append([])
                    Clk.append([])
                    Clk.append([])
                iClk = ClkName.index(cWords[5])
            # Epoch in MJD
            rMJD = float(cWords[12])
            iEpo = -1
            # Search for the epoch index
            for i in range(len(Clk[iClk*3])):
                if np.abs(Clk[iClk*3][i]-rMJD)*86400.0 < 1.0:
                    iEpo = i
                    break
            if iEpo == -1:
                # New epoch
                Clk[iClk*3].append(rMJD)
                Clk[iClk*3+1].append(np.nan)
                Clk[iClk*3+2].append(np.nan)
                iEpo = len(Clk[iClk*3])-1
            # Polynomial degree for the clock model
            iDeg = int(cWords[1][6:7])
            # Estimation, meter -> sec
            if lAPriori:
                # Add the a-priori values
                Clk[iClk*3+iDeg +
                    1][iEpo] = (float(cWords[6])+float(cWords[10]))/299792458.0
            else:
                # Only estimates
                Clk[iClk*3+iDeg+1][iEpo] = float(cWords[10])/299792458.0
    nClk = len(ClkName)
    ClkName1 = ClkName.copy()
    ClkName1.sort()

    fig, axs = plt.subplots(nClk, 3, sharex='col',
                            squeeze=False, figsize=(18, nClk*1.5))
    formatterx = mpl.ticker.StrMethodFormatter('{x:7.1f}')

    for i in range(nClk):
        j = ClkName.index(ClkName1[i])
        # Data for this clock and sort it along epoch
        Clk0 = np.array(Clk[j*3:j*3+3])
        ind = np.argsort(Clk0[0])
        nEpo = Clk0[0].size
        t = np.zeros(nEpo-1)
        # Clock rate calculated from the offset and its difference from
        # the directly estimated rate
        VClk = np.zeros(nEpo-1)
        dVClk = np.zeros(nEpo-1)
        for k in range(nEpo-1):
            t[k] = Clk0[0][ind[k]]
            if np.isnan(Clk0[1][ind[k]]):
                VClk[k] = np.nan
                dVClk[k] = np.nan
            elif np.isnan(Clk0[1][ind[k+1]]):
                VClk[k] = np.nan
                dVClk[k] = np.nan
            else:
                # Clock rate calculated from the estimated offset
                # in nanosec/sec
                VClk[k] = (Clk0[1][ind[k+1]]-Clk0[1][ind[k]]) / \
                    (Clk0[0][ind[k+1]]-Clk0[0][ind[k]])/86400.0*1e9
                # If the directly estimated clock rate available
                if np.isnan(Clk0[2][ind[k]]):
                    dVClk[k] = np.nan
                else:
                    dVClk[k] = Clk0[2][ind[k]]*1e9 - VClk[k]
        # calculated rate, in ps/s
        axs[i, 0].plot(t, VClk*1e3, '.b', ms=2)
        strTmp = 'Cal {:>7.1f}+/-{:>7.2f}'.format(
            np.nanmean(VClk*1e3), np.nanstd(VClk*1e3))
        axs[i, 0].text(0.98, 0.98, strTmp, transform=axs[i, 0].transAxes, ha='right', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkblue'})
        # Clock rate directly estimated, sec/s -> ps/s
        axs[i, 0].plot(Clk0[0], Clk0[2]*1e12, '.g', ms=2)
        strTmp = 'Est {:>7.1f}+/-{:>7.2f}'.format(
            np.nanmean(Clk0[2]*1e12), np.nanstd(Clk0[2]*1e12))
        axs[i, 0].text(0.98, 0.02, strTmp, transform=axs[i, 0].transAxes, ha='right', va='bottom',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkgreen'})
        # the difference between cal and est rate, in ps/s
        axs[i, 1].plot(t, dVClk*1e3, '.r', ms=2)
        strTmp = '{:>7.2f}+/-{:>7.3f}'.format(
            np.nanmean(dVClk*1e3), np.nanstd(dVClk*1e3))
        axs[i, 1].text(0.98, 0.98, strTmp, transform=axs[i, 1].transAxes, ha='right', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkred'})

        # Clock offset estimation
        axs[i, 2].plot(Clk0[0], Clk0[1]*1e9, '.g', ms=2)
        strTmp = 'Est {:>7.3f}+/-{:>7.4f}'.format(
            np.nanmean(Clk0[1]*1e9), np.nanstd(Clk0[1]*1e9))
        axs[i, 2].text(0.98, 0.02, strTmp, transform=axs[i, 2].transAxes, ha='right', va='bottom',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkgreen'})

        axs[i, 0].set_ylabel('[ps/s]', fontname='Arial', fontsize=16)
        axs[i, 1].set_ylabel(r'$\Delta$ [ps/s]', fontname='Arial', fontsize=16)
        axs[i, 2].set_ylabel('[ns]', fontname='Arial', fontsize=16)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        for tl in axs[i, 1].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        for tl in axs[i, 2].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        axs[i, 0].text(0.02, 0.98, ClkName[j], transform=axs[i, 0].transAxes, ha='left', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})

    axs[i, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    axs[i, 1].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    axs[i, 2].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    axs[i, 0].xaxis.set_major_formatter(formatterx)
    axs[i, 1].xaxis.set_major_formatter(formatterx)
    axs[i, 2].xaxis.set_major_formatter(formatterx)
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    for tl in axs[i, 1].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    for tl in axs[i, 2].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotClkPar2(fPar, ClkList, OutFilePrefix, OutFileSuffix):
    '''
    Plot post-sigma of clock parameters and its observation numbers
    '''

    ClkName = []
    Clk = []
    with open(fPar, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cWords[1][0:6] != 'SATCLK' and cWords[1][0:6] != 'RECCLK':
                continue
            if int(cWords[2]) != 0:
                # Receiver clock
                if (ClkList[0] != 'ALL') and (ClkList[0] != 'RCV') and (cWords[3] not in ClkList):
                    continue
                elif cWords[3] not in ClkName:
                    # New clock
                    ClkName.append(cWords[3])
                    # Epoch, post-sigma, obs number for clock offset est
                    Clk.append([])
                    Clk.append([])
                    Clk.append([])
                    # Epoch, post-sigma, obs number for clock rate est
                    Clk.append([])
                    Clk.append([])
                    Clk.append([])
                iClk = ClkName.index(cWords[3])
            else:
                # Satellite clock
                if (ClkList[0] != 'ALL') and (ClkList[0] != 'SAT') and (cWords[5] not in ClkList):
                    continue
                elif cWords[5] not in ClkName:
                    # New clock
                    ClkName.append(cWords[5])
                    # Epoch, post-sigma, obs number for clock offset est
                    Clk.append([])
                    Clk.append([])
                    Clk.append([])
                    # Epoch, post-sigma, obs number for clock rate est
                    Clk.append([])
                    Clk.append([])
                    Clk.append([])
                iClk = ClkName.index(cWords[5])
             # Polynomial degree for the clock model
            iDeg = int(cWords[1][6:7])
            # Epoch in MJD
            rMJD = float(cWords[12])
            Clk[iClk*6+iDeg*3].append(rMJD)
            # post-sigma, meter -> ns
            Clk[iClk*6+iDeg*3+1].append(float(cWords[11])/299792458.0*1e9)
            # number of obs
            Clk[iClk*6+iDeg*3+2].append(int(cWords[14]))
    nClk = len(ClkName)
    ClkName1 = ClkName.copy()
    ClkName1.sort()

    fig, axs = plt.subplots(nClk, 2, sharex='col',
                            squeeze=False, figsize=(28, nClk*1.5))

    for i in range(nClk):
        j = ClkName.index(ClkName1[i])

        # post-sigma for clock offset
        axs[i, 0].plot(Clk[j*6], Clk[j*6+1], '.r', ms=2)
        # obs number for clock offset
        ax0 = axs[i, 0].twinx()
        ax0.plot(Clk[j*6], Clk[j*6+2], '-g', ms=2)
        ax0.set_ylabel('Obs num', fontname='Arial', fontsize=16)
        for tl in ax0.get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        # post-sigma for clock rate
        axs[i, 1].plot(Clk[j*6+3], Clk[j*6+4], '.r', ms=2)
        # obs number for clock rate
        ax1 = axs[i, 1].twinx()
        ax1.plot(Clk[j*6+3], Clk[j*6+5], '-g', ms=2)
        ax1.set_ylabel('Obs num', fontname='Arial', fontsize=16)
        for tl in ax1.get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        axs[i, 0].set_ylabel('offset [ns]', fontname='Arial', fontsize=16)
        axs[i, 1].set_ylabel('rate [ns/s]', fontname='Arial', fontsize=16)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        for tl in axs[i, 1].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        axs[i, 0].text(0.02, 0.98, ClkName[j], transform=axs[i, 0].transAxes, ha='left', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})

    axs[i, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    axs[i, 1].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    axs[i, 0].xaxis.set_major_formatter('{x:7.1f}')
    axs[i, 1].xaxis.set_major_formatter('{x:7.1f}')
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    for tl in axs[i, 1].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotClkPar3(fParList, dIntv, nMinObs, ClkList, OutFilePrefix, OutFileSuffix):
    '''
    Plot observation numbers for clock offset parameters to indicate the
    observability. Note that only clock offset parameters are checked.

      dIntv --- interval for clock estimation, in seconds
    nMinObs --- Min number of obs to count a clock offset parameter as
                observed
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # Clock name list
    cClk = []
    # Min && Max epoch time for each clock
    rSes = []
    rClk = []

    for i in range(len(fParList)):
        with open(fParList[i], mode='rt') as fOb:
            for cLine in fOb:
                cWords = cLine.split()
                if cWords[1][0:7] != 'SATCLK0' and cWords[1][0:7] != 'RECCLK0':
                    continue
                if int(cWords[2]) != 0:
                    # Receiver clock
                    if (ClkList[0] != 'ALL') and (ClkList[0] != 'RCV') and (cWords[3] not in ClkList):
                        continue
                    elif cWords[3] not in cClk:
                        # New clock
                        cClk.append(cWords[3])
                        # Min epoch time
                        rSes.append(99999)
                        # Max epoch time
                        rSes.append(-99999)
                        # Epoch
                        rClk.append([])
                        # Observation number
                        rClk.append([])
                    iClk = cClk.index(cWords[3])
                else:
                    # Satellite clock
                    if (ClkList[0] != 'ALL') and (ClkList[0] != 'SAT') and (cWords[5] not in ClkList):
                        continue
                    elif cWords[5] not in cClk:
                        # New clock
                        cClk.append(cWords[5])
                        rSes.append(99999)
                        rSes.append(-99999)
                        rClk.append([])
                        rClk.append([])
                    iClk = cClk.index(cWords[5])
                # Epoch in MJD
                rMJD = float(cWords[12])
                if rMJD < rSes[iClk*2]:
                    rSes[iClk*2] = rMJD
                if rMJD > rSes[iClk*2+1]:
                    rSes[iClk*2+1] = rMJD
                rClk[iClk*2].append(rMJD)
                # number of obs
                rClk[iClk*2+1].append(int(cWords[14]))
    nClk = len(cClk)
    ClkName = cClk.copy()
    ClkName.sort()

    # Report percentage of tracked arc to the terminal
    strTmp = '{: >4s} {: >7s} {: >7s} {: >6s}'.format(
        'Clk', 'nEpoT', 'nEpoA', 'Ratio')
    print(strTmp)

    fig, axs = plt.subplots(nClk, 1, sharex='col',
                            squeeze=False, figsize=(8, nClk*1.5))
    for i in range(nClk):
        j = cClk.index(ClkName[i])

        # Cal the theoretical number of observed epochs for each clock
        nEpoT = int((rSes[j*2+1]-rSes[j*2])*86400/dIntv) + 1
        # Count the actual observed epochs
        nEpoA = 0
        for k in range(len(rClk[j*2])):
            if rClk[j*2+1][k] < nMinObs:
                continue
            nEpoA = nEpoA+1
        Ratio = nEpoA/nEpoT
        strTmp = '{: >4s} {: >7d} {: >7d} {: >6.4f}'.format(
            cClk[j], nEpoT, nEpoA, Ratio)
        print(strTmp)

        # number of obs for clock offset
        axs[i, 0].plot(rClk[j*2], rClk[j*2+1], '.r', ms=2)
        strTmp = 'tracked arc = {: >5.1f}%'.format(Ratio*100)
        axs[i, 0].text(0.98, 0.98, strTmp, transform=axs[i, 0].transAxes, ha='right', va='top',
                       family='Arial', size=16, weight='bold', color='darkgreen')

        axs[i, 0].set_ylabel('Obs Num', fontname='Arial', fontsize=16)

        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        axs[i, 0].text(0.02, 0.98, cClk[j], transform=axs[i, 0].transAxes, ha='left', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})

    axs[i, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    axs[i, 0].xaxis.set_major_formatter('{x:7.1f}')
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotEOPPar1(fPar, OutFilePrefix, OutFileSuffix):
    '''
    Plot EOP parameters, including cal && est rate, their difference
    and offset estimation

    NOTE: Only up to 1-degree PWP model supported
    '''

    cEOP = []
    XEOP = []
    with open(fPar, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cWords[1][0:len(cWords[1])-1] not in ['XPOLE', 'YPOLE', 'UT1', 'DPSI', 'DEPSI']:
                continue
            if cWords[1][0:len(cWords[1])-1] not in cEOP:
                # New EOP parameter
                cEOP.append(cWords[1][0:len(cWords[1])-1])
                # Epoch, offset and rate
                XEOP.append([])
                XEOP.append([])
                XEOP.append([])
            iEOP = cEOP.index(cWords[1][0:len(cWords[1])-1])
            # Epoch in MJD
            rMJD = float(cWords[12])
            iEpo = -1
            # Search for the epoch index
            for i in range(len(XEOP[iEOP*3])):
                if np.abs(XEOP[iEOP*3][i]-rMJD)*86400.0 < 1.0:
                    iEpo = i
                    break
            if iEpo == -1:
                # New epoch
                XEOP[iEOP*3].append(rMJD)
                XEOP[iEOP*3+1].append(np.nan)
                XEOP[iEOP*3+2].append(np.nan)
                iEpo = len(XEOP[iEOP*3])-1
            # Polynomial degree for the par model
            iDeg = int(cWords[1][-1:])
            # Only estimates, in us/uas
            XEOP[iEOP*3+iDeg+1][iEpo] = float(cWords[10])*1e6
    nEOP = len(cEOP)

    fig, axs = plt.subplots(nEOP, 3, sharex='col',
                            squeeze=False, figsize=(27, nEOP*1.5))
    fig.subplots_adjust(wspace=0.15)
    formatterx = mpl.ticker.StrMethodFormatter('{x:7.1f}')

    for i in range(nEOP):
        j = i
        # Data for this EOP par and sort it along epoch
        X0 = np.array(XEOP[j*3:j*3+3])
        ind = np.argsort(X0[0])
        nEpo = X0[0].size
        t = np.zeros(nEpo-1)
        # Rate calculated from the offset and its difference from
        # the directly estimated rate
        VX0 = np.zeros(nEpo-1)
        dVX0 = np.zeros(nEpo-1)
        for k in range(nEpo-1):
            t[k] = X0[0][ind[k]]
            if np.isnan(X0[1][ind[k]]):
                VX0[k] = np.nan
                dVX0[k] = np.nan
            elif np.isnan(X0[1][ind[k+1]]):
                VX0[k] = np.nan
                dVX0[k] = np.nan
            else:
                # Rate calculated from the estimated offset
                VX0[k] = (X0[1][ind[k+1]]-X0[1][ind[k]]) / \
                    (X0[0][ind[k+1]]-X0[0][ind[k]])
                # If the directly estimated rate available
                if np.isnan(X0[2][ind[k]]):
                    dVX0[k] = np.nan
                else:
                    dVX0[k] = X0[2][ind[k]] - VX0[k]
        # calculated rate
        axs[i, 0].plot(t, VX0, '^b', ms=4)
        strTmp = 'Cal {:>7.1f}+/-{:>7.2f}'.format(
            np.nanmean(VX0), np.nanstd(VX0))
        axs[i, 0].text(0.98, 0.98, strTmp, transform=axs[i, 0].transAxes, ha='right', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkblue'})
        # Rate directly estimated
        axs[i, 0].plot(X0[0], X0[2], '.g', ms=4)
        strTmp = 'Est {:>7.1f}+/-{:>7.2f}'.format(
            np.nanmean(X0[2]), np.nanstd(X0[2]))
        axs[i, 0].text(0.98, 0.02, strTmp, transform=axs[i, 0].transAxes, ha='right', va='bottom',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkgreen'})
        # the difference between cal and est rate
        axs[i, 1].plot(t, dVX0, '.r', ms=4)
        strTmp = '{:>7.2f}+/-{:>7.3f}'.format(
            np.nanmean(dVX0), np.nanstd(dVX0))
        axs[i, 1].text(0.98, 0.98, strTmp, transform=axs[i, 1].transAxes, ha='right', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkred'})

        # offset estimation
        axs[i, 2].plot(X0[0], X0[1], '.g', ms=4)
        strTmp = 'Est {:>7.3f}+/-{:>7.4f}'.format(
            np.nanmean(X0[1]), np.nanstd(X0[1]))
        axs[i, 2].text(0.98, 0.02, strTmp, transform=axs[i, 2].transAxes, ha='right', va='bottom',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkgreen'})

        axs[i, 0].set_ylabel('[us/s]', fontname='Arial', fontsize=16)
        axs[i, 1].set_ylabel(r'$\Delta$ [us/s]', fontname='Arial', fontsize=16)
        axs[i, 2].set_ylabel('[us]', fontname='Arial', fontsize=16)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        for tl in axs[i, 1].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        for tl in axs[i, 2].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        axs[i, 0].text(0.02, 0.98, cEOP[j], transform=axs[i, 0].transAxes, ha='left', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})

    axs[i, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    axs[i, 1].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    axs[i, 2].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    axs[i, 0].xaxis.set_major_formatter(formatterx)
    axs[i, 1].xaxis.set_major_formatter(formatterx)
    axs[i, 2].xaxis.set_major_formatter(formatterx)
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    for tl in axs[i, 1].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    for tl in axs[i, 2].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotEOPPar2(fParList, OutFilePrefix, OutFileSuffix):
    '''
    Plot the post-sigma series of EOP parameters

    NOTE: Only up to 1-degree PWP model supported
    '''

    cEOP = []
    XEOP = []
    for i in range(len(fParList)):
        with open(fParList[i], mode='rt') as fOb:
            for cLine in fOb:
                cWords = cLine.split()
                if cWords[1] not in ['XPOLE0', 'YPOLE0', 'UT10', 'DPSI0', 'DEPSI0',
                                     'XPOLE1', 'YPOLE1', 'UT11', 'DPSI1', 'DEPSI1']:
                    continue
                if cWords[1] not in cEOP:
                    # New EOP parameter
                    cEOP.append(cWords[1])
                    # Epoch, post-sigma and obs number
                    XEOP.append([])
                    XEOP.append([])
                    XEOP.append([])
                iEOP = cEOP.index(cWords[1])
                # Epoch in MJD
                rMJD = float(cWords[12])
                XEOP[iEOP*3].append(rMJD)
                # post-sigma, in 1e-6
                XEOP[iEOP*3+1].append(float(cWords[11])*1e6)
                # number of obs
                XEOP[iEOP*3+2].append(float(cWords[14]))

    fig, axs = plt.subplots(
        5, 2, sharex='col', squeeze=False, figsize=(14, 5*3))
    # fig.subplots_adjust(wspace=0.15)
    # formatterx=mpl.ticker.StrMethodFormatter('{x:7.1f}')

    cEOP0 = [['XPOLE0', 'XPOLE1'], ['YPOLE0', 'YPOLE1'], ['UT10', 'UT11'],
             ['DPSI0', 'DPSI1'], ['DEPSI0', 'DEPSI1']]
    cYLab = [[r'xpole [$\mu$as]', r'xpole rate [$\mu$as/d]'],
             [r'ypole [$\mu$as]', r'ypole rate [$\mu$as/d]'],
             [r'UT1 [$\mu$s]', r'LOD [$\mu$s/d]'],
             [r'dX [$\mu$as]', r'dX rate [$\mu$as/d]'],
             [r'dY [$\mu$as]', r'dY rate [$\mu$as/d]']]

    for i in range(5):
        for j in range(2):
            if cEOP0[i][j] not in cEOP:
                continue
            k = cEOP.index(cEOP0[i][j])
            axs[i, j].plot(XEOP[k*3], XEOP[k*3+1], 'v--r', ms=3, lw=1)
            # ax0=axs[i,j].twinx()
            # ax0.plot(XEOP[k*3],XEOP[k*3+2],'-g',ms=2)
            # ax0.set_ylabel('Obs num',fontname='Arial',fontsize=16)
            # ax0.set_yticklabels([])
            # for tl in ax0.get_yticklabels():
            #     tl.set_fontname('Arial'); tl.set_fontsize(14)
            axs[i, j].grid(which='both', axis='y', color='darkgray', linestyle='--',
                           linewidth=0.8)
            axs[i, j].set_ylabel(cYLab[i][j], fontname='Arial', fontsize=16)
            for tl in axs[i, j].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
    for j in range(2):
        axs[i, j].set_xlabel('Modified Julian Day',
                             fontname='Arial', fontsize=16)
        # axs[i,j].xaxis.set_major_formatter(formatterx)
        for tl in axs[i, j].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotEOPPar3(fParList, OutFilePrefix, OutFileSuffix):
    '''
    Plot the estimates && post-sigma series of EOP dX/dY parameters

    NOTE: Only up to 1-degree PWP model supported
    '''

    cEOP = []
    XEOP = []
    for i in range(len(fParList)):
        with open(fParList[i], mode='rt') as fOb:
            for cLine in fOb:
                cWords = cLine.split()
                if cWords[1] not in ['DPSI0', 'DEPSI0', 'DPSI1', 'DEPSI1']:
                    continue
                if cWords[1] not in cEOP:
                    # New EOP parameter
                    cEOP.append(cWords[1])
                    # Epoch, estimates and post-sigma
                    XEOP.append([])
                    XEOP.append([])
                    XEOP.append([])
                iEOP = cEOP.index(cWords[1])
                # Epoch in MJD
                rMJD = float(cWords[12])
                XEOP[iEOP*3].append(rMJD)
                # estimates, in 1e-6
                XEOP[iEOP*3+1].append(float(cWords[10])*1e6)
                # post-sigma, in 1e-6
                XEOP[iEOP*3+2].append(float(cWords[11])*1e6)

    fig, axs = plt.subplots(
        2, 2, sharex='col', squeeze=False, figsize=(14, 2*3))
    # fig.subplots_adjust(wspace=0.15)
    # formatterx=mpl.ticker.StrMethodFormatter('{x:7.1f}')

    cEOP0 = [['DPSI0', 'DPSI1'], ['DEPSI0', 'DEPSI1']]
    cYLab = [[r'dX [$\mu$as]', r'dX rate [$\mu$as/d]'],
             [r'dY [$\mu$as]', r'dY rate [$\mu$as/d]']]

    for i in range(2):
        for j in range(2):
            if cEOP0[i][j] not in cEOP:
                continue
            k = cEOP.index(cEOP0[i][j])
            axs[i, j].errorbar(XEOP[k*3], XEOP[k*3+1], yerr=XEOP[k*3+2], fmt='v--r',
                               ms=3, lw=1, capsize=3)
            axs[i, j].grid(which='both', axis='y', color='darkgray', linestyle='--',
                           linewidth=0.8)
            axs[i, j].set_ylabel(cYLab[i][j], fontname='Arial', fontsize=16)
            for tl in axs[i, j].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
    for j in range(2):
        axs[i, j].set_xlabel('Modified Julian Day',
                             fontname='Arial', fontsize=16)
        # axs[i,j].xaxis.set_major_formatter(formatterx)
        for tl in axs[i, j].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotEOPPar40(cSer, fSerList, lOnlyERP, iPlot, OutFilePrefix, OutFileSuffix):
    '''
    Plot EOP estimates or formal error series for several solutions

    lOnlyERP --- Whether plot only ERP
       iPlot ---
                 # 0, estimates
                 # 1, formal error
                 # 2, both
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    if not lOnlyERP:
        nEOP = 7
        EOP = ['XPOLE0', 'XPOLE1', 'YPOLE0',
               'YPOLE1', 'UT11', 'DPSI1', 'DEPSI1']
        yLab = [r'$x_p$ [$\mu$as]', r'$\.x_p$ [$\mu$as/d]',
                r'$y_p$ [$\mu$as]', r'$\.y_p$ [$\mu$as/d]',
                r'LOD [$\mu$s]', r'$\.{dX}$ [$\mu$as]', r'$\.{dY}$ [$\mu$as]']
    else:
        nEOP = 5
        EOP = ['XPOLE0', 'XPOLE1', 'YPOLE0', 'YPOLE1', 'UT11']
        yLab = [r'$x_p$ [$\mu$as]', r'$\.x_p$ [$\mu$as/d]',
                r'$y_p$ [$\mu$as]', r'$\.y_p$ [$\mu$as/d]',
                r'LOD [$\mu$s]']
    cft = ['o--', 'o--', 'o--', 'o--', 'o--', 'o--', 'o--']

    fig, axs = plt.subplots(nEOP, 1, sharex='col',
                            squeeze=False, figsize=(8, nEOP*2))
    # fig.subplots_adjust(hspace=0.1)

    nSer = len(cSer)
    for k in range(nSer):
        t = []
        x = []
        s = []
        n = []
        for i in range(nEOP):
            #For each EOP, epoch, est, sig, obs#
            t.append([])
            x.append([])
            s.append([])
            n.append([])
        nFile = len(fSerList[k])
        for i in range(nFile):
            with open(fSerList[k][i], mode='rt') as fOb:
                nLine = 0
                for cLine in fOb:
                    nLine = nLine+1
                    if nLine == 1:
                        continue
                    cWords = cLine.split()
                    if cWords[1] not in EOP:
                        continue
                    iEOP = EOP.index(cWords[1])
                    t[iEOP].append(float(cWords[12]))
                    # as -> uas or s -> us
                    x[iEOP].append(float(cWords[10])*1e6)
                    s[iEOP].append(float(cWords[11])*1e6)
                    n[iEOP].append(int(cWords[14]))
        for i in range(nEOP):
            if iPlot == 0:
                # Only estimates
                axs[i, 0].plot(t[i], x[i], cft[i], ms=4, lw=1, label=cSer[k])
                axs[i, 0].axhline(color='darkgray',
                                  linestyle='dashed', alpha=0.5, lw=0.8)
            elif iPlot == 1:
                # Only formal error
                axs[i, 0].plot(t[i], s[i], cft[i], ms=4, lw=1, label=cSer[k])
            else:
                # Estimates and formal error
                axs[i, 0].errorbar(
                    t[i], x[i], yerr=s[i], fmt=cft[i], ms=4, lw=1, capsize=3, label=cSer[k])
                axs[i, 0].axhline(color='darkgray',
                                  linestyle='dashed', alpha=0.5, lw=0.8)
    axs[0, 0].legend(ncol=nSer, loc='lower center', framealpha=0.6, bbox_to_anchor=(0.5, 1.0),
                     prop={'family': 'Arial', 'size': 14})
    for i in range(nEOP):
        axs[i, 0].grid(which='major', axis='y', c='darkgray',
                       ls='--', lw=0.4, alpha=0.5)
        axs[i, 0].set_axisbelow(True)
        axs[i, 0].set_ylabel(yLab[i], fontname='Arial', fontsize=16)
        axs[i, 0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
    axs[i, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotEOPPar41(cSer, fSerList, lOnlyERP, iPlot, OutFilePrefix, OutFileSuffix):
    '''
    Plot EOP average formal error or STD for several solutions

    lOnlyERP --- Whether plot only ERP or full EOP parameters

    iPlot    ---
                 # 0, STD of estimates
                 # 1, mean formal error

    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    if not lOnlyERP:
        nEOP = 7
        EOP = ['XPOLE0', 'XPOLE1', 'YPOLE0',
               'YPOLE1', 'UT11', 'DPSI1', 'DEPSI1']
        xLab = [r'$x_p$', r'$\.x_p$', r'$y_p$',
                r'$\.y_p$', r'LOD', r'$\.{dX}$', r'$\.{dY}$']
    else:
        nEOP = 5
        EOP = ['XPOLE0', 'XPOLE1', 'YPOLE0', 'YPOLE1', 'UT11']
        xLab = [r'$x_p$', r'$\.x_p$', r'$y_p$', r'$\.y_p$', r'LOD']

    nSer = len(cSer)
    RMS = np.zeros((nEOP, nSer))
    for k in range(nSer):
        #For each EOP, epoch, est, sig, obs#
        t = []
        x = []
        s = []
        n = []
        for i in range(nEOP):
            t.append([])
            x.append([])
            s.append([])
            n.append([])
        nFile = len(fSerList[k])
        for i in range(nFile):
            with open(fSerList[k][i], mode='rt') as fOb:
                nLine = 0
                for cLine in fOb:
                    nLine = nLine+1
                    if nLine == 1:
                        continue
                    cWords = cLine.split()
                    if cWords[1] not in EOP:
                        continue
                    iEOP = EOP.index(cWords[1])
                    t[iEOP].append(float(cWords[12]))
                    if iEOP == 4:
                        # For UT11 (i.e., LOD), s -> us -> uas
                        x[iEOP].append(float(cWords[10])*1e6*15)
                        s[iEOP].append(float(cWords[11])*1e6*15)
                    else:
                        # as -> uas
                        x[iEOP].append(float(cWords[10])*1e6)
                        s[iEOP].append(float(cWords[11])*1e6)
                    n[iEOP].append(int(cWords[14]))

        for i in range(nEOP):
            if iPlot == 0:
                # STD of estimates
                RMS[i, k] = np.std(x[i])
            else:
                # Mean of formal error
                RMS[i, k] = np.mean(s[i])
    # Report to the terminal
    strTmp = '{: <15s}'.format('EOP')
    for j in range(nSer):
        strTmp = strTmp+' {: >10s}'.format(cSer[j])
    print(strTmp)
    for i in range(nEOP):
        strTmp = '{: <15s}'.format(EOP[i])
        for j in range(nSer):
            strTmp = strTmp+' {: >10.2f}'.format(RMS[i, j])
        print(strTmp)

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(nEOP*nSer*0.5, 5))
    x = np.arange(nEOP)

    axs[0, 0].set_xlim(left=-1, right=nEOP)

    # the width of the bars
    w = 1/(nSer+1)
    for i in range(nSer):
        axs[0, 0].bar(x+(i-nSer/2)*w, RMS[:, i], w,
                      align='edge', label=cSer[i])

    axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
    axs[0, 0].set_axisbelow(True)
    if nSer > 1:
        axs[0, 0].legend(ncol=nSer, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                         framealpha=0.6, prop={'family': 'Arial', 'size': 14})

    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(
        xLab, fontdict={'fontsize': 14, 'fontname': 'Arial'})
    if lOnlyERP:
        axs[0, 0].set_xlabel('Earth Rotation Parameters',
                             fontname='Arial', fontsize=16)
    else:
        axs[0, 0].set_xlabel('Earth Orientation Parameters',
                             fontname='Arial', fontsize=16)
    if iPlot == 0:
        strTmp = r'STD [$\mu$as(/d)]'
    else:
        strTmp = r'Average formal errors [$\mu$as(/d)]'
    axs[0, 0].set_ylabel(strTmp, fontname='Arial', fontsize=16)
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


def PlotGCCPar1(fParList, OutFilePrefix, OutFileSuffix):
    '''
    Plot Geo Center parameter estimates && post-sigma series
    '''

    cGCC = []
    XGCC = []
    for i in range(len(fParList)):
        with open(fParList[i], mode='rt') as fOb:
            for cLine in fOb:
                cWords = cLine.split()
                if cWords[1] not in ['GEOCX0', 'GEOCY0', 'GEOCZ0',
                                     'GEOCX1', 'GEOCY1', 'GEOCZ1']:
                    continue
                if cWords[1] not in cGCC:
                    # New GCC parameter
                    cGCC.append(cWords[1])
                    # Epoch, estimates and post-sigma
                    XGCC.append([])
                    XGCC.append([])
                    XGCC.append([])
                iGCC = cGCC.index(cWords[1])
                # Epoch in MJD
                rMJD = float(cWords[12])
                XGCC[iGCC*3].append(rMJD)
                # estimates, in mm
                XGCC[iGCC*3+1].append(float(cWords[10])*1e3)
                # post-sigma, in mm
                XGCC[iGCC*3+2].append(float(cWords[11])*1e3)

    fig, axs = plt.subplots(
        3, 1, sharex='col', squeeze=False, figsize=(12, 3*3))
    # fig.subplots_adjust(wspace=0.15)
    # formatterx=mpl.ticker.StrMethodFormatter('{x:7.1f}')

    cGCC0 = ['GEOCX0', 'GEOCY0', 'GEOCZ0']
    cYLab = [r'GCC X [mm]', r'GCC Y [mm]', r'GCC Z [mm]']

    for i in range(3):
        if cGCC0[i] not in cGCC:
            continue
        k = cGCC.index(cGCC0[i])
        axs[i, 0].errorbar(XGCC[k*3], XGCC[k*3+1],
                           yerr=XGCC[k*3+2], fmt='v--r', ms=2, capsize=3)
        axs[i, 0].set_ylabel(cYLab[i], fontname='Arial', fontsize=16)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
    axs[2, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    # axs[2,0].xaxis.set_major_formatter(formatterx)
    for tl in axs[2, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotGCCPar20(cSer, fSerList, iPlot, OutFilePrefix, OutFileSuffix):
    '''
    Plot Geo Center estimates or formal error series for several solutions

    iPlot ---
             # 0, estimates
             # 1, formal error
             # 2, both

    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    GCC = ['GEOCX0', 'GEOCY0', 'GEOCZ0']
    yLab = ['XGCC [mm]', 'YGCC [mm]', 'ZGCC [mm]']

    fig, axs = plt.subplots(
        3, 1, sharex='col', squeeze=False, figsize=(8, 3*2))
    # fig.subplots_adjust(hspace=0.1)

    cft = ['o--', 'o--', 'o--']
    nSer = len(cSer)
    for k in range(nSer):
        # Epoch for X/Y/Z
        t = [[], [], []]
        # Estimates
        x = [[], [], []]
        # Sigma
        s = [[], [], []]
        # Number of observations
        n = [[], [], []]
        nFile = len(fSerList[k])
        for i in range(nFile):
            with open(fSerList[k][i], mode='rt') as fOb:
                nLine = 0
                lFound = [False, False, False]
                for cLine in fOb:
                    nLine = nLine+1
                    if nLine == 1:
                        continue
                    cWords = cLine.split()
                    if cWords[1][0:6] not in GCC:
                        continue
                    iGCC = GCC.index(cWords[1][0:6])
                    t[iGCC].append(float(cWords[12]))
                    x[iGCC].append(float(cWords[10])*1e3)
                    s[iGCC].append(float(cWords[11])*1e3)
                    n[iGCC].append(int(cWords[14]))
                    lFound[iGCC] = True
                    if lFound[0] and lFound[1] and lFound[2]:
                        break
        for i in range(3):
            if iPlot == 0:
                # Only estimates
                axs[i, 0].plot(t[i], x[i], cft[i], ms=4, lw=1, label=cSer[k])
                axs[i, 0].axhline(color='darkgray',
                                  linestyle='dashed', alpha=0.5, lw=0.8)
            elif iPlot == 1:
                # Only formal error
                axs[i, 0].plot(t[i], s[i], cft[i], ms=4, lw=1, label=cSer[k])
            else:
                # Estimates and formal error
                axs[i, 0].errorbar(
                    t[i], x[i], yerr=s[i], fmt=cft[i], ms=4, lw=1, capsize=3, label=cSer[k])
                axs[i, 0].axhline(color='darkgray',
                                  linestyle='dashed', alpha=0.5, lw=0.8)
    axs[0, 0].legend(ncol=nSer, loc='lower center', framealpha=0.6, bbox_to_anchor=(0.5, 1.0),
                     prop={'family': 'Arial', 'size': 14})
    for i in range(3):
        axs[i, 0].set_ylabel(yLab[i], fontname='Arial', fontsize=16)
        axs[i, 0].yaxis.set_major_formatter('{x: >5.1f}')
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        axs[i, 0].grid(which='major', axis='both',
                       c='darkgray', ls='--', lw=0.4, alpha=0.5)
        axs[i, 0].set_axisbelow(True)
    axs[i, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotGCCPar21(cSer, fSerList, iPlot, OutFilePrefix, OutFileSuffix):
    '''
    Plot Geo Center average formal error or STD for several solutions

    iPlot ---
             # 0, STD of estimates
             # 1, mean formal error

    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    GCC = ['GEOCX0', 'GEOCY0', 'GEOCZ0']
    xLab = ['XGCC', 'YGCC', 'ZGCC']

    nSer = len(cSer)
    RMS = np.zeros((3, nSer))
    for k in range(nSer):
        #For GCC X/Y/Z, epoch, est, sig, obs#
        t = [[], [], []]
        x = [[], [], []]
        s = [[], [], []]
        n = [[], [], []]
        nFile = len(fSerList[k])
        for i in range(nFile):
            with open(fSerList[k][i], mode='rt') as fOb:
                nLine = 0
                lFound = [False, False, False]
                for cLine in fOb:
                    nLine = nLine+1
                    if nLine == 1:
                        continue
                    cWords = cLine.split()
                    if cWords[1][0:6] not in GCC:
                        continue
                    iGCC = GCC.index(cWords[1][0:6])
                    t[iGCC].append(float(cWords[12]))
                    # m -> mm
                    x[iGCC].append(float(cWords[10])*1e3)
                    # m -> mm
                    s[iGCC].append(float(cWords[11])*1e3)
                    n[iGCC].append(int(cWords[14]))
                    lFound[iGCC] = True
                    if lFound[0] and lFound[1] and lFound[2]:
                        break
        for i in range(3):
            if iPlot == 0:
                # STD of estimates
                RMS[i, k] = np.std(x[i])
            else:
                # Mean of formal error
                RMS[i, k] = np.mean(s[i])
    # Report to the terminal
    strTmp = '{: <15s}'.format('GCC')
    for j in range(nSer):
        strTmp = strTmp+' {: >10s}'.format(cSer[j])
    print(strTmp)
    for i in range(3):
        strTmp = '{: <15s}'.format(GCC[i])
        for j in range(nSer):
            strTmp = strTmp+' {: >10.2f}'.format(RMS[i, j])
        print(strTmp)

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(3*nSer*1.5, 5))
    x = np.arange(3)

    axs[0, 0].set_xlim(left=-1, right=3)

    # the width of the bars
    w = 1/(nSer+1)
    for i in range(nSer):
        axs[0, 0].bar(x+(i-nSer/2)*w, RMS[:, i], w,
                      align='edge', label=cSer[i])

    axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
    axs[0, 0].set_axisbelow(True)
    if nSer > 1:
        axs[0, 0].legend(ncol=nSer, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                         framealpha=0.6, prop={'family': 'Arial', 'size': 14})

    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(
        xLab, fontdict={'fontsize': 14, 'fontname': 'Arial'})
    axs[0, 0].set_xlabel('Geocenter Coordinates',
                         fontname='Arial', fontsize=16)
    if iPlot == 0:
        axs[0, 0].set_ylabel('STD [mm]', fontname='Arial', fontsize=16)
    else:
        axs[0, 0].set_ylabel('Average formal errors [mm]',
                             fontname='Arial', fontsize=16)
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


def PlotGNSBias00(fParList, cBias, OutFilePrefix, OutFileSuffix):
    '''
    Plot the sum of specified (daily) GNSS recevier-specific time bias parameters
    for each file.

    This is mainly for validation of Zero-Mean-Constraint.

    cBias --- Specified GNSS recevier-specific time bias parameter
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # Epoch, sum, number of stations
    rSum = [[], [], []]
    for i in range(len(fParList)):
        # Get the epoch from the file name
        YYYY = int(os.path.basename(fParList[i])[-7:-3])
        DOY = int(os.path.basename(fParList[i])[-3:])
        rSum[0].append(GNSSTime.doy2mjd(YYYY, DOY))
        nSta = 0
        xSum = 0
        with open(fParList[i], mode='rt') as fOb:
            for cLine in fOb:
                cWords = cLine.split()
                if cWords[1] != cBias:
                    continue
                # m -> mm
                nSta = nSta+1
                xSum = xSum + float(cWords[10])*1e3
        rSum[1].append(xSum)
        rSum[2].append(nSta)

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(8, 4))

    axs[0, 0].plot(rSum[0], rSum[1], 'o--r', ms=4, lw=1)
    axs[0, 0].ticklabel_format(axis='y', useOffset=False, useMathText=True)
    axs[0, 0].set_ylabel(
        cBias+' sum [mm]', fontname='Arial', fontsize=16, color='r')
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
        tl.set_color('r')
    axs[0, 0].axhline(color='darkgray', linestyle='dashed', alpha=0.5)

    axe = axs[0, 0].twinx()
    axe.plot(rSum[0], rSum[2], '^--g', ms=4, lw=1)
    axe.set_ylabel('Sta #', fontname='Arial', fontsize=16, color='g')
    for tl in axe.get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
        tl.set_color('g')

    axs[0, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    axs[0, 0].ticklabel_format(axis='x', useOffset=False, useMathText=True)
    for tl in axs[0, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotGNSBias01(fParList, cBias, OutFilePrefix, OutFileSuffix):
    '''
    Report the stations that have no really valid obs for the receiver-specific bias
    parameters in each file.

    As there should be observations from both sat groups if you want to estimate
    the bias between them, the number of obs for bias parameters should be always
    smaller than the total number of obs for this station. We just check this to see
    if a bias parameter is really estimated.
    '''

    for i in range(len(fParList)):
        # Check for each file
        cSta = []
        nObs = [[], []]
        with open(fParList[i], mode='rt') as fOb:
            for cLine in fOb:
                cWords = cLine.split()
                if cWords[1] != cBias and cWords[1] != 'PXSTA':
                    continue
                if cWords[3] not in cSta:
                    cSta.append(cWords[3])
                    # Obs number of BIAS par
                    nObs[0].append(np.nan)
                    # Obs number of XSTA par
                    nObs[1].append(np.nan)
                iSta = cSta.index(cWords[3])
                if cWords[1] == cBias:
                    if np.isnan(nObs[0][iSta]):
                        nObs[0][iSta] = int(cWords[14])
                    else:
                        # Duplicated bias parameters
                        sys.exit('Duplicated BIAS parameters, ' +
                                 cWords[3]+' '+fParList[i])
                else:
                    if np.isnan(nObs[1][iSta]):
                        nObs[1][iSta] = int(cWords[14])
                    else:
                        # Duplicated bias parameters
                        sys.exit('Duplicated XSTA parameters, ' +
                                 cWords[3]+' '+fParList[i])
        # Do the check
        for j in range(len(cSta)):
            if np.isnan(nObs[0][j]) or np.isnan(nObs[1][j]):
                sys.exit('Not found BIAS/XSTA parameters, ' +
                         cSta[j]+' '+fParList[i])
            elif nObs[0][j] >= nObs[1][j]:
                print('No obs for BIAS parameters, '+cSta[j]+' '+fParList[i])


def PlotGNSBias11(fPar, cBias, yLab, fCtr, iSort, lReport, OutFilePrefix, OutFileSuffix):
    '''
    Plot the estimates and their post-sigma for specified (daily) GNSS
    receiver-specific time bias parametrs in the given par-file

    cBias --- Specified GNSS recevier-specific time bias parameter
    yLab  --- y-axis label for the figure
    fCtr  --- ctrl-file which provides the rec && ant info for stations
    iSort --- Whether sort the parameters along rec/ant type
              # 0, Only sort along station names
              # 1, Sort along Rec type first
              # 2, Sort along Ant type first
              # 3, Sort along Rec type, then along Ant type
    lReport --- Whether report the sorted list

    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    cSta = []
    xPar = [[], []]
    with open(fPar, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cWords[1] != cBias:
                continue
            if cWords[3] not in cSta:
                cSta.append(cWords[3])
                # bias estimates, meter -> ns
                xPar[0].append(float(cWords[10])/299792458*1e9)
                # Sigma of bias estimates, meter -> ns
                xPar[1].append(float(cWords[11])/299792458*1e9)
            else:
                sys.exit('Duplicated stations')
    nSta = len(cSta)
    # Get ant && rec info
    cRec = ReadCtrl.GetStaRec0(fCtr, cSta)

    # Sort the station list
    cSta0 = cSta.copy()
    cRec0 = cRec.copy()
    for i in range(0, nSta-1):
        for j in range(i+1, nSta):
            if iSort == 1:
                # Sort along Rec type first
                if cRec0[0][j] > cRec0[0][i]:
                    continue
                elif cRec0[0][j] < cRec0[0][i]:
                    cTmp = cSta0[i]
                    cSta0[i] = cSta0[j]
                    cSta0[j] = cTmp
                    for k in range(4):
                        cTmp = cRec0[k][i]
                        cRec0[k][i] = cRec0[k][j]
                        cRec0[k][j] = cTmp
                elif cSta0[j] < cSta0[i]:
                    cTmp = cSta0[i]
                    cSta0[i] = cSta0[j]
                    cSta0[j] = cTmp
                    for k in range(4):
                        cTmp = cRec0[k][i]
                        cRec0[k][i] = cRec0[k][j]
                        cRec0[k][j] = cTmp
            elif iSort == 2:
                # Sort along Ant type first
                if cRec0[3][j] > cRec0[3][i]:
                    continue
                elif cRec0[3][j] < cRec0[3][i]:
                    cTmp = cSta0[i]
                    cSta0[i] = cSta0[j]
                    cSta0[j] = cTmp
                    for k in range(4):
                        cTmp = cRec0[k][i]
                        cRec0[k][i] = cRec0[k][j]
                        cRec0[k][j] = cTmp
                elif cSta0[j] < cSta0[i]:
                    cTmp = cSta0[i]
                    cSta0[i] = cSta0[j]
                    cSta0[j] = cTmp
                    for k in range(4):
                        cTmp = cRec0[k][i]
                        cRec0[k][i] = cRec0[k][j]
                        cRec0[k][j] = cTmp
            elif iSort == 3:
                # Sort along Rec type first, then sort along Ant type
                if cRec0[0][j] > cRec0[0][i]:
                    continue
                elif cRec0[0][j] < cRec0[0][i]:
                    cTmp = cSta0[i]
                    cSta0[i] = cSta0[j]
                    cSta0[j] = cTmp
                    for k in range(4):
                        cTmp = cRec0[k][i]
                        cRec0[k][i] = cRec0[k][j]
                        cRec0[k][j] = cTmp
                elif cRec0[3][j] > cRec0[3][i]:
                    continue
                elif cRec0[3][j] < cRec0[3][i]:
                    cTmp = cSta0[i]
                    cSta0[i] = cSta0[j]
                    cSta0[j] = cTmp
                    for k in range(4):
                        cTmp = cRec0[k][i]
                        cRec0[k][i] = cRec0[k][j]
                        cRec0[k][j] = cTmp
                elif cSta0[j] < cSta0[i]:
                    cTmp = cSta0[i]
                    cSta0[i] = cSta0[j]
                    cSta0[j] = cTmp
                    for k in range(4):
                        cTmp = cRec0[k][i]
                        cRec0[k][i] = cRec0[k][j]
                        cRec0[k][j] = cTmp
            else:
                # Sort only along station name
                if cSta0[j] < cSta0[i]:
                    cTmp = cSta0[i]
                    cSta0[i] = cSta0[j]
                    cSta0[j] = cTmp
                    for k in range(4):
                        cTmp = cRec0[k][i]
                        cRec0[k][i] = cRec0[k][j]
                        cRec0[k][j] = cTmp
    xPar0 = np.zeros((nSta, 2))
    for i in range(nSta):
        j = cSta.index(cSta0[i])
        xPar0[i, 0] = xPar[0][j]
        xPar0[i, 1] = xPar[1][j]
        if lReport:
            cStr = '{:>4s} {:>20s} {:>20s}'.format(
                cSta0[i], cRec0[0][i], cRec0[3][i])
            print(cStr)

    # print('{} {:3d} {} {:15.4f} {} {:15.4f}'.format('nSta=',nSta,
    # 'Mean_est=',np.mean(xPar[0]),'Mean_sig',np.mean(xPar[1])))

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(14, 4))
    x = np.arange(nSta)
    axs[0, 0].set_xlim(left=-1, right=nSta)

    # axs[0,0].errorbar(x,xPar0[:,0],yerr=xPar0[:,1],fmt='.',capsize=5,ms=6)
    w = 1/(1+1)
    axs[0, 0].bar(x+(0-1/2)*w, xPar0[:, 0], w, align='edge', yerr=xPar0[:, 1],
                  error_kw=dict(ecolor='r', capsize=2, elinewidth=1, capthick=0.6))
    axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)

    axs[0, 0].axhline(color='darkgray', linestyle='dashed', alpha=0.5, lw=0.8)
    axs[0, 0].set_ylabel(yLab+' [ns]', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    axs[0, 0].set_xlabel('Station Codes', fontname='Arial', fontsize=16)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(cSta0, rotation=90, c='darkblue',
                              fontdict={'fontsize': 5, 'fontname': 'monospace', 'horizontalalignment': 'center'})

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotGNSBias12(fPar, cBias, yLab, fCtr, iSort, lReport, OutFilePrefix, OutFileSuffix):
    '''
    Plot the estimates diff between 2 specified (daily) GNSS
    receiver-specific time bias parametrs in the given par-file

    cBias --- the 2 specified GNSS recevier-specific time bias parameters
    yLab  --- y-axis label for the figure
    fCtr  --- ctrl-file which provides the rec && ant info for stations
    iSort --- Whether sort the parameters along rec/ant type
              # 0, Only sort along station names
              # 1, Sort along Rec type first
              # 2, Sort along Ant type first
              # 3, Sort along Rec type, then along Ant type
    lReport --- Whether report the sorted list

    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    cSta = []
    xPar = [[], [], [], [], [], [], [], []]
    with open(fPar, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cWords[1] != cBias[0] and cWords[1] != cBias[1]:
                continue
            if cWords[3] not in cSta:
                cSta.append(cWords[3])
                # for the 1st bias: est, sig, obs num
                xPar[0].append(0)
                xPar[1].append(-1)
                xPar[2].append(0)
                # for the 2nd bias: est, sig, obs num
                xPar[3].append(0)
                xPar[4].append(-1)
                xPar[5].append(0)
                # for the diff btw the 1st and 2nd bias: diff, sig
                xPar[6].append(np.nan)
                xPar[7].append(np.nan)
            iSta = cSta.index(cWords[3])
            if cWords[1] == cBias[0]:
                # bias estimates, meter -> ns
                xPar[0][iSta] = float(cWords[10])/299792458*1e9
                # Sigma of bias estimates, meter -> ns
                xPar[1][iSta] = float(cWords[11])/299792458*1e9
                # Number of obs
                xPar[2][iSta] = int(cWords[14])
            else:
                # bias estimates, meter -> ns
                xPar[3][iSta] = float(cWords[10])/299792458*1e9
                # Sigma of bias estimates, meter -> ns
                xPar[4][iSta] = float(cWords[11])/299792458*1e9
                # Number of obs
                xPar[5][iSta] = int(cWords[14])
    nSta = len(cSta)
    # Cal the diff btw the 1st and 2nd bias est
    for i in range(nSta):
        if xPar[1][i] < 0 or xPar[4][i] < 0:
            # One of the 2 bias parametrs is missing
            continue
        if xPar[2][i] < 2 or xPar[5][i] < 2:
            # One of the 2 bias parameters does not have enough obs
            continue
        xPar[6][i] = xPar[3][i] - xPar[0][i]
        # Sigma for diff
        xPar[7][i] = np.sqrt(xPar[1][i]**2 + xPar[4][i]**2)

    # Get ant && rec info
    cRec = ReadCtrl.GetStaRec0(fCtr, cSta)
    # Sort the station list
    cSta0 = cSta.copy()
    cRec0 = cRec.copy()
    for i in range(0, nSta-1):
        for j in range(i+1, nSta):
            if iSort == 1:
                # Sort along Rec type first
                if cRec0[0][j] > cRec0[0][i]:
                    continue
                elif cRec0[0][j] < cRec0[0][i]:
                    cTmp = cSta0[i]
                    cSta0[i] = cSta0[j]
                    cSta0[j] = cTmp
                    for k in range(4):
                        cTmp = cRec0[k][i]
                        cRec0[k][i] = cRec0[k][j]
                        cRec0[k][j] = cTmp
                elif cSta0[j] < cSta0[i]:
                    cTmp = cSta0[i]
                    cSta0[i] = cSta0[j]
                    cSta0[j] = cTmp
                    for k in range(4):
                        cTmp = cRec0[k][i]
                        cRec0[k][i] = cRec0[k][j]
                        cRec0[k][j] = cTmp
            elif iSort == 2:
                # Sort along Ant type first
                if cRec0[3][j] > cRec0[3][i]:
                    continue
                elif cRec0[3][j] < cRec0[3][i]:
                    cTmp = cSta0[i]
                    cSta0[i] = cSta0[j]
                    cSta0[j] = cTmp
                    for k in range(4):
                        cTmp = cRec0[k][i]
                        cRec0[k][i] = cRec0[k][j]
                        cRec0[k][j] = cTmp
                elif cSta0[j] < cSta0[i]:
                    cTmp = cSta0[i]
                    cSta0[i] = cSta0[j]
                    cSta0[j] = cTmp
                    for k in range(4):
                        cTmp = cRec0[k][i]
                        cRec0[k][i] = cRec0[k][j]
                        cRec0[k][j] = cTmp
            elif iSort == 3:
                # Sort along Rec type first, then sort along Ant type
                if cRec0[0][j] > cRec0[0][i]:
                    continue
                elif cRec0[0][j] < cRec0[0][i]:
                    cTmp = cSta0[i]
                    cSta0[i] = cSta0[j]
                    cSta0[j] = cTmp
                    for k in range(4):
                        cTmp = cRec0[k][i]
                        cRec0[k][i] = cRec0[k][j]
                        cRec0[k][j] = cTmp
                elif cRec0[3][j] > cRec0[3][i]:
                    continue
                elif cRec0[3][j] < cRec0[3][i]:
                    cTmp = cSta0[i]
                    cSta0[i] = cSta0[j]
                    cSta0[j] = cTmp
                    for k in range(4):
                        cTmp = cRec0[k][i]
                        cRec0[k][i] = cRec0[k][j]
                        cRec0[k][j] = cTmp
                elif cSta0[j] < cSta0[i]:
                    cTmp = cSta0[i]
                    cSta0[i] = cSta0[j]
                    cSta0[j] = cTmp
                    for k in range(4):
                        cTmp = cRec0[k][i]
                        cRec0[k][i] = cRec0[k][j]
                        cRec0[k][j] = cTmp
            else:
                # Sort only along station name
                if cSta0[j] < cSta0[i]:
                    cTmp = cSta0[i]
                    cSta0[i] = cSta0[j]
                    cSta0[j] = cTmp
                    for k in range(4):
                        cTmp = cRec0[k][i]
                        cRec0[k][i] = cRec0[k][j]
                        cRec0[k][j] = cTmp
    xPar0 = np.zeros((nSta, 2))
    for i in range(nSta):
        j = cSta.index(cSta0[i])
        xPar0[i, 0] = xPar[6][j]
        xPar0[i, 1] = xPar[7][j]
        # Not show the error bar
        xPar0[i, 1] = np.nan
        if lReport:
            cStr = '{:>4s} {:>20s} {:>20s}'.format(
                cSta0[i], cRec0[0][i], cRec0[3][i])
            print(cStr)

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(14, 4))
    x = np.arange(nSta)
    axs[0, 0].set_xlim(left=-1, right=nSta)

    w = 1/(1+1)
    axs[0, 0].bar(x+(0-1/2)*w, xPar0[:, 0], w, align='edge')
    axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)

    axs[0, 0].axhline(color='darkgray', linestyle='dashed', alpha=0.5, lw=0.8)
    axs[0, 0].set_ylabel(yLab+' [ns]', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    axs[0, 0].set_xlabel('Station Codes', fontname='Arial', fontsize=16)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(cSta0, rotation=90, c='darkblue',
                              fontdict={'fontsize': 5, 'fontname': 'monospace', 'horizontalalignment': 'center'})

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotGNSBias2(fPar, OutFilePrefix, OutFileSuffix):
    '''
    Plot GNSS system (geometric) bias, estimates and post-sigma
    '''

    # GISBX,GISBY,GISBZ,GISBT
    cSta = []
    xPar = [[], [], [], [], [], [], [], []]
    with open(fPar, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cWords[1][0:4] != 'GISB':
                continue
            # Exclude BSHM currently
            if cWords[3] == 'BSHM':
                continue
            if cWords[3] not in cSta:
                cSta.append(cWords[3])
                for i in range(8):
                    xPar[i].append(np.nan)
            iSta = cSta.index(cWords[3])
            if cWords[1][0:5] == 'GISBX' and int(cWords[14]) > 1:
                # in mm
                xPar[0][iSta] = float(cWords[10])
                xPar[1][iSta] = float(cWords[11])
            elif cWords[1][0:5] == 'GISBY' and int(cWords[14]) > 1:
                xPar[2][iSta] = float(cWords[10])
                xPar[3][iSta] = float(cWords[11])
            elif cWords[1][0:5] == 'GISBZ' and int(cWords[14]) > 1:
                xPar[4][iSta] = float(cWords[10])
                xPar[5][iSta] = float(cWords[11])
            elif cWords[1][0:5] == 'GISBT' and int(cWords[14]) > 1:
                xPar[6][iSta] = float(cWords[10])
                xPar[7][iSta] = float(cWords[11])
    nSta = len(cSta)

    fig, axs = plt.subplots(4, 1, squeeze=False, sharex='col', figsize=(8, 10))
    x = np.arange(nSta)
    yLabel = ['dX [mm]', 'dY [mm]', 'dZ [mm]', 'ZTD [mm]']
    for i in range(4):
        if np.count_nonzero(~np.isnan(xPar[2*i])) > 0:
            # Test whether this data exists
            axs[i, 0].errorbar(x, xPar[2*i], yerr=xPar[2*i+1],
                               fmt='.', capsize=5, markersize=6)
            strTmp = '{}{:6.2f} {}{:6.2f}'.format(
                'Mea=', np.nanmean(xPar[2*i]), 'Med=', np.nanmedian(xPar[2*i]))
            axs[i, 0].text(0.98, 0.98, strTmp, transform=axs[i, 0].transAxes, ha='right', va='top',
                           fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
        axs[i, 0].axhline(color='darkgray',
                          linestyle='dashed', alpha=0.5, lw=0.8)
        axs[i, 0].set_ylabel(yLabel[i], fontname='Arial', fontsize=16)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
    axs[i, 0].set_xlabel('Receiver Index', fontname='Arial', fontsize=16)
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotGNSBias3(fPar, OutFilePrefix, OutFileSuffix):
    '''
    Plot GNSS system (geometric) bias, in ENU coordinate system
    '''
    # GISBX,GISBY,GISBZ,GISBT
    cSta = []
    xPar = [[], [], [], [], [], [], [], []]
    # Station coordinates
    xPos = [[], [], []]
    with open(fPar, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            # Exclude BSHM currently
            if cWords[3] == 'BSHM':
                continue
            if cWords[1][0:4] == 'GISB':
                if cWords[3] not in cSta:
                    cSta.append(cWords[3])
                    for i in range(8):
                        xPar[i].append(np.nan)
                    for i in range(3):
                        xPos[i].append(np.nan)
                iSta = cSta.index(cWords[3])
                if cWords[1][0:5] == 'GISBX' and int(cWords[14]) > 1:
                    # in mm
                    xPar[0][iSta] = float(cWords[10])
                    xPar[1][iSta] = float(cWords[11])
                elif cWords[1][0:5] == 'GISBY' and int(cWords[14]) > 1:
                    xPar[2][iSta] = float(cWords[10])
                    xPar[3][iSta] = float(cWords[11])
                elif cWords[1][0:5] == 'GISBZ' and int(cWords[14]) > 1:
                    xPar[4][iSta] = float(cWords[10])
                    xPar[5][iSta] = float(cWords[11])
                elif cWords[1][0:5] == 'GISBT' and int(cWords[14]) > 1:
                    xPar[6][iSta] = float(cWords[10])
                    xPar[7][iSta] = float(cWords[11])
            elif cWords[1][0:5] == 'PXSTA' or \
                    cWords[1][0:5] == 'PYSTA' or \
                    cWords[1][0:5] == 'PZSTA':
                if cWords[3] not in cSta:
                    cSta.append(cWords[3])
                    for i in range(8):
                        xPar[i].append(np.nan)
                    for i in range(3):
                        xPos[i].append(np.nan)
                iSta = cSta.index(cWords[3])
                if cWords[1][0:5] == 'PXSTA':
                    # in meter
                    xPos[0][iSta] = float(cWords[10]) + float(cWords[6])
                elif cWords[1][0:5] == 'PYSTA':
                    xPos[1][iSta] = float(cWords[10]) + float(cWords[6])
                elif cWords[1][0:5] == 'PZSTA':
                    xPos[2][iSta] = float(cWords[10]) + float(cWords[6])
    nSta = len(cSta)
    # TRS -> ENU
    BLH = np.zeros(3)
    dXYZ = np.zeros(3)
    dENU = np.zeros(3)
    xENU = np.zeros((3, nSta))
    for i in range(nSta):
        BLH[0], BLH[1], BLH[2] = CorSys.XYZ2BLH(
            xPos[0][i], xPos[1][i], xPos[2][i])
        R = CorSys.RotENU2TRS(BLH[0], BLH[1])
        dXYZ[0] = xPar[0][i]
        dXYZ[1] = xPar[2][i]
        dXYZ[2] = xPar[4][i]
        dENU = np.dot(dXYZ, R)
        xENU[0][i] = dENU[0]
        xENU[1][i] = dENU[1]
        xENU[2][i] = dENU[2]

    fig, axs = plt.subplots(1, 2, squeeze=False, figsize=(11, 4))
    # East VS North
    axs[0, 0].plot(xENU[0], xENU[1], '.r', ms=5)
    axs[0, 0].axhline(color='darkgray', linestyle='dashed', alpha=0.5, lw=0.8)
    axs[0, 0].axvline(color='darkgray', linestyle='dashed', alpha=0.5, lw=0.8)
    axs[0, 0].set_xlabel('East [mm]', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    axs[0, 0].set_ylabel('North [mm]', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    # ZTD VS Up
    axs[0, 1].plot(xPar[6], xENU[2], '.r', ms=5)
    axs[0, 1].axhline(color='darkgray', linestyle='dashed', alpha=0.5, lw=0.8)
    axs[0, 1].axvline(color='darkgray', linestyle='dashed', alpha=0.5, lw=0.8)
    axs[0, 1].set_xlabel('ZTD [mm]', fontname='Arial', fontsize=16)
    for tl in axs[0, 1].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    axs[0, 1].set_ylabel('Up [mm]', fontname='Arial', fontsize=16)
    for tl in axs[0, 1].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotGNSBias4(fParList, cBias0, cSta0, lENU, OutFilePrefix, OutFileSuffix):
    '''
    Plot inter-system bias (ISB, POS and ZTD) time series for specified stations

    cBias0 --- Specified bias types
     cSta0 --- Specified station list
      lENU --- Whether transform the Pos Bias from TRS to ENU
    '''

    nBias = 5
    cBias = ['ISB', 'GISBX', 'GISBY', 'GISBZ', 'GISBT']
    cSta = []
    xBias = []

    nFile = len(fParList)
    for i in range(nFile):
        # Get the epoch from the file name
        YYYY = int(os.path.basename(fParList[i])[-7:-3])
        DOY = int(os.path.basename(fParList[i])[-3:])
        rMJD = GNSSTime.doy2mjd(YYYY, DOY)
        # Get station coordinates for TRS -> ENU
        cStaPos = []
        xPos = [[], [], []]
        with open(fParList[i], mode='rt') as fOb:
            for cLine in fOb:
                cWords = cLine.split()
                if cWords[1][0:5] in cBias:
                    j = cBias.index(cWords[1][0:5])
                elif cWords[1][0:3] in cBias:
                    j = cBias.index(cWords[1][0:3])
                elif cWords[1][0:5] == 'PXSTA' or \
                        cWords[1][0:5] == 'PYSTA' or \
                        cWords[1][0:5] == 'PZSTA':
                    if cWords[3] not in cStaPos:
                        cStaPos.append(cWords[3])
                        for k in range(3):
                            xPos[k].append(np.nan)
                    k = cStaPos.index(cWords[3])
                    if cWords[1][0:5] == 'PXSTA':
                        # in meter
                        xPos[0][k] = float(cWords[10]) + float(cWords[6])
                    elif cWords[1][0:5] == 'PYSTA':
                        xPos[1][k] = float(cWords[10]) + float(cWords[6])
                    elif cWords[1][0:5] == 'PZSTA':
                        xPos[2][k] = float(cWords[10]) + float(cWords[6])
                    continue
                else:
                    continue

                if cSta0[0] != 'ALL' and cWords[3] not in cSta0:
                    continue
                if cWords[3] not in cSta:
                    cSta.append(cWords[3])
                    for k in range(1+nBias):
                        xBias.append([])
                        for l in range(nFile):
                            xBias[(len(cSta)-1)*(1+nBias)+k].append(np.nan)
                k = cSta.index(cWords[3])
                if np.isnan(xBias[k*(1+nBias)][i]):
                    xBias[k*(1+nBias)][i] = rMJD
                xBias[k*(1+nBias)+1+j][i] = float(cWords[10])
        if not lENU:
            continue
        # TRS -> ENU
        BLH = np.zeros(3)
        dXYZ = np.zeros(3)
        dENU = np.zeros(3)
        for j in range(len(cSta)):
            # Whether GISB Pos exist
            if np.isnan(xBias[j*(1+nBias)+2][i]):
                continue
            # Index in the coordinates station list
            if cSta[j] not in cStaPos:
                sys.exit('Coordinates not found '+cSta[j]+' '+fParList[i])
            k = cStaPos.index(cSta[j])
            BLH[0], BLH[1], BLH[2] = CorSys.XYZ2BLH(
                xPos[0][k], xPos[1][k], xPos[2][k])
            R = CorSys.RotENU2TRS(BLH[0], BLH[1])
            dXYZ[0] = xBias[j*(1+nBias)+2][i]
            dXYZ[1] = xBias[j*(1+nBias)+3][i]
            dXYZ[2] = xBias[j*(1+nBias)+4][i]
            dENU = np.dot(dXYZ, R)
            xBias[j*(1+nBias)+2][i] = dENU[0]
            xBias[j*(1+nBias)+3][i] = dENU[1]
            xBias[j*(1+nBias)+4][i] = dENU[2]

    nSta = len(cSta)
    cSta1 = cSta
    cSta1.sort()

    if cBias0[0] == 'ALL':
        nBias0 = 3
        iBias0 = [0, 1, 2]
    else:
        nBias0 = 0
        iBias0 = []
        if 'ISB' in cBias0:
            nBias0 = nBias0+1
            iBias0.append(0)
        if 'POS' in cBias0:
            nBias0 = nBias0+1
            iBias0.append(1)
        if 'ZTD' in cBias0:
            nBias0 = nBias0+1
            iBias0.append(2)

    fig, axs = plt.subplots(nSta, nBias0, squeeze=False,
                            sharex='col', figsize=(nBias0*4, nSta*3))
    formatterx = mpl.ticker.StrMethodFormatter('{x:6.0f}')
    if lENU:
        cPos = ['dE', 'dN', 'dU']
    else:
        cPos = ['dX', 'dY', 'dZ']

    for i in range(nSta):
        k = cSta.index(cSta1[i])
        for j in range(nBias0):
            if iBias0[j] == 0:
                # ISB
                axs[i, j].plot(xBias[k*(1+nBias)],
                               xBias[k*(1+nBias)+1], '.--r', ms=4, lw=1)
                strTmp = cSta[k]+' ISB'
            elif iBias0[j] == 1:
                # POS
                axs[i, j].set_ylim(bottom=-25, top=25)
                axs[i, j].axhline(color='darkgray',
                                  linestyle='dashed', alpha=0.5, lw=1)
                axs[i, j].set_ylabel('[mm]', fontname='Arial', fontsize=12)
                axs[i, j].plot(xBias[k*(1+nBias)], xBias[k *
                               (1+nBias)+2], '.--r', ms=4, lw=1, label=cPos[0])
                axs[i, j].plot(xBias[k*(1+nBias)], xBias[k *
                               (1+nBias)+3], 'o--g', ms=4, lw=1, label=cPos[1])
                axs[i, j].plot(xBias[k*(1+nBias)], xBias[k *
                               (1+nBias)+4], '^--b', ms=4, lw=1, label=cPos[2])
                axs[i, j].legend(ncol=3, loc='upper right', bbox_to_anchor=(1.0, 1.0),
                                 framealpha=0.0, prop={'family': 'Arial', 'size': 10})
                strTmp = cSta[k]+' POS'
            else:
                # ZTD
                axs[i, j].set_ylim(bottom=-15, top=15)
                axs[i, j].axhline(color='darkgray',
                                  linestyle='dashed', alpha=0.5, lw=1)
                axs[i, j].set_ylabel('[mm]', fontname='Arial', fontsize=12)
                axs[i, j].plot(xBias[k*(1+nBias)],
                               xBias[k*(1+nBias)+5], '.--r', ms=4, lw=1)
                strTmp = cSta[k]+' ZTD'
            axs[i, j].text(0.02, 0.98, strTmp, transform=axs[i, j].transAxes, ha='left', va='top',
                           fontdict={'fontsize': 10, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkblue'})
            for tl in axs[i, j].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(10)
            if i == (nSta-1):
                axs[i, j].xaxis.set_major_formatter(formatterx)
                axs[i, j].set_xlabel('MJD', fontname='Arial', fontsize=12)
                for tl in axs[i, j].get_xticklabels():
                    tl.set_fontname('Arial')
                    tl.set_fontsize(10)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotGNSBias50(fParList, cBias, cStaX, iPlot, fCtrList, iSort, OutFilePrefix, OutFileSuffix):
    '''
    Plot the time series of specified (daily) GNSS recevier-specific
    time bias parameters for specified stations

    cBias --- Specified GNSS recevier-specific time bias parameter
    cStaX --- Depends on the first elemant, it specifies
              I, Stations to be Included
              E, Stations to be Excluded
    iPlot --- Specify which info should be plotted
              # 0, only plot the time series
              # 1, only plot the mean && std for each station
              # 2, plot both time series and mean/std
 fCtrList --- ctrl-file list which provides the rec && ant info for stations
    iSort --- Whether sort the parameters along rec/ant type
              # 0, Only sort along station names
              # 1, Sort along Rec type first
              # 2, Sort along Ant type first
              # 3, Sort along Rec type, then along Ant type

    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    xBias = []
    cSta = []
    for i in range(len(fParList)):
        with open(fParList[i], mode='rt') as fOb:
            for cLine in fOb:
                cWords = cLine.split()
                if cWords[1] != cBias:
                    continue
                if (cStaX[0] == "I" and (cWords[3] not in cStaX[1:])) or \
                   (cStaX[0] == "E" and (cWords[3] in cStaX[1:])):
                    continue
                if cWords[3] not in cSta:
                    cSta.append(cWords[3])
                    # epoch
                    xBias.append([])
                    # estimates
                    xBias.append([])
                    # post-sigma
                    xBias.append([])
                    # Number of obs
                    xBias.append([])
                iSta = cSta.index(cWords[3])
                xBias[iSta*4].append(float(cWords[12]))
                # m -> ns
                xBias[iSta*4+1].append(float(cWords[10])/299792458*1e9)
                # m -> ns
                xBias[iSta*4+2].append(float(cWords[11])/299792458*1e9)
                xBias[iSta*4+3].append(float(cWords[14]))
    nSta = len(cSta)
    if iSort >= 0:
        # Firstly, get the euqipment info for the stations
        # Rec type, Rec no., Rec firmware version, Ant type
        cRec = [[], [], [], []]
        for i in range(nSta):
            for j in range(4):
                cRec[j].append('****')
        # Search in each ctrl-file until find the info for all stations
        for i in range(len(fCtrList)):
            # Get the stations that still missing info
            cSta0 = []
            iSta0 = []
            for j in range(nSta):
                if cRec[1][j] == '****':
                    cSta0.append(cSta[j])
                    iSta0.append(j)
            nSta0 = len(cSta0)
            if nSta0 == 0:
                # Info of all stations are found
                break
            cRec0 = ReadCtrl.GetStaRec0(fCtrList[i], cSta0)
            # Fill the info list
            for j in range(nSta0):
                # Whether info for this station is found
                lFound = True
                for k in range(4):
                    if cRec0[k][j] == '****':
                        lFound = False
                        break
                if not lFound:
                    continue
                for k in range(4):
                    cRec[k][iSta0[j]] = cRec0[k][j]
        # Check if info found for all stations
        cSta0 = []
        for j in range(nSta):
            if cRec[1][j] == '****':
                cSta0.append(cSta[j])
        nSta0 = len(cSta0)
        if nSta0 > 0:
            print('Stations missing info:')
            for j in range(nSta0):
                print(cSta0[j])
            sys.exit(0)

        # Secondly, sort the station list
        cSta0 = cSta.copy()
        cRec0 = cRec.copy()
        for i in range(0, nSta-1):
            for j in range(i+1, nSta):
                if iSort == 1:
                    # Sort along Rec type first
                    if cRec0[0][j] > cRec0[0][i]:
                        continue
                    elif cRec0[0][j] < cRec0[0][i]:
                        # Switch the station
                        cTmp = cSta0[i]
                        cSta0[i] = cSta0[j]
                        cSta0[j] = cTmp
                        # Switch the equipment table
                        for k in range(4):
                            cTmp = cRec0[k][i]
                            cRec0[k][i] = cRec0[k][j]
                            cRec0[k][j] = cTmp
                    elif cSta0[j] < cSta0[i]:
                        cTmp = cSta0[i]
                        cSta0[i] = cSta0[j]
                        cSta0[j] = cTmp
                        for k in range(4):
                            cTmp = cRec0[k][i]
                            cRec0[k][i] = cRec0[k][j]
                            cRec0[k][j] = cTmp
                elif iSort == 2:
                    # Sort along Ant type first
                    if cRec0[3][j] > cRec0[3][i]:
                        continue
                    elif cRec0[3][j] < cRec0[3][i]:
                        cTmp = cSta0[i]
                        cSta0[i] = cSta0[j]
                        cSta0[j] = cTmp
                        for k in range(4):
                            cTmp = cRec0[k][i]
                            cRec0[k][i] = cRec0[k][j]
                            cRec0[k][j] = cTmp
                    elif cSta0[j] < cSta0[i]:
                        cTmp = cSta0[i]
                        cSta0[i] = cSta0[j]
                        cSta0[j] = cTmp
                        for k in range(4):
                            cTmp = cRec0[k][i]
                            cRec0[k][i] = cRec0[k][j]
                            cRec0[k][j] = cTmp
                elif iSort == 3:
                    # Sort along Rec type first, then sort along Ant type
                    if cRec0[0][j] > cRec0[0][i]:
                        continue
                    elif cRec0[0][j] < cRec0[0][i]:
                        cTmp = cSta0[i]
                        cSta0[i] = cSta0[j]
                        cSta0[j] = cTmp
                        for k in range(4):
                            cTmp = cRec0[k][i]
                            cRec0[k][i] = cRec0[k][j]
                            cRec0[k][j] = cTmp
                    elif cRec0[3][j] > cRec0[3][i]:
                        continue
                    elif cRec0[3][j] < cRec0[3][i]:
                        cTmp = cSta0[i]
                        cSta0[i] = cSta0[j]
                        cSta0[j] = cTmp
                        for k in range(4):
                            cTmp = cRec0[k][i]
                            cRec0[k][i] = cRec0[k][j]
                            cRec0[k][j] = cTmp
                    elif cSta0[j] < cSta0[i]:
                        cTmp = cSta0[i]
                        cSta0[i] = cSta0[j]
                        cSta0[j] = cTmp
                        for k in range(4):
                            cTmp = cRec0[k][i]
                            cRec0[k][i] = cRec0[k][j]
                            cRec0[k][j] = cTmp
                else:
                    # Sort only along station name
                    if cSta0[j] < cSta0[i]:
                        cTmp = cSta0[i]
                        cSta0[i] = cSta0[j]
                        cSta0[j] = cTmp
                        for k in range(4):
                            cTmp = cRec0[k][i]
                            cRec0[k][i] = cRec0[k][j]
                            cRec0[k][j] = cTmp
        # Thirdly, switch the data according to the sorted station list
        xBias0 = []
        for i in range(nSta):
            j = cSta.index(cSta0[i])
            for k in range(4):
                xBias0.append(xBias[j*4+k])
        # Fourthly, rename the variables
        cSta = cSta0
        cRec = cRec0
        xBias = xBias0

    fOut = open(OutFilePrefix+os.path.splitext(OutFileSuffix)[0], 'w')
    strTmp = '{: <4s} {: <20s} {: <20s} {: >8s} {: >8s}\n'.format(
        'Sta', 'Rec', 'Ant', 'Mean', 'STD')
    fOut.write(strTmp)

    Mea = []
    Std = []
    for i in range(nSta):
        Mea0 = np.mean(xBias[i*4+1])
        Sig0 = np.std(xBias[i*4+1])
        Mea.append(Mea0)
        Std.append(Sig0)
        strTmp = '{: <4s} {: <20s} {: <20s} {: >8.2f} {: >8.2f}\n'.format(cSta[i],
                                                                          cRec[0][i], cRec[3][i], Mea0, Sig0)
        fOut.write(strTmp)
    fOut.close()

    if iPlot == 0:
        # Only the time series
        fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 4))
        for i in range(nSta):
            axs[0, 0].plot(xBias[i*4], xBias[i*4+1], 'o--', ms=3, lw=1)
        axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
        axs[0, 0].ticklabel_format(axis='y', useOffset=False, useMathText=True)
        axs[0, 0].set_ylabel('Estimates [ns]', fontname='Arial', fontsize=16)
        for tl in axs[0, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        axs[0, 0].set_xlabel('Modified Julian Day',
                             fontname='Arial', fontsize=16)
        axs[0, 0].ticklabel_format(axis='x', useOffset=False, useMathText=True)
        for tl in axs[0, 0].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
    elif iPlot == 1:
        # Only plot the mean && std for each station
        fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 4))
        x = np.arange(nSta)
        axs[0, 0].set_xlim(left=-1, right=nSta)

        w = 1/(1+1)
        axs[0, 0].bar(x+(0-1/2)*w, Mea, w, align='edge', yerr=Std,
                      error_kw=dict(ecolor='r', capsize=2, elinewidth=1, capthick=0.6))
        axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)

        axs[0, 0].ticklabel_format(axis='y', useOffset=False, useMathText=True)
        axs[0, 0].set_ylabel('Mean and STD [ns]',
                             fontname='Arial', fontsize=16)
        for tl in axs[0, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        axs[0, 0].set_xlabel('Station Codes', fontname='Arial', fontsize=16)
        axs[0, 0].set_xticks(x)
        axs[0, 0].set_xticklabels([])
        # axs[0,0].set_xticklabels(cSta,rotation=90,c='darkblue',
        #     fontdict={'fontsize':5,'fontname':'monospace','horizontalalignment':'center'})
    else:
        # Plot both time series and mean/std
        fig, axs = plt.subplots(2, 1, squeeze=False, figsize=(12, 8))
        for i in range(nSta):
            axs[0, 0].plot(xBias[i*4], xBias[i*4+1], 'o--', ms=3, lw=1)
        axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
        axs[0, 0].ticklabel_format(axis='y', useOffset=False, useMathText=True)
        axs[0, 0].set_ylabel('Estimates [ns]', fontname='Arial', fontsize=16)
        for tl in axs[0, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        axs[0, 0].set_xlabel('Modified Julian Day',
                             fontname='Arial', fontsize=16)
        axs[0, 0].ticklabel_format(axis='x', useOffset=False, useMathText=True)
        for tl in axs[0, 0].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        # the 2nd figure
        x = np.arange(nSta)
        axs[1, 0].set_xlim(left=-1, right=nSta)

        w = 1/(1+1)
        axs[1, 0].bar(x+(0-1/2)*w, Mea, w, align='edge', yerr=Std,
                      error_kw=dict(ecolor='r', capsize=2, elinewidth=1, capthick=0.6))
        axs[1, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)

        axs[1, 0].ticklabel_format(axis='y', useOffset=False, useMathText=True)
        axs[1, 0].set_ylabel('Mean and STD [ns]',
                             fontname='Arial', fontsize=16)
        for tl in axs[1, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        axs[1, 0].set_xlabel('Station Codes', fontname='Arial', fontsize=16)
        axs[1, 0].set_xticks(x)
        # axs[1,0].set_xticklabels([])
        axs[1, 0].set_xticklabels(cSta, rotation=90, c='darkblue',
                                  fontdict={'fontsize': 5, 'fontname': 'monospace', 'horizontalalignment': 'center'})

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotGNSBias51(fBias, cBias, cStaX, iPlot, fCtrList, iSort, OutFilePrefix, OutFileSuffix):
    '''
    Similar to PlotGNSBias50, but take the extracted bias-file as the input
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    xBias = []
    cSta = []
    iSta = []
    with open(fBias, mode='rt') as fOb:
        iLine = 0
        for cLine in fOb:
            iLine = iLine+1
            if iLine == 1:
                # The first line is the station list
                cWords = cLine.split()
                for i in range(len(cWords)-1):
                    if (cStaX[0] == "I" and (cWords[i] not in cStaX[1:])) or \
                       (cStaX[0] == "E" and (cWords[i] in cStaX[1:])):
                        continue
                    cSta.append(cWords[i])
                    iSta.append(i)
                nSta = len(cSta)
                # The epoch list, common for all stations
                xBias.append([])
                for j in range(nSta):
                    xBias.append([])
            elif len(cLine) < 5:
                continue
            else:
                cWords = cLine.split()
                # epoch
                xBias[0].append(float(cWords[0])+float(cWords[1])/86400)
                for j in range(nSta):
                    if cWords[3+iSta[j]] == 'NaN':
                        xBias[1+j].append(np.nan)
                    else:
                        # m -> ns
                        xBias[1+j].append(float(cWords[3+iSta[j]]
                                                )/299792458*1e9)
    if iSort >= 0:
        # Firstly, get the euqipment info for the stations
        # Rec type, Rec no., Rec firmware version, Ant type
        cRec = [[], [], [], []]
        for i in range(nSta):
            for j in range(4):
                cRec[j].append('****')
        # Search in each ctrl-file until find the info for all stations
        for i in range(len(fCtrList)):
            # Get the stations that still missing info
            cSta0 = []
            iSta0 = []
            for j in range(nSta):
                if cRec[1][j] == '****':
                    cSta0.append(cSta[j])
                    iSta0.append(j)
            nSta0 = len(cSta0)
            if nSta0 == 0:
                # Info of all stations are found
                break
            cRec0 = ReadCtrl.GetStaRec0(fCtrList[i], cSta0)
            # Fill the info list
            for j in range(nSta0):
                # Whether info for this station is found
                lFound = True
                for k in range(4):
                    if cRec0[k][j] == '****':
                        lFound = False
                        break
                if not lFound:
                    continue
                for k in range(4):
                    cRec[k][iSta0[j]] = cRec0[k][j]
        # Check if info found for all stations
        cSta0 = []
        for j in range(nSta):
            if cRec[1][j] == '****':
                cSta0.append(cSta[j])
        nSta0 = len(cSta0)
        if nSta0 > 0:
            print('Stations missing info:')
            for j in range(nSta0):
                print(cSta0[j])
            sys.exit(0)

        # Secondly, sort the station list
        cSta0 = cSta.copy()
        cRec0 = cRec.copy()
        for i in range(0, nSta-1):
            for j in range(i+1, nSta):
                if iSort == 1:
                    # Sort along Rec type first
                    if cRec0[0][j] > cRec0[0][i]:
                        continue
                    elif cRec0[0][j] < cRec0[0][i]:
                        # Switch the station
                        cTmp = cSta0[i]
                        cSta0[i] = cSta0[j]
                        cSta0[j] = cTmp
                        # Switch the equipment table
                        for k in range(4):
                            cTmp = cRec0[k][i]
                            cRec0[k][i] = cRec0[k][j]
                            cRec0[k][j] = cTmp
                    elif cSta0[j] < cSta0[i]:
                        cTmp = cSta0[i]
                        cSta0[i] = cSta0[j]
                        cSta0[j] = cTmp
                        for k in range(4):
                            cTmp = cRec0[k][i]
                            cRec0[k][i] = cRec0[k][j]
                            cRec0[k][j] = cTmp
                elif iSort == 2:
                    # Sort along Ant type first
                    if cRec0[3][j] > cRec0[3][i]:
                        continue
                    elif cRec0[3][j] < cRec0[3][i]:
                        cTmp = cSta0[i]
                        cSta0[i] = cSta0[j]
                        cSta0[j] = cTmp
                        for k in range(4):
                            cTmp = cRec0[k][i]
                            cRec0[k][i] = cRec0[k][j]
                            cRec0[k][j] = cTmp
                    elif cSta0[j] < cSta0[i]:
                        cTmp = cSta0[i]
                        cSta0[i] = cSta0[j]
                        cSta0[j] = cTmp
                        for k in range(4):
                            cTmp = cRec0[k][i]
                            cRec0[k][i] = cRec0[k][j]
                            cRec0[k][j] = cTmp
                elif iSort == 3:
                    # Sort along Rec type first, then sort along Ant type
                    if cRec0[0][j] > cRec0[0][i]:
                        continue
                    elif cRec0[0][j] < cRec0[0][i]:
                        cTmp = cSta0[i]
                        cSta0[i] = cSta0[j]
                        cSta0[j] = cTmp
                        for k in range(4):
                            cTmp = cRec0[k][i]
                            cRec0[k][i] = cRec0[k][j]
                            cRec0[k][j] = cTmp
                    elif cRec0[3][j] > cRec0[3][i]:
                        continue
                    elif cRec0[3][j] < cRec0[3][i]:
                        cTmp = cSta0[i]
                        cSta0[i] = cSta0[j]
                        cSta0[j] = cTmp
                        for k in range(4):
                            cTmp = cRec0[k][i]
                            cRec0[k][i] = cRec0[k][j]
                            cRec0[k][j] = cTmp
                    elif cSta0[j] < cSta0[i]:
                        cTmp = cSta0[i]
                        cSta0[i] = cSta0[j]
                        cSta0[j] = cTmp
                        for k in range(4):
                            cTmp = cRec0[k][i]
                            cRec0[k][i] = cRec0[k][j]
                            cRec0[k][j] = cTmp
                else:
                    # Sort only along station name
                    if cSta0[j] < cSta0[i]:
                        cTmp = cSta0[i]
                        cSta0[i] = cSta0[j]
                        cSta0[j] = cTmp
                        for k in range(4):
                            cTmp = cRec0[k][i]
                            cRec0[k][i] = cRec0[k][j]
                            cRec0[k][j] = cTmp
        # Thirdly, switch the data according to the sorted station list
        xBias0 = []
        xBias0.append(xBias[0])
        for i in range(nSta):
            j = cSta.index(cSta0[i])
            xBias0.append(xBias[j+1])
        # Fourthly, rename the variables
        cSta = cSta0
        cRec = cRec0
        xBias = xBias0

    fOut = open(OutFilePrefix+os.path.splitext(OutFileSuffix)[0], 'w')
    strTmp = '{: <4s} {: <20s} {: <20s} {: <20s} {: <20s} {: >8s} {: >8s}\n'.format('Sta',
                                                                                    'RecTyp', 'RecSer', 'RecVer', 'AntTyp', 'Mean', 'STD')
    fOut.write(strTmp)

    Mea = []
    Std = []
    for i in range(nSta):
        Mea0 = np.nanmean(xBias[i+1])
        Sig0 = np.nanstd(xBias[i+1])
        Mea.append(Mea0)
        Std.append(Sig0)
        strTmp = '{: <4s} {: <20s} {: <20s} {: <20s} {: <20s} {: >8.2f} {: >8.2f}\n'.format(cSta[i],
                                                                                            cRec[0][i], cRec[1][i], cRec[2][i], cRec[3][i], Mea0, Sig0)
        fOut.write(strTmp)
    fOut.write('{: <4s} {: <20s} {: <20s} {: <20s} {: <20s} {: >8.2f} {: >8.2f}\n'.format('Avg',
               'Mean of STD', 'STD of STD', '', '', np.mean(Std), np.std(Std)))
    fOut.close()

    if iPlot == 0:
        # Only the time series
        fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 4))
        for i in range(nSta):
            axs[0, 0].plot(xBias[0], xBias[i+1], 'o--', ms=3, lw=1)
        axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
        axs[0, 0].ticklabel_format(axis='y', useOffset=False, useMathText=True)
        axs[0, 0].set_ylabel('Estimates [ns]', fontname='Arial', fontsize=16)
        for tl in axs[0, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        axs[0, 0].set_xlabel('Modified Julian Day',
                             fontname='Arial', fontsize=16)
        axs[0, 0].ticklabel_format(axis='x', useOffset=False, useMathText=True)
        for tl in axs[0, 0].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
    elif iPlot == 1:
        # Only plot the mean && std for each station
        fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 4))
        x = np.arange(nSta)
        axs[0, 0].set_xlim(left=-1, right=nSta)

        w = 1/(1+1)
        axs[0, 0].bar(x+(0-1/2)*w, Mea, w, align='edge', yerr=Std,
                      error_kw=dict(ecolor='r', capsize=2, elinewidth=1, capthick=0.6))
        axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)

        axs[0, 0].ticklabel_format(axis='y', useOffset=False, useMathText=True)
        axs[0, 0].set_ylabel('Mean and STD [ns]',
                             fontname='Arial', fontsize=16)
        for tl in axs[0, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        axs[0, 0].set_xlabel('Stations', fontname='Arial', fontsize=16)
        axs[0, 0].set_xticks(x)
        axs[0, 0].set_xticklabels([])
        # axs[0,0].set_xticklabels(cSta,rotation=90,c='darkblue',
        #     fontdict={'fontsize':5,'fontname':'monospace','horizontalalignment':'center'})
    else:
        # Plot both time series and mean/std
        fig, axs = plt.subplots(2, 1, squeeze=False, figsize=(12, 8))
        for i in range(nSta):
            axs[0, 0].plot(xBias[0], xBias[i+1], 'o--', ms=3, lw=1)
        axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
        axs[0, 0].ticklabel_format(axis='y', useOffset=False, useMathText=True)
        axs[0, 0].set_ylabel('Estimates [ns]', fontname='Arial', fontsize=16)
        for tl in axs[0, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        axs[0, 0].set_xlabel('Modified Julian Day',
                             fontname='Arial', fontsize=16)
        axs[0, 0].ticklabel_format(axis='x', useOffset=False, useMathText=True)
        for tl in axs[0, 0].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        # the 2nd figure
        x = np.arange(nSta)
        axs[1, 0].set_xlim(left=-1, right=nSta)

        w = 1/(1+1)
        axs[1, 0].bar(x+(0-1/2)*w, Mea, w, align='edge', yerr=Std,
                      error_kw=dict(ecolor='r', capsize=2, elinewidth=1, capthick=0.6))
        axs[1, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)

        axs[1, 0].ticklabel_format(axis='y', useOffset=False, useMathText=True)
        axs[1, 0].set_ylabel('Mean and STD [ns]',
                             fontname='Arial', fontsize=16)
        for tl in axs[1, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        axs[1, 0].set_xlabel('Stations', fontname='Arial', fontsize=16)
        axs[1, 0].set_xticks(x)
        # axs[1,0].set_xticklabels([])
        axs[1, 0].set_xticklabels(cSta, rotation=90, c='darkblue',
                                  fontdict={'fontsize': 5, 'fontname': 'monospace', 'horizontalalignment': 'center'})

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotGNSPCO1(fPar, OutFilePrefix, OutFileSuffix):
    '''
    Plot the estimates and its post-prior sigma of GNSS PCO
    '''

    cAnt = []
    PCO = [[], [], [], [], [], []]
    with open(fPar, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cWords[1][0:6] != 'GNSPCO':
                continue
            if int(cWords[2]) != 0:
                # Receiver antenna
                if cWords[3] not in cAnt:
                    cAnt.append(cWords[3])
                    for i in range(6):
                        PCO[i].append(np.nan)
                iAnt = cAnt.index(cWords[3])
            else:
                # Satellite antenna
                if cWords[5] not in cAnt:
                    cAnt.append(cWords[5])
                    for i in range(6):
                        PCO[i].append(np.nan)
                iAnt = cAnt.index(cWords[5])
            if cWords[1][0:7] == 'GNSPCOE' or cWords[1][0:7] == 'GNSPCOX':
                PCO[0][iAnt] = float(cWords[10])
                PCO[1][iAnt] = float(cWords[11])
            elif cWords[1][0:7] == 'GNSPCON' or cWords[1][0:7] == 'GNSPCOY':
                PCO[2][iAnt] = float(cWords[10])
                PCO[3][iAnt] = float(cWords[11])
            else:
                PCO[4][iAnt] = float(cWords[10])
                PCO[5][iAnt] = float(cWords[11])
    nAnt = len(cAnt)

    fig, axs = plt.subplots(3, 1, sharex='col', squeeze=False, figsize=(8, 9))

    x = np.arange(nAnt)
    # axs[0].set_ylim(bottom=-150,top=150)
    axs[0, 0].errorbar(x, PCO[0], yerr=PCO[1], fmt='.', capsize=5, ms=6)
    axs[0, 0].axhline(color='darkgray', linestyle='dashed', alpha=0.5, lw=0.8)
    axs[0, 0].set_ylabel('X/E dPCO [mm]', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    # axs[1].set_ylim(bottom=-150,top=75)
    axs[1, 0].errorbar(x, PCO[2], yerr=PCO[3], fmt='.', capsize=5, ms=6)
    axs[1, 0].axhline(color='darkgray', linestyle='dashed', alpha=0.5, lw=0.8)
    axs[1, 0].set_ylabel('Y/N dPCO [mm]', fontname='Arial', fontsize=16)
    for tl in axs[1, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    # axs[2].set_ylim(bottom=-500,top=300)
    axs[2, 0].errorbar(x, PCO[4], yerr=PCO[5], fmt='.', capsize=5, ms=6)
    axs[2, 0].axhline(color='darkgray', linestyle='dashed', alpha=0.5, lw=0.8)
    axs[2, 0].set_ylabel('Z/U PCO [mm]', fontname='Arial', fontsize=16)
    for tl in axs[2, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    axs[2, 0].set_xlabel('Antenna Index', fontname='Arial', fontsize=16)
    for tl in axs[2, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotGNSPCO2(fPar, cStaAnt, OutFilePrefix, OutFileSuffix):
    '''
    Plot the estimates of GNSS receiver PCO in ENU. Put stations equipped with
    the same antenna together for comparison

    cStaAnt --- station list of each antenna
    '''

    cSta = []
    PCO = [[], [], [], [], [], []]
    with open(fPar, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cWords[1][0:6] != 'GNSPCO':
                continue
            if int(cWords[2]) == 0:
                continue
            # Receiver antenna
            if cWords[3] not in cSta:
                cSta.append(cWords[3])
                for i in range(6):
                    PCO[i].append(np.nan)
            iAnt = cSta.index(cWords[3])
            if cWords[1][0:7] == 'GNSPCOE':
                PCO[0][iAnt] = float(cWords[10])
                PCO[1][iAnt] = float(cWords[11])
            elif cWords[1][0:7] == 'GNSPCON':
                PCO[2][iAnt] = float(cWords[10])
                PCO[3][iAnt] = float(cWords[11])
            else:
                PCO[4][iAnt] = float(cWords[10])
                PCO[5][iAnt] = float(cWords[11])
    nSta = len(cSta)

    fig, axs = plt.subplots(1, 2, squeeze=False, figsize=(11, 4))
    axs[0, 0].axhline(color='darkgray', linestyle='dashed', alpha=0.5, lw=0.8)
    axs[0, 0].axvline(color='darkgray', linestyle='dashed', alpha=0.5, lw=0.8)
    axs[0, 0].set_xlabel('East [mm]', fontname='Arial', fontsize=16)
    axs[0, 0].set_ylabel('North [mm]', fontname='Arial', fontsize=16)

    axs[0, 1].axhline(color='darkgray', linestyle='dashed', alpha=0.5, lw=0.8)
    axs[0, 1].axvline(color='darkgray', linestyle='dashed', alpha=0.5, lw=0.8)
    axs[0, 1].set_xlabel('Station Index', fontname='Arial', fontsize=16)
    axs[0, 1].set_ylabel('Up [mm]', fontname='Arial', fontsize=16)

    nAnt = len(cStaAnt)
    m = 0
    for i in range(nAnt):
        nStaAnt = len(cStaAnt[i])
        # East VS North
        x = np.zeros(nStaAnt)
        y = np.zeros(nStaAnt)
        x[:] = np.nan
        y[:] = np.nan
        for j in range(nStaAnt):
            if cStaAnt[i][j] not in cSta:
                continue
            k = cSta.index(cStaAnt[i][j])
            x[j] = PCO[0][k]
            y[j] = PCO[2][k]
        axs[0, 0].plot(x, y, '.', ms=7)
        # Up
        x = np.arange(nStaAnt)+m
        y = np.zeros(nStaAnt)
        y[:] = np.nan
        for j in range(nStaAnt):
            if cStaAnt[i][j] not in cSta:
                continue
            k = cSta.index(cStaAnt[i][j])
            y[j] = PCO[4][k]
        axs[0, 1].plot(x, y, '.', ms=7)
        m = m+nStaAnt

    for tl in axs[0, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    for tl in axs[0, 1].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    for tl in axs[0, 1].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotISLBias1(fParList, cSat0, lNorm, lRange, OutFilePrefix, OutFileSuffix):
    '''
    Plot the MEAN && STD of ISL bias parameters for specified satellites and
    write out to a file.

    cSat0 --- PRN list of specified satellites
    lNorm --- Whether remove the MEAN, i.e. only plot the STD
   lRange --- Whether use meter instead of nanosec as the unit
              In this case, if lNorm==True, use cm for STD
    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nFile = len(fParList)
    cSat = []
    cBias = []
    xBias = []
    cPar = []
    for i in range(nFile):
        with open(fParList[i], mode='rt') as fOb:
            for cLine in fOb:
                cWords = cLine.split()
                if cWords[1][0:7] != 'ISLBIAS':
                    continue
                if cSat0[0] != 'ALL' and cWords[5] not in cSat0:
                    continue
                if cWords[5] not in cSat:
                    cSat.append(cWords[5])
                    cBias.append([])
                    xBias.append([])
                j = cSat.index(cWords[5])
                # the overall parameter set
                if cWords[1] not in cPar:
                    cPar.append(cWords[1])
                if cWords[1] not in cBias[j]:
                    cBias[j].append(cWords[1])
                    # Epoch
                    xBias[j].append([])
                    # Estimates
                    xBias[j].append([])
                    # Post-priori sigma
                    xBias[j].append([])
                k = cBias[j].index(cWords[1])
                # Epoch, in MJD
                xBias[j][3*k].append(float(cWords[12]))
                # Estimates and post-priori Sigma
                if not lRange:
                    # in nanosec
                    xBias[j][3*k+1].append(float(cWords[10])/299792458*1e9)
                    xBias[j][3*k+2].append(float(cWords[11])/299792458*1e9)
                else:
                    # in meters
                    xBias[j][3*k+1].append(float(cWords[10]))
                    xBias[j][3*k+2].append(float(cWords[11]))
    nSat = len(cSat)
    cPRN = cSat.copy()
    cPRN.sort()
    nPar = len(cPar)
    cPar.sort()

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(nSat*0.6, 4))
    x = np.arange(nSat)

    axs[0, 0].set_xlim(left=-1, right=nSat)
    # if lNorm:
    #     axs[0,0].set_ylim(bottom=-15,top=15)

    fOut = open(OutFilePrefix+os.path.splitext(OutFileSuffix)[0], 'w')
    cStr = []
    for i in range(nSat):
        cStr.append([])
        cStr[i] = cPRN[i]
    strTmp = 'PRN'
    for i in range(nPar):
        strTmp = strTmp + \
            ' {: >20s} {: >20s}'.format(cPar[i]+'_Mea', cPar[i]+'_STD')
    fOut.write(strTmp+'\n')

    # the width of the bars
    w = 1/(nPar+1)
    for i in range(nPar):
        Mea = np.zeros(nSat)
        Mea[:] = np.nan
        Std = np.zeros(nSat)
        Std[:] = np.nan
        for j in range(nSat):
            iSat = cSat.index(cPRN[j])
            iPar = cBias[iSat].index(cPar[i])
            Mea[j] = np.mean(xBias[iSat][3*iPar+1])
            Std[j] = np.std(xBias[iSat][3*iPar+1])
            cStr[j] = cStr[j]+' {: >20.4f}'.format(Mea[j])
            if lNorm:
                Mea[j] = 0.0
                if lRange:
                    # m -> cm
                    Std[j] = Std[j]*1e2
            cStr[j] = cStr[j]+' {: >20.4f}'.format(Std[j])
            if i == nPar-1:
                # Write to output file
                fOut.write(cStr[j]+'\n')
        axs[0, 0].bar(x+(i-nPar/2)*w, Mea[:], w, align='edge', label=cPar[i],
                      yerr=Std[:], ecolor='darkred', capsize=3)
    fOut.close()
    axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
    axs[0, 0].set_axisbelow(True)

    # axs[0,0].legend(ncol=nPar,loc='upper center',bbox_to_anchor=(0.5,1.0),
    #                 prop={'family':'Arial','size':14})

    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(
        cPRN, fontdict={'fontsize': 14, 'fontname': 'Arial'})
    axs[0, 0].set_xlabel('Satellite PRNs', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    if not lRange:
        axs[0, 0].set_ylabel('[ns]', fontname='Arial', fontsize=16)
    elif lNorm:
        axs[0, 0].set_ylabel('[cm]', fontname='Arial', fontsize=16)
    else:
        axs[0, 0].set_ylabel('[m]', fontname='Arial', fontsize=16)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotISLBias2(fParList, cSat0, cPar0, lNorm, lRange, OutFilePrefix, OutFileSuffix):
    '''
    Plot ISL bias parameter time series for specific satellite(s) within
    a single axis

     cSat0 --- Specified satellite list
     cPar0 --- Specified ISL bias par. If cPar0[0]=='ISLBIAS', means
               ploting for all ISL bias parameters
     lNorm --- Whether remove the mean of bias estimates
    lRange --- Whether present in range [m/cm], i.e.
               if lNorm==True, use cm for STD
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nFile = len(fParList)
    cSat = []
    cBias = []
    xBias = []
    for i in range(nFile):
        with open(fParList[i], mode='rt') as fOb:
            for cLine in fOb:
                cWords = cLine.split()
                if cWords[1][0:7] != 'ISLBIAS':
                    continue
                if cPar0[0] != 'ISLBIAS' and cWords[1] not in cPar0:
                    continue
                if cSat0[0] != 'ALL' and cWords[5] not in cSat0:
                    continue
                # Exclude not-estimated ones
                if int(cWords[14]) <= 1:
                    continue
                if cWords[5] not in cSat:
                    cSat.append(cWords[5])
                    cBias.append([])
                    xBias.append([])
                j = cSat.index(cWords[5])
                if cWords[1] not in cBias[j]:
                    cBias[j].append(cWords[1])
                    # Epoch
                    xBias[j].append([])
                    # Estimates
                    xBias[j].append([])
                    # Post-priori sigma
                    xBias[j].append([])
                k = cBias[j].index(cWords[1])
                # Epoch
                xBias[j][3*k].append(float(cWords[12]))
                # Estimates and post-priori Sigma
                if not lRange:
                    # in nanosec
                    xBias[j][3*k+1].append(float(cWords[10])/299792458*1e9)
                    xBias[j][3*k+2].append(float(cWords[11])/299792458*1e9)
                else:
                    # in meters
                    xBias[j][3*k+1].append(float(cWords[10]))
                    xBias[j][3*k+2].append(float(cWords[11]))
    nSat = len(cSat)
    cPRN = cSat.copy()
    cPRN.sort()

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 4))

    axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
    axs[0, 0].set_axisbelow(True)
    # axs[0,0].set_ylim(bottom=-30,top=30)
    axs[0, 0].set_prop_cycle(color=['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r',
                                    'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g',
                                    'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b',
                                    'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c',
                                    'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm', 'm',
                                    'y', 'y', 'y', 'y', 'y', 'y', 'y', 'y', 'y',
                                    'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k', 'k'],
                             marker=['.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                     '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                     '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                     '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                     '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                     '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                     '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X'])
    # Report to the terminal
    if lRange:
        print('{: <3s} {: >15s} {: >13s} {: >13s} {: >13s}'.format('PRN',
              'cBias', 'Mean[m]', 'STD[m]', 'AvgSig[m]'))
    else:
        print('{: <3s} {: >15s} {: >13s} {: >13s} {: >13s}'.format('PRN',
              'cBias', 'Mean[ns]', 'STD[ns]', 'AvgSig[ns]'))
    for iPRN in range(nSat):
        i = cSat.index(cPRN[iPRN])
        cPar = cBias[i].copy()
        cPar.sort()
        for iPar in range(len(cPar)):
            j = cBias[i].index(cPar[iPar])
            # Epoch
            x = np.array(xBias[i][3*j])
            # Estimates
            y = np.array(xBias[i][3*j+1])
            # Post-sigma
            z = np.array(xBias[i][3*j+2])
            # Mean of estimates and its post-priori sigma
            Mea = np.mean(y)
            Sig = np.mean(z)
            Std = np.std(y)
            if lNorm and not lRange:
                # Remove the mean
                y = y - Mea
            elif lNorm and lRange:
                # meter -> cm
                y = (y - Mea)*1e2
                z = z*1e2
            ind = np.argsort(x)
            # axs[0, 0].errorbar(x[ind], y[ind], yerr=z[ind], capsize=4,
            #                    ls='--', lw=1, label=cSat[i]+' '+cBias[i][j])
            axs[0, 0].plot(x[ind], y[ind], ls='--', lw=1,
                           label=cSat[i]+' '+cBias[i][j])
            strTmp = '{: <3s} {: >15s} {: >13.3f} {: >13.3f} {: >13.3f}'.format(cSat[i],
                                                                                cBias[i][j], Mea, Std, Sig)
            print(strTmp)
    axs[0, 0].legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5), framealpha=0.6,
                     labelspacing=0.1, borderpad=0.1, prop={'family': 'Arial', 'size': 14})
    if lNorm and lRange:
        axs[0, 0].set_ylabel('Hardware Delays [cm]',
                             fontname='Arial', fontsize=16)
    elif not lNorm and lRange:
        axs[0, 0].set_ylabel('Hardware Delays [m]',
                             fontname='Arial', fontsize=16)
    else:
        axs[0, 0].set_ylabel('Hardware Delays [ns]',
                             fontname='Arial', fontsize=16)
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


def PlotISLBias3(cSer, fSerList, cSat0, lRange, OutFilePrefix, OutFileSuffix):
    '''
    Plot the comparison of STD of ISL bias parameters for specified satellites
    from different solutions

    cSat0 --- List of specified satellites
   lRange --- Present in range unit (cm) or time unit (ns)
    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nSer = len(cSer)
    cSat = []
    cBias = []
    xBias = []
    cPar = []
    for iSer in range(nSer):
        nFile = len(fSerList[iSer])
        for i in range(nFile):
            with open(fSerList[iSer][i], mode='rt') as fOb:
                for cLine in fOb:
                    cWords = cLine.split()
                    if cWords[1][0:7] != 'ISLBIAS':
                        continue
                    if cSat0[0] != 'ALL' and cWords[5] not in cSat0:
                        continue
                    if cWords[5] not in cSat:
                        # New satellite
                        cSat.append(cWords[5])
                        for j in range(nSer):
                            # bias list for this satellite in each solutions
                            cBias.append([])
                            xBias.append([])
                    j = cSat.index(cWords[5])
                    # the overall parameter set
                    if cWords[1] not in cPar:
                        cPar.append(cWords[1])
                    if cWords[1] not in cBias[j*nSer+iSer]:
                        # new bias for this satellite in the current solution
                        cBias[j*nSer+iSer].append(cWords[1])
                        # Epoch
                        xBias[j*nSer+iSer].append([])
                        # Estimates
                        xBias[j*nSer+iSer].append([])
                        # Post-priori sigma
                        xBias[j*nSer+iSer].append([])
                    k = cBias[j*nSer+iSer].index(cWords[1])
                    # epoch
                    xBias[j*nSer+iSer][3*k].append(float(cWords[12]))
                    if not lRange:
                        # in nanosec
                        xBias[j*nSer+iSer][3*k +
                                           1].append(float(cWords[10])/299792458*1e9)
                        xBias[j*nSer+iSer][3*k +
                                           2].append(float(cWords[11])/299792458*1e9)
                    else:
                        #Estimates in meter
                        xBias[j*nSer+iSer][3*k+1].append(float(cWords[10]))
                        xBias[j*nSer+iSer][3*k+2].append(float(cWords[11]))
    cPRN = cSat.copy()
    cPRN.sort()
    nSat = len(cSat)
    nPar = len(cPar)
    cPar.sort()

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(nSat*0.6, 4))
    x = np.arange(nSat)

    axs[0, 0].set_xlim(left=-1, right=nSat)
    axs[0, 0].set_xlabel('Satellite PRNs', fontname='Arial', fontsize=16)
    if lRange:
        axs[0, 0].set_ylim(bottom=0, top=15)
    else:
        axs[0, 0].set_ylim(bottom=0, top=0.5)
    axs[0, 0].grid(which='both', axis='y', c='darkgray', ls='--', lw=0.4)
    axs[0, 0].set_axisbelow(True)

    # Report to the terminal
    strTmp = '{: <8s} {: <15s}'.format('Ser', 'BiasPar')
    for k in range(nSat):
        strTmp = strTmp+' {: >8s}'.format(cPRN[k])
    print(strTmp)

    # the width of the bars
    w = 1/(nSer*nPar+1)
    for i in range(nSer):
        for j in range(nPar):
            Std = np.zeros(nSat)
            Std[:] = np.nan
            strTmp = '{: <8s} {: <15s}'.format(cSer[i], cPar[j])
            for k in range(nSat):
                iSat = cSat.index(cPRN[k])
                if cPar[j] not in cBias[iSat*nSer+i]:
                    # no this bias for the satellite
                    strTmp = strTmp+' {: >8.2f}'.format(9999.99)
                    continue
                iPar = cBias[iSat*nSer+i].index(cPar[j])
                if lRange:
                    # m -> cm
                    Std[k] = np.std(xBias[iSat*nSer+i][3*iPar+1])*1e2
                else:
                    # ns
                    Std[k] = np.std(xBias[iSat*nSer+i][3*iPar+1])
                strTmp = strTmp+' {: >8.2f}'.format(Std[k])
            print(strTmp)
            axs[0, 0].bar(x+(i*nPar+j-nSer*nPar/2)*w, Std,
                          w, align='edge', label=cSer[i])

    # Number of Cols for legend
    if nSer <= 6:
        nColLG = nSer
    else:
        nColLG = 6
    axs[0, 0].legend(ncol=nColLG, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                     framealpha=0.6, prop={'family': 'Arial', 'size': 14})

    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(
        cPRN, fontdict={'fontsize': 14, 'fontname': 'Arial'})
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    if lRange:
        axs[0, 0].set_ylabel('STD [cm]', fontname='Arial', fontsize=16)
    else:
        axs[0, 0].set_ylabel('STD [ns]', fontname='Arial', fontsize=16)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotISLBias4(cSer, fSerList, cSat0, lRange, yMax, OutFilePrefix, OutFileSuffix):
    '''
    Plot the comparison of differences of ISL bias parameter MEANs with respect to
    a specified solution of several other solutions.

    The first solution in the input list is taken as the reference for differencing.

     cSat0 --- Specified satellites
    lRange --- Whether use range unit for the bias
      yMax --- the min && max limit for the y-axis
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nSer = len(cSer)
    cSat = []
    cBias = []
    xBias = []
    cPar = []
    for iSer in range(nSer):
        nFile = len(fSerList[iSer])
        for i in range(nFile):
            with open(fSerList[iSer][i], mode='rt') as fOb:
                for cLine in fOb:
                    cWords = cLine.split()
                    if cWords[1][0:7] != 'ISLBIAS':
                        continue
                    if cSat0[0] != 'ALL' and cWords[5] not in cSat0:
                        continue
                    # Global satellite list
                    if cWords[5] not in cSat:
                        cSat.append(cWords[5])
                        # Bias list from each solution for this satellite
                        for j in range(nSer):
                            cBias.append([])
                            xBias.append([])
                    j = cSat.index(cWords[5])
                    # Global bias parameters list
                    if cWords[1] not in cPar:
                        cPar.append(cWords[1])
                    if cWords[1] not in cBias[j*nSer+iSer]:
                        cBias[j*nSer+iSer].append(cWords[1])
                        # Epoch
                        xBias[j*nSer+iSer].append([])
                        # Estimates
                        xBias[j*nSer+iSer].append([])
                        # Post-priori sigma
                        xBias[j*nSer+iSer].append([])
                    k = cBias[j*nSer+iSer].index(cWords[1])
                    # epoch
                    xBias[j*nSer+iSer][3*k].append(float(cWords[12]))
                    if not lRange:
                        # in nanosec
                        xBias[j*nSer+iSer][3*k +
                                           1].append(float(cWords[10])/299792458*1e9)
                        xBias[j*nSer+iSer][3*k +
                                           2].append(float(cWords[11])/299792458*1e9)
                    else:
                        #Estimates in meter
                        xBias[j*nSer+iSer][3*k+1].append(float(cWords[10]))
                        xBias[j*nSer+iSer][3*k+2].append(float(cWords[11]))
    cPRN = cSat.copy()
    cPRN.sort()
    nSat = len(cSat)
    nPar = len(cPar)
    cPar.sort()

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(nSat*0.6, 4))
    x = np.arange(nSat)

    axs[0, 0].set_xlim(left=-1, right=nSat)
    axs[0, 0].set_xlabel('Satellite PRNs', fontname='Arial', fontsize=16)

    # the mean from the first solution
    Mea0 = np.zeros((nSat, nPar))
    for j in range(nPar):
        for k in range(nSat):
            iSat = cSat.index(cPRN[k])
            if cPar[j] in cBias[iSat*nSer]:
                iPar = cBias[iSat*nSer].index(cPar[j])
                Mea0[k, j] = np.mean(xBias[iSat*nSer][3*iPar+1])
            else:
                # This bias parameters does not exist in
                # the first solution for this satellite
                Mea0[k, j] = np.nan

    # the width of the bars
    w = 1/(nSer*nPar+1)
    for i in range(nSer):
        for j in range(nPar):
            MeaDiff = np.zeros(nSat)
            if i > 0:
                for k in range(nSat):
                    iSat = cSat.index(cPRN[k])
                    strTmp = '{:<10s} - {:<10s} {: >20s} {: >3s}'.format(
                        cSer[i], cSer[0], cPar[j], cPRN[k])
                    if cPar[j] in cBias[iSat*nSer+i]:
                        iPar = cBias[iSat*nSer+i].index(cPar[j])
                        if np.isnan(Mea0[k]):
                            MeaDiff[k] = np.nan
                            strTmp = strTmp+' {: >11.3f}'.format(999.999)
                        else:
                            if lRange:
                                # m -> cm
                                MeaDiff[k] = (
                                    np.mean(xBias[iSat*nSer+i][3*iPar+1])-Mea0[k])*1e2
                            else:
                                # ns
                                MeaDiff[k] = np.mean(
                                    xBias[iSat*nSer+i][3*iPar+1])-Mea0[k]
                            strTmp = strTmp+' {: >11.3f}'.format(MeaDiff[k])
                    else:
                        MeaDiff[k] = np.nan
                    print(strTmp)
            axs[0, 0].bar(x+(i*nPar+j-nSer*nPar/2)*w, MeaDiff,
                          w, align='edge', label=cSer[i])

    # Number of Cols for legend
    if nSer <= 6:
        nColLG = nSer
    else:
        nColLG = 6
    axs[0, 0].legend(ncol=nColLG, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                     framealpha=0.6, prop={'family': 'Arial', 'size': 14})

    axs[0, 0].grid(which='major', axis='y', c='darkgray',
                   ls='--', lw=0.4, alpha=0.5)
    axs[0, 0].set_axisbelow(True)

    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(
        cPRN, fontdict={'fontsize': 14, 'fontname': 'Arial'})
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    if yMax[0] < yMax[1]:
        axs[0, 0].set_ylim(bottom=yMax[0], top=yMax[1])

    if lRange:
        axs[0, 0].set_ylabel('Mean [cm]', fontname='Arial', fontsize=16)
    else:
        axs[0, 0].set_ylabel('Mean [ns]', fontname='Arial', fontsize=16)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotISLBias5(fParList, cSat0, lRange, OutFilePrefix, OutFileSuffix):
    '''
    Plot the std of ISL bias parameters for specified satellites
    Similar to PlotISLBias1 but not plot the mean

    cSat0 --- Specified satellites
   lRange --- Whether use range (cm) instead of time (ns) as the unit
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nFile = len(fParList)
    cSat = []
    cBias = []
    xBias = []
    cPar = []
    for i in range(nFile):
        with open(fParList[i], mode='rt') as fOb:
            for cLine in fOb:
                cWords = cLine.split()
                if cWords[1][0:7] != 'ISLBIAS':
                    continue
                if cSat0[0] != 'ALL' and cWords[5] not in cSat0:
                    continue
                if cWords[5] not in cSat:
                    cSat.append(cWords[5])
                    cBias.append([])
                    xBias.append([])
                j = cSat.index(cWords[5])
                # the overall parameter set
                if cWords[1] not in cPar:
                    cPar.append(cWords[1])
                if cWords[1] not in cBias[j]:
                    cBias[j].append(cWords[1])
                    # Epoch
                    xBias[j].append([])
                    # Estimates
                    xBias[j].append([])
                    # Post-priori sigma
                    xBias[j].append([])
                k = cBias[j].index(cWords[1])
                # Epoch
                xBias[j][3*k].append(float(cWords[12]))
                # Estimates and post-priori Sigma
                if not lRange:
                    # in nanosec
                    xBias[j][3*k+1].append(float(cWords[10])/299792458*1e9)
                    xBias[j][3*k+2].append(float(cWords[11])/299792458*1e9)
                else:
                    # in meters
                    xBias[j][3*k+1].append(float(cWords[10]))
                    xBias[j][3*k+2].append(float(cWords[11]))
    nSat = len(cSat)
    cPRN = cSat.copy()
    cPRN.sort()
    nPar = len(cPar)
    cPar.sort()

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(nSat*0.6, 4))
    x = np.arange(nSat)

    axs[0, 0].set_xlim(left=-1, right=nSat)
    axs[0, 0].set_xlabel('Satellite PRNs', fontname='Arial', fontsize=16)
    if lRange:
        axs[0, 0].set_ylim(bottom=0, top=15)
    else:
        axs[0, 0].set_ylim(bottom=0, top=0.5)
    axs[0, 0].grid(which='both', axis='y', color='darkgray', linestyle='--',
                   linewidth=0.8)
    axs[0, 0].set_axisbelow(True)

    # For report to the terminal
    cStr = []
    for i in range(nSat):
        cStr.append([])
        cStr[i] = cPRN[i]

    # the width of the bars
    w = 1/(nPar+1)
    for i in range(nPar):
        Mea = np.zeros(nSat)
        Mea[:] = np.nan
        Std = np.zeros(nSat)
        Std[:] = np.nan
        for j in range(nSat):
            iSat = cSat.index(cPRN[j])
            iPar = cBias[iSat].index(cPar[i])
            Mea[j] = np.mean(xBias[iSat][3*iPar+1])
            Std[j] = np.std(xBias[iSat][3*iPar+1])
            if lRange:
                # m -> cm
                Std[j] = Std[j]*1e2
            cStr[j] = cStr[j]+' {:12.3f}'.format(Mea[j])
            cStr[j] = cStr[j]+' {:5.1f}'.format(Std[j])
            if i == nPar-1:
                print(cStr[j])
        axs[0, 0].bar(x+(i-nPar/2)*w, Std[:], w, align='edge', label=cPar[i])

    # axs[0,0].legend(ncol=nPar,loc='upper center',bbox_to_anchor=(0.5,1.0),
    #                 prop={'family':'Arial','size':14})

    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(
        cPRN, fontdict={'fontsize': 14, 'fontname': 'Arial'})
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    if lRange:
        axs[0, 0].set_ylabel('STD [cm]', fontname='Arial', fontsize=16)
    else:
        axs[0, 0].set_ylabel('STD [ns]', fontname='Arial', fontsize=16)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotISLPCO1(fParList, OutFilePrefix, OutFileSuffix):
    '''
    Plot the mean && std of ISL PCO parameters for all satellites
    '''

    nFile = len(fParList)
    # The whole satellite set in all files
    cSat = []
    for i in range(nFile):
        with open(fParList[i], mode='rt') as fOb:
            for cLine in fOb:
                cWords = cLine.split()
                if cWords[1] != 'ISLPCOX' and cWords[1] != 'ISLPCOY' and cWords[1] != 'ISLPCOZ':
                    continue
                if cWords[5] in cSat:
                    continue
                cSat.append(cWords[5])
    cSat.sort()
    nSat = len(cSat)

    PCO = np.zeros((3, nSat, nFile))
    PCO[:, :, :] = np.nan
    for i in range(nFile):
        with open(fParList[i], mode='rt') as fOb:
            for cLine in fOb:
                cWords = cLine.split()
                if cWords[1] == 'ISLPCOX':
                    iSat = cSat.index(cWords[5])
                    PCO[0, iSat, i] = float(cWords[10])
                elif cWords[1] == 'ISLPCOY':
                    iSat = cSat.index(cWords[5])
                    PCO[1, iSat, i] = float(cWords[10])
                elif cWords[1] == 'ISLPCOZ':
                    iSat = cSat.index(cWords[5])
                    PCO[2, iSat, i] = float(cWords[10])
    M = np.zeros((3, nSat))
    M[:, :] = np.nan
    S = np.zeros((3, nSat))
    S[:] = np.nan
    for i in range(nSat):
        M[0, i] = np.nanmean(PCO[0, i, :])
        S[0, i] = np.nanstd(PCO[0, i, :])
        M[1, i] = np.nanmean(PCO[1, i, :])
        S[1, i] = np.nanstd(PCO[1, i, :])
        M[2, i] = np.nanmean(PCO[2, i, :])
        S[2, i] = np.nanstd(PCO[2, i, :])
        strTmp = cSat[i]+' Mea={:6.1f} STD={:5.1f}'.format(M[0, i], S[0, i])
        strTmp = strTmp+' Mea={:6.1f} STD={:5.1f}'.format(M[1, i], S[1, i])
        strTmp = strTmp+' Mea={:6.1f} STD={:5.1f}'.format(M[2, i], S[2, i])
        print(strTmp)
    # Average STD
    strTmp = 'Ave_STD={:5.1f}'.format(np.nanmean(S[0, :]))
    strTmp = strTmp+' Ave_STD={:5.1f}'.format(np.nanmean(S[1, :]))
    strTmp = strTmp+' Ave_STD={:5.1f}'.format(np.nanmean(S[2, :]))
    print(strTmp)

    fig, axs = plt.subplots(3, 1, sharex='col', figsize=(nSat*0.8, 12))
    fig.subplots_adjust(hspace=0.1)
    x = np.arange(nSat)
    axs[0].errorbar(x, M[0, :], yerr=S[0, :], fmt='.',
                    capsize=8, markersize=18, label='X PCO')
    axs[0].set_ylabel('X PCO [mm]', fontsize=18)
    axs[0].tick_params(labelsize=16)

    axs[1].errorbar(x, M[1, :], yerr=S[1, :], fmt='.',
                    capsize=8, markersize=18, label='Y PCO')
    axs[1].set_ylabel('Y PCO [mm]', fontsize=18)
    axs[1].tick_params(labelsize=16)

    axs[2].errorbar(x, M[2, :], yerr=S[2, :], fmt='.',
                    capsize=8, markersize=18, label='Z PCO')
    axs[2].set_ylabel('Z PCO [mm]', fontsize=18)
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(cSat)
    axs[2].tick_params(labelsize=16)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotISLPCO2(fParList, cSat0, OutFilePrefix, OutFileSuffix):
    '''
    Plot the estimates series of ISL PCO parameters for specific satellites
    '''

    nFile = len(fParList)
    # The whole satellite set in all files
    cSat = []
    for i in range(nFile):
        with open(fParList[i], mode='rt') as fOb:
            for cLine in fOb:
                cWords = cLine.split()
                if cWords[1] != 'ISLPCOX' and cWords[1] != 'ISLPCOY' and cWords[1] != 'ISLPCOZ':
                    continue
                if cSat0[0] != 'ALL' and cWords[5] not in cSat0:
                    continue
                if cWords[5] in cSat:
                    continue
                cSat.append(cWords[5])
    cSat.sort()
    nSat = len(cSat)

    fig, axs = plt.subplots(nSat, 3, sharex='col',
                            squeeze=False, figsize=(3*5, nSat*3))
    fig.subplots_adjust(hspace=0.1)
    formatterx = mpl.ticker.StrMethodFormatter('{x:8.2f}')

    for i in range(nSat):
        PCO = np.zeros((nFile, 9))
        PCO[:, :] = np.nan
        for j in range(nFile):
            with open(fParList[j], mode='rt') as fOb:
                for cLine in fOb:
                    cWords = cLine.split()
                    if cWords[5] != cSat[i]:
                        continue
                    if cWords[1] == 'ISLPCOX':
                        # Epoch
                        PCO[j, 0] = float(cWords[12])
                        # Estimates
                        PCO[j, 1] = float(cWords[10])
                        # Sigma
                        PCO[j, 2] = float(cWords[11])
                    elif cWords[1] == 'ISLPCOY':
                        PCO[j, 3] = float(cWords[12])
                        PCO[j, 4] = float(cWords[10])
                        PCO[j, 5] = float(cWords[11])
                    elif cWords[1] == 'ISLPCOZ':
                        PCO[j, 6] = float(cWords[12])
                        PCO[j, 7] = float(cWords[10])
                        PCO[j, 8] = float(cWords[11])
        # Mean X PCO
        Mea = np.nanmean(PCO[:, 1])
        axs[i, 0].hlines(Mea, np.nanmin(PCO[:, 0]),
                         np.nanmax(PCO[:, 0]), colors=['r'])
        axs[i, 0].errorbar(PCO[:, 0], PCO[:, 1], yerr=PCO[:, 2],
                           fmt='.--', capsize=8, markersize=16)
        axs[i, 0].text(0.05, 0.95, cSat[i], transform=axs[i,
                       0].transAxes, ha='left', va='top')
        axs[i, 0].set_ylabel('X PCO [mm]')
        Mea = np.nanmean(PCO[:, 4])
        axs[i, 1].hlines(Mea, np.nanmin(PCO[:, 3]),
                         np.nanmax(PCO[:, 3]), colors=['r'])
        axs[i, 1].errorbar(PCO[:, 3], PCO[:, 4], yerr=PCO[:, 5],
                           fmt='.--', capsize=8, markersize=16)
        axs[i, 1].text(0.05, 0.95, cSat[i], transform=axs[i,
                       1].transAxes, ha='left', va='top')
        axs[i, 1].set_ylabel('Y PCO [mm]')
        Mea = np.nanmean(PCO[:, 7])
        axs[i, 2].hlines(Mea, np.nanmin(PCO[:, 6]),
                         np.nanmax(PCO[:, 6]), colors=['r'])
        axs[i, 2].errorbar(PCO[:, 6], PCO[:, 7], yerr=PCO[:, 8],
                           fmt='.--', capsize=8, markersize=16)
        axs[i, 2].text(0.05, 0.95, cSat[i], transform=axs[i,
                       2].transAxes, ha='left', va='top')
        axs[i, 2].set_ylabel('Z PCO [mm]')
    # axs[i,0].xaxis.set_major_formatter(formatterx)
    axs[i, 0].set_xlabel('Modified Julian Day')
    # axs[i,1].xaxis.set_major_formatter(formatterx)
    axs[i, 1].set_xlabel('Modified Julian Day')
    # axs[i,2].xaxis.set_major_formatter(formatterx)
    axs[i, 2].set_xlabel('Modified Julian Day')

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotISLPCO3(fPar, OutFilePrefix, OutFileSuffix):
    '''
    Plot the estimates and its post-prior sigma of ISL PCO estimates for all satellites
    '''
    cSat = []
    with open(fPar, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cWords[1] != 'ISLPCOX' and cWords[1] != 'ISLPCOY' and cWords[1] != 'ISLPCOZ':
                continue
            if cWords[5] in cSat:
                continue
            cSat.append(cWords[5])
    cSat.sort()
    nSat = len(cSat)

    PCO = np.zeros((nSat, 6))
    PCO[:, :] = np.nan
    with open(fPar, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cWords[1] == 'ISLPCOX':
                iSat = cSat.index(cWords[5])
                PCO[iSat, 0] = float(cWords[10])
                PCO[iSat, 1] = float(cWords[11])
            elif cWords[1] == 'ISLPCOY':
                iSat = cSat.index(cWords[5])
                PCO[iSat, 2] = float(cWords[10])
                PCO[iSat, 3] = float(cWords[11])
            elif cWords[1] == 'ISLPCOZ':
                iSat = cSat.index(cWords[5])
                PCO[iSat, 4] = float(cWords[10])
                PCO[iSat, 5] = float(cWords[11])

    fig, axs = plt.subplots(3, 1, sharex='col', figsize=(nSat*0.8, 12))
    fig.subplots_adjust(hspace=0.1)

    x = np.arange(nSat)
    # axs[0].set_ylim(bottom=-150,top=150)
    axs[0].errorbar(x, PCO[:, 0], yerr=PCO[:, 1],
                    fmt='.', capsize=8, markersize=18)
    axs[0].set_ylabel('X PCO [mm]')
    # axs[1].set_ylim(bottom=-150,top=75)
    axs[1].errorbar(x, PCO[:, 2], yerr=PCO[:, 3],
                    fmt='.', capsize=8, markersize=18)
    axs[1].set_ylabel('Y PCO [mm]')
    # axs[2].set_ylim(bottom=-500,top=300)
    axs[2].errorbar(x, PCO[:, 4], yerr=PCO[:, 5],
                    fmt='.', capsize=8, markersize=18)
    axs[2].set_ylabel('Z PCO [mm]')
    axs[2].set_xticks(x)
    axs[2].set_xticklabels(cSat)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotISLPCV1(fPar, OutFilePrefix, OutFileSuffix):
    '''
    Plot the estimated spherical harmonic PCV model (scatter)
    '''

    cSat = []
    with open(fPar, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cWords[1][0:9] != 'ISLPCVS_A' and cWords[1][0:9] != 'ISLPCVS_B':
                continue
            if cWords[5] in cSat:
                continue
            cSat.append(cWords[5])
    cSat.sort()
    nSat = len(cSat)

    Anm = np.zeros((nSat, 10, 10))
    Bnm = np.zeros((nSat, 10, 10))
    with open(fPar, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cWords[1][0:9] == 'ISLPCVS_A':
                n = int(cWords[1][9:10])
                m = int(cWords[1][10:11])
                if cWords[5] not in cSat:
                    continue
                iSat = cSat.index(cWords[5])
                Anm[iSat, n, m] = float(cWords[10])
            elif cWords[1][0:9] == 'ISLPCVS_B':
                n = int(cWords[1][9:10])
                m = int(cWords[1][10:11])
                if cWords[5] not in cSat:
                    continue
                iSat = cSat.index(cWords[5])
                Bnm[iSat, n, m] = float(cWords[10])

    fig, axs = plt.subplots(nSat, 1, squeeze=False, figsize=(12, nSat*6),
                            subplot_kw={'projection': 'polar'})
    fig.subplots_adjust(hspace=0.2)

    for i in range(nSat):
        a, z, y = PCV.GetPCVSphHarm(
            0, 360, 10, 70, 4, 4, Anm[i, :, :], Bnm[i, :, :])
        theta = []
        r = []
        v = []
        for j in range(len(a)):
            for k in range(len(z)):
                theta.append(a[j])
                r.append(z[k])
                v.append(y[j, k])
        sc = axs[i, 0].scatter(theta, r, c=v, s=2.5)
        axs[i, 0].set_rmax(90)
        axs[i, 0].set_rmin(0)
        axs[i, 0].set_rgrids((10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60),
                             labels=('10', '', '20', '', '', '', '40', '', '', '', '60'))
        axs[i, 0].grid(True, color='darkgray', linestyle='--', linewidth=0.8)
        axs[i, 0].set_axisbelow(True)
        cbar = fig.colorbar(sc, ax=axs[i, 0])
        cbar.set_label('[mm]', loc='center')
        axs[i, 0].set_title(
            cSat[i]+' PCV model (Spherical harmonic function)', va='bottom')
    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotISLPCV2(fPar, OutFilePrefix, OutFileSuffix):
    '''
    Plot the estimated spherical harmonic PCV model (pcolormesh)
    '''

    cSat = []
    with open(fPar, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cWords[1][0:9] != 'ISLPCVS_A' and cWords[1][0:9] != 'ISLPCVS_B':
                continue
            if cWords[5] in cSat:
                continue
            cSat.append(cWords[5])
    cSat.sort()
    nSat = len(cSat)

    Anm = np.zeros((nSat, 10, 10))
    Bnm = np.zeros((nSat, 10, 10))
    with open(fPar, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cWords[1][0:9] == 'ISLPCVS_A':
                n = int(cWords[1][9:10])
                m = int(cWords[1][10:11])
                if cWords[5] not in cSat:
                    continue
                iSat = cSat.index(cWords[5])
                Anm[iSat, n, m] = float(cWords[10])
            elif cWords[1][0:9] == 'ISLPCVS_B':
                n = int(cWords[1][9:10])
                m = int(cWords[1][10:11])
                if cWords[5] not in cSat:
                    continue
                iSat = cSat.index(cWords[5])
                Bnm[iSat, n, m] = float(cWords[10])

    fig, axs = plt.subplots(nSat, 1, squeeze=False, figsize=(12, nSat*6),
                            subplot_kw={'projection': 'polar'})
    fig.subplots_adjust(hspace=0.2)

    for i in range(nSat):
        a, z, C = PCV.GetPCVSphHarm(
            0, 360, 10, 70, 4, 4, Anm[i, :, :], Bnm[i, :, :])

        X, Y = np.meshgrid(a, z, indexing='ij')
        sc = axs[i, 0].pcolormesh(X, Y, C, shading='nearest')
        axs[i, 0].set_rmax(90)
        axs[i, 0].set_rmin(0)
        axs[i, 0].set_rgrids((10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60),
                             labels=('10', '', '20', '', '', '', '40', '', '', '', '60'))
        axs[i, 0].grid(True, color='darkgray', linestyle='--', linewidth=0.8)
        axs[i, 0].set_axisbelow(True)
        cbar = fig.colorbar(sc, ax=axs[i, 0])
        cbar.set_label('[mm]', loc='center')
        axs[i, 0].set_title(
            cSat[i]+' PCV model (Spherical harmonic function)', va='bottom')
    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotISLPCV3(fPar, OutFilePrefix, OutFileSuffix):
    '''
    Plot the estimates and its post-prior sigma of ISL PCV parameters for all satellites
    '''

    cSat = []
    with open(fPar, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cWords[1][0:9] != 'ISLPCVS_A' and cWords[1][0:9] != 'ISLPCVS_B':
                continue
            if cWords[5] in cSat:
                continue
            cSat.append(cWords[5])
    cSat.sort()
    nSat = len(cSat)

    Anm = np.zeros((nSat, 10, 10))
    Anm[:, :, :] = np.nan
    SigA = np.zeros((nSat, 10, 10))
    Bnm = np.zeros((nSat, 10, 10))
    Bnm[:, :, :] = np.nan
    SigB = np.zeros((nSat, 10, 10))
    with open(fPar, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cWords[1][0:9] == 'ISLPCVS_A':
                n = int(cWords[1][9:10])
                m = int(cWords[1][10:11])
                iSat = cSat.index(cWords[5])
                # mm -> cm
                Anm[iSat, n, m] = float(cWords[10])/10
                SigA[iSat, n, m] = float(cWords[11])/10
            elif cWords[1][0:9] == 'ISLPCVS_B':
                n = int(cWords[1][9:10])
                m = int(cWords[1][10:11])
                iSat = cSat.index(cWords[5])
                # mm -> cm
                Bnm[iSat, n, m] = float(cWords[10])/10
                SigB[iSat, n, m] = float(cWords[11])/10

    fig, axs = plt.subplots(nSat, 1, squeeze=False, figsize=(12, nSat*4))
    fig.subplots_adjust(hspace=0.2)

    for i in range(nSat):
        Y = []
        SigY = []
        cPar = []
        for n in range(10):
            for m in range(n+1):
                if np.isnan(Anm[i, n, m]):
                    continue
                cPar.append('A{:1d}{:1d}'.format(n, m))
                Y.append(Anm[i, n, m])
                SigY.append(SigA[i, n, m])
        for n in range(10):
            for m in range(n+1):
                if np.isnan(Bnm[i, n, m]):
                    continue
                cPar.append('B{:1d}{:1d}'.format(n, m))
                Y.append(Bnm[i, n, m])
                SigY.append(SigB[i, n, m])
        X = np.arange(len(cPar))
        axs[i, 0].errorbar(X, Y, yerr=SigY, fmt='.', capsize=8, markersize=18)
        axs[i, 0].set_ylabel('Spherical harmonic coefficients [cm]')
        axs[i, 0].set_xticks(X)
        axs[i, 0].set_xticklabels(cPar)
        axs[i, 0].text(0.05, 0.95, cSat[i], transform=axs[i,
                       0].transAxes, ha='left', va='top')

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotISLPCV4(fPar, OutFilePrefix, OutFileSuffix):
    '''
    Plot the nadir-/zenith-dependent PCV based on estimated spherical harmonic
    function model
    '''

    cSat = []
    with open(fPar, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cWords[1][0:9] != 'ISLPCVS_A' and cWords[1][0:9] != 'ISLPCVS_B':
                continue
            if cWords[5] in cSat:
                continue
            cSat.append(cWords[5])
    cSat.sort()
    nSat = len(cSat)

    Anm = np.zeros((nSat, 10, 10))
    Bnm = np.zeros((nSat, 10, 10))
    with open(fPar, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cWords[1][0:9] == 'ISLPCVS_A':
                n = int(cWords[1][9:10])
                m = int(cWords[1][10:11])
                iSat = cSat.index(cWords[5])
                Anm[iSat, n, m] = float(cWords[10])
            elif cWords[1][0:9] == 'ISLPCVS_B':
                n = int(cWords[1][9:10])
                m = int(cWords[1][10:11])
                iSat = cSat.index(cWords[5])
                Bnm[iSat, n, m] = float(cWords[10])

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 6))
    fig.subplots_adjust(hspace=0.2)

    for i in range(nSat):
        a, z, y = PCV.GetPCVSphHarm(
            0, 360, 10, 70, 4, 4, Anm[i, :, :], Bnm[i, :, :])

        axs[0, 0].plot(z, y[0, :], label=cSat[i])
        axs[0, 0].set_xlim(left=0, right=90)
        axs[0, 0].grid(True, color='darkgray', linestyle='--', linewidth=0.8)
        axs[0, 0].set_axisbelow(True)
    axs[0, 0].set_ylabel('PCV [mm]')
    axs[0, 0].set_xlabel('Nadir [deg]')
    axs[0, 0].legend(loc='upper right', framealpha=0.3)

    strTmp = OutFilePrefix+OutFileSuffix
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
    fSerList = []

    InFilePrefix = os.path.join(cWrkPre0, r'PRO_2019001_2020366_WORK/')
    # InFilePrefix=r'Y:/PRO_2019001_2020366_WORK/C6/WORK2019347_ERROR/'
    # InFilePrefix=r'Y:/PRO_2019001_2020366_WORK/I2_G_6/WORK2019???/'
    # InFilePrefix=r'Y:/PRO_2019001_2020366_WORK/I2_1/2019/PAR_POD/'
    # InFilePrefix = r'D:/Code/PROJECT/WORK2022054/'

    fParList = glob.glob(InFilePrefix+'J600/2019/PAR_POD/par_20193??')
    # fParList = glob.glob(InFilePrefix+'par_2022054')

    # fCtrList = glob.glob(InFilePrefix+'WORK2019???/cf_net')

    OutFilePrefix = os.path.join(cDskPre0, r'PRO_2019001_2020366/J642/HD/')
    # OutFilePrefix = os.path.join(
    #     cWrkPre0, r'PRO_2019001_2020366_WORK/I121/WORK2019335/')
    # OutFilePrefix=r'Z:/PRO_2019001_2020366/C63/PAR/'
    # OutFilePrefix = r'D:/Code/PROJECT/WORK2022054/'

    # OutFileSuffix='BiasSum_IBB_BDS-3_G'
    # PlotGNSBias00(fParList,'IBB_BDS-3_G',OutFilePrefix,OutFileSuffix)
    # PlotGNSBias01(fParList,'IBB_BDS-3_BDS-2',OutFilePrefix,OutFileSuffix)

    # OutFileSuffix='IBB_BDS-3_BDS-2'
    # PlotGNSBias50(fParList,'IBB_BDS-3_BDS-2',['E','DJIG','RGDG'],2,fCtrList,3,OutFilePrefix,OutFileSuffix)

    # fBias=os.path.join(cDskPre0,r'PRO_2019001_2020366/C6/BIAS/RecBias_Align')
    # OutFileSuffix='IBB_BDS-3_BDS-2_Align'
    # PlotGNSBias51(fBias,'IBB_BDS-3_BDS-2',['E','DJIG','RGDG'],2,fCtrList,3,OutFilePrefix,OutFileSuffix)

    fPar = os.path.join(
        cWrkPre0, r'PRO_2019001_2020366_WORK/I121/WORK2019335/par_2019335')
    fCtr = os.path.join(
        cWrkPre0, r'PRO_2019001_2020366_WORK/C6/WORK2019345/cf_net')

    # OutFileSuffix='BiasEst_2019345'
    # PlotGNSBias11(fPar,'IBB_BDS-3_BDS-2','',fCtr,3,False,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='BiasDif_2019345'
    # PlotGNSBias12(fPar,['IBB_BDS-2_G','IBB_BDS-3_G'],'',fCtr,3,False,OutFilePrefix,OutFileSuffix)

    # cSer.append('GloA')
    # fParList = glob.glob(InFilePrefix+'C02/2019/PAR_POD/par_20193??')
    # fSerList.append(fParList)

    # cSer.append('GloA+ISL')
    # fParList = glob.glob(InFilePrefix+'D660/2019/PAR_POD/par_20193??')
    # fSerList.append(fParList)

    # OutFileSuffix = 'GCC_STD_Comp'
    # PlotGCCPar1(fParList,OutFilePrefix,OutFileSuffix)
    # PlotGCCPar20(cSer,fSerList,1,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix = 'GCC_AvgFormalError_Comp'
    # PlotGCCPar21(cSer, fSerList, 1, OutFilePrefix, OutFileSuffix)
    # OutFileSuffix = 'GCC_STD_Comp'
    # PlotGCCPar21(cSer, fSerList, 0, OutFilePrefix, OutFileSuffix)
    # OutFileSuffix='EOP_FormalError_Comp'
    # PlotEOPPar40(cSer,fSerList,True,1,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix = 'EOP_AvgFormalError_Comp'
    # PlotEOPPar41(cSer, fSerList, True, 1, OutFilePrefix, OutFileSuffix)

    # cSer.append('None')
    # fParList = glob.glob(InFilePrefix+'J600/2019/PAR_POD/par_20193??')
    # fSerList.append(fParList)

    # cSer.append('Model 1')
    # fParList = glob.glob(InFilePrefix+'J601/2019/PAR_POD/par_20193??')
    # fSerList.append(fParList)

    # cSer.append('Model 2')
    # fParList = glob.glob(InFilePrefix+'J602/2019/PAR_POD/par_20193??')
    # fSerList.append(fParList)

    # cSer.append('Model 3')
    # fParList = glob.glob(InFilePrefix+'J603/2019/PAR_POD/par_20193??')
    # fSerList.append(fParList)

    cSer.append('1 cm')
    fParList = glob.glob(InFilePrefix+'J640/2019/PAR_POD/par_20193??')
    fSerList.append(fParList)

    cSer.append('2 cm')
    fParList = glob.glob(InFilePrefix+'J641/2019/PAR_POD/par_20193??')
    fSerList.append(fParList)

    cSer.append('4 cm')
    fParList = glob.glob(InFilePrefix+'J642/2019/PAR_POD/par_20193??')
    fSerList.append(fParList)

    cSer.append('8 cm')
    fParList = glob.glob(InFilePrefix+'J646/2019/PAR_POD/par_20193??')
    fSerList.append(fParList)

    cSer.append('12 cm')
    fParList = glob.glob(InFilePrefix+'J647/2019/PAR_POD/par_20193??')
    fSerList.append(fParList)

    cSer.append('15 cm')
    fParList = glob.glob(InFilePrefix+'J648/2019/PAR_POD/par_20193??')
    fSerList.append(fParList)
    #
    cSer.append('18 cm')
    fParList = glob.glob(InFilePrefix+'J651/2019/PAR_POD/par_20193??')
    fSerList.append(fParList)

    cSer.append('20 cm')
    fParList = glob.glob(InFilePrefix+'J653/2019/PAR_POD/par_20193??')
    fSerList.append(fParList)

    # cSer=[]; fPathSer=[]
    # cSer.append('C1'); fPathSer.append(r'Y:/PRO_2019001_2020366_WORK/C1/2019/PAR_POD/')
    # cSer.append('C2'); fPathSer.append(r'Y:/PRO_2019001_2020366_WORK/C2/2019/PAR_POD/')

    # OutFileSuffix='SatObsNum'
    # PlotOrbPar0(cSer,fPathSer,58818,31,OutFilePrefix,OutFileSuffix)

    # OutFileSuffix = 'Bias_2019335_2019365_std'
    # PlotISLBias1(fParList, ['ALL'], True, False, OutFilePrefix, OutFileSuffix)
    # OutFileSuffix = 'ISLBias_std_2019335_2019365_Comp'
    # PlotISLBias3(cSer, fSerList, ['ALL'], False, OutFilePrefix, OutFileSuffix)
    OutFileSuffix = 'Bias_2019335_2019365_C33'
    # PlotISLBias2(fParList, ['C33'], ['ISLBIAS'], True,
    #              False, OutFilePrefix, OutFileSuffix)
    OutFileSuffix = 'Bias_mea_2019335_2019365_Comp'
    PlotISLBias4(cSer, fSerList, ['ALL'], False,
                 [-0.5, 0.2], OutFilePrefix, OutFileSuffix)
    # OutFileSuffix='ISLBias_std_2019335_2019365'
    # PlotISLBias5(fParList,['ALL'],True,OutFilePrefix,OutFileSuffix)

    # cSer.append('S2')
    # fSerList.append(fParList)

    # fPar=r'Y:/PRO_2019001_2020366_WORK/I2_4/WORK2019335/par_2019335'
    # OutFileSuffix='PCO_2019365_1.png'
    # PlotGNSBias3(fPar,OutFilePrefix,OutFileSuffix)
    # PlotGNSPCO1(fPar,OutFilePrefix,OutFileSuffix)
    # cStaAnt=[['BSHM','CUT0','PTGG','METG','RGDG','KITG','GANP','NKLG'],
    #          ['BOR1','NIUM','HARB','CPVG','JOZ2','PNGM','TLSE','KIRI','MCHL','TONG','SEYG','MRO1','TLSG',
    #           'CHPG','DJIG','ZIM2','ASCG','PERT','JFNG','OWMG','KARR','GAMB','FTNA','AREG','MAYG','ZIM3'],
    #          ['ABMF','UNB3','LMMF','BRST'],
    #          ['BOAV','SALU','POVE','POAL','SAVO','BELE','UFPR','TOPL'],
    #          ['USN7','USN8'],
    #          ['STHL','GODS','GODN'],
    #          ['POL2','CHPI'],
    #          ['PADO','KIRU','GOP6','MRC1'],
    #          ['NNOR','CEBR','MIZU','REDU','JOZE','KOUR','OUS2'],
    #          ['YEL2','FAA1','MAL2','STJ3','MAS1','ROAG','MGUE'],
    #          ['LEIJ','WARN','KRGG','GAMG','PTBB','GRAZ','OHI3','HUEG'],
    #          ['TOW2','HERS','STR2'],
    #          ['KIR8','FFMJ','MAR7','KAT1','MAO0','THTG','KOUG','WTZZ'],
    #          ['KIR0','SPT0'],
    #          ['SGPO','WUH2','POTS','URUM','SGOC','WIND','ENAO','NYA2','SUTM','LPGS','ULAB'],
    #          ['GUAM','NLIB','GCGO','MET3','ARHT','CRO1','SOD3'],
    #          ['DARW','BOGT','CKIS','SAMO','CUSV','POHN','VACS','SOLO'],
    #          ['SUTH','KOKB'],['QUIN','KERG'],['IISC','DGAR'],['PARK','YKRO','STR1'],
    #          ['MAR6','VIS0'],['HOB2','TID1','JPLM','CEDU','ZAMB'],['USUD','GODE']
    #         ]
    # OutFileSuffix='PCO_2019365_2.png'
    # PlotGNSPCO2(fPar,cStaAnt,OutFilePrefix,OutFileSuffix)

    # fPar=r'D:/Code/PROJECT/WORK2019336_ERROR/par_2019336'
    # fPar=r'Y:/PRO_2019001_2020366_WORK/I0_1/WORK2019346/par_2019346'
    OutFileSuffix = 'Clk_Sig_2019335'
    # PlotClkPar2(fPar, ['C20', 'C21', 'C22', 'C23', 'C24',
    #             'C25', 'C26'], OutFilePrefix, OutFileSuffix)

    OutFileSuffix = 'ClkObs_2019335'
    # PlotClkPar3(fParList, 300, 2, ['C20', 'C21', 'C23', 'C24', 'C27',
    #             'C28'], OutFilePrefix, OutFileSuffix)

    # cSta=['BSHM','CUT0','PTGG','METG','RGDG','KITG','GANP','NKLG']
    # cSta=['BOR1','NIUM','HARB','CPVG','JOZ2','PNGM','TLSE','KIRI','MCHL','TONG','SEYG','MRO1','TLSG',
    #       'CHPG','DJIG','ZIM2','ASCG','PERT','JFNG','OWMG','KARR','GAMB','FTNA','AREG','MAYG','ZIM3']
    # cSta=['BOAV','SALU','POVE','POAL','SAVO','BELE','UFPR','TOPL']
    # OutFileSuffix='GISB_4_Ant3.pdf'
    # PlotGNSBias4(fParList,['ISB','POS'],cSta,True,OutFilePrefix,OutFileSuffix)

    # fPar=r'D:/Code/PROJECT/WORK2019335_ERROR/par_2019335'
    # OutFileSuffix='EOPPar1.png'
    # PlotEOPPar1(fPar,OutFilePrefix,OutFileSuffix)

    # OutFileSuffix='EOPPar_Sig.png'
    # PlotEOPPar2(fParList,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='EOPPar_Sig3.png'
    # PlotEOPPar3(fParList,OutFilePrefix,OutFileSuffix)
