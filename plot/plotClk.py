#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Plot clock info from RINEX clock products
'''
__author__ = 'hanbing'

# Standard library imports
import os
import sys
import os.path
import glob
import math
from astropy.stats.sigma_clipping import sigma_clip

# Related third party imports
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import warnings

import allantools
from astropy import stats
import astropy.modeling.models as AsMod
import astropy.modeling.fitting as AsFit

# Local application/library specific imports
from PySRC.miscellaneous import GNSSTime
from PySRC.plot import plotGeoAng

from PySRC.plot import plotISL as ISL


def ReadRnxClk(fClkList, ClkList, ReSamp):
    '''
    Read RINEX clock file and return the clock data
    as numpy.ndarray. Clock name and epoch list are also
    returned.

    Input :
      fClkList --- RINEX clock files list
       ClkList --- Specified clock list to read
        ReSamp --- if positive, re-sampling the clock only
                   keep points at the integer times of the value

    Return:
        EpoStr --- List of epoch string
       ClkName --- List of clock name
           Clk --- numpy.ndarray of clock data in sec
                   # 0, rMJD epoch
                   # 1, clock offset, in sec
                   # 2, clock rate, in sec/sec
                   # 3, clock rate drift, in sec/sec^2
    '''
    # Clock name and Epoch list
    ClkName = []
    EpoStr = []
    Clk = []
    for i in range(len(fClkList)):
        with open(fClkList[i], mode='rt') as fOb:
            while True:
                cLine = fOb.readline()
                if not cLine:
                    # EOF
                    break
                if cLine[0:3] != 'AR ' and cLine[0:3] != 'AS ':
                    continue
                cWords = cLine.split()
                if ClkList[0][1:3] == 'XX':
                    # Specified GNSS system
                    if cLine[0:3] != 'AS ' or cWords[1][0:1] != ClkList[0][0:1]:
                        continue
                elif ClkList[0] == 'EXCL':
                    # Specified to be excluded clocks
                    if cWords[1] in ClkList[1:]:
                        continue
                elif ClkList[0] == 'ALLS':
                    # Specified all satellites
                    if cLine[0:3] != 'AS ':
                        continue
                elif ClkList[0] == 'ALLR':
                    # Specified all stations
                    if cLine[0:3] != 'AR ':
                        continue
                elif ClkList[0] != 'ALL':
                    # Specified clock list
                    if cWords[1] not in ClkList:
                        continue
                MJD = GNSSTime.dom2mjd(
                    int(cWords[2]), int(cWords[3]), int(cWords[4]))
                SOD = int(cWords[5])*3600 + int(cWords[6]) * \
                    60 + float(cWords[7])
                if ReSamp > 0 and math.fabs(math.fmod(SOD, ReSamp)) > 1:
                    continue
                if cWords[1] not in ClkName:
                    ClkName.append(cWords[1])
                    # Epoch and clock offset, clock rate, drift
                    Clk.append([])
                    Clk.append([])
                    Clk.append([])
                    Clk.append([])
                    # str epoch
                    EpoStr.append([])
                j = ClkName.index(cWords[1])
                EpoStr[j].append(cLine[8:34])
                # epoch in float MJD
                Clk[j*4].append(MJD + SOD/86400)
                # clock offset, in sec
                Clk[j*4+1].append(float(cWords[9]))
                nterm = int(cWords[8])
                if nterm > 2:
                    cLine = fOb.readline()
                    cWords = cLine.split()
                    # clock rate
                    Clk[j*4+2].append(float(cWords[0]))
                    if nterm > 4:
                        # clock drift
                        Clk[j*4+3].append(float(cWords[2]))
                    else:
                        Clk[j*4+3].append(np.nan)
                else:
                    Clk[j*4+2].append(np.nan)
                    Clk[j*4+3].append(np.nan)

    nClk = len(ClkName)
    if nClk == 0:
        print('No clock found')
        return ClkName, EpoStr, Clk
    else:
        # Convert to np array manually because of inhomogeneous shape
        nCol = nClk*4
        nRow = 0
        for i in range(nClk):
            if nRow >= len(Clk[i*4]):
                continue
            # The most epoch number
            nRow = len(Clk[i*4])
        x = np.zeros((nRow, nCol))
        x[:, :] = np.nan
        for i in range(nClk):
            for j in range(len(Clk[i*4])):
                x[j, i*4] = Clk[i*4][j]
                x[j, i*4+1] = Clk[i*4+1][j]
                x[j, i*4+2] = Clk[i*4+2][j]
                x[j, i*4+3] = Clk[i*4+3][j]
        return ClkName, EpoStr, x


def DiffRnxClk0(fClkList1, fClkList2, ClkList):
    '''
    Cal the diff of clock offset from two sets of clock files by direct differencing

 Return:
      dClk --- clock offset && rate diff in nanosec

    '''

    ClkName = []
    dClk = []

    ClkName1, EpoStr1, Clk1 = ReadRnxClk(fClkList1, ClkList, 0)
    ClkName2, EpoStr2, Clk2 = ReadRnxClk(fClkList2, ClkList, 0)
    # Get the common clock set
    for strTmp in ClkName1:
        if strTmp not in ClkName2:
            continue
        ClkName.append(strTmp)
    nClk = len(ClkName)
    ClkName.sort()
    if nClk == 0:
        print('No common clock found!')
        return ClkName, dClk

    for i in range(nClk):
        # Epoch, clk diff, rate diff
        dClk.append([])
        dClk.append([])
        dClk.append([])
        dClk.append([])
        i1 = ClkName1.index(ClkName[i])
        i2 = ClkName2.index(ClkName[i])
        # Sort them first!
        ind1 = np.argsort(Clk1[:, i1*4])
        ind2 = np.argsort(Clk2[:, i2*4])
        for j in range(ind1.size):
            k = -1
            for l in range(ind2.size):
                if (Clk1[ind1[j], i1*4]-Clk2[ind2[l], i2*4])*86400 > 0.5:
                    continue
                elif (Clk1[ind1[j], i1*4]-Clk2[ind2[l], i2*4])*86400 < -0.5:
                    break
                else:
                    k = l
                    break
            if k < 0:
                # Not found
                continue
            # Epoch
            dClk[i*4].append(Clk1[ind1[j], i1*4])
            # clock offset, sec -> nansec
            dClk[i*4+1].append((Clk1[ind1[j], i1*4+1] -
                               Clk2[ind2[k], i2*4+1])*1e9)
            # clock rate, in nansec/sec
            if np.isnan(Clk1[ind1[j], i1*4+2]) or np.isnan(Clk2[ind2[k], i2*4+2]):
                dClk[i*4+2].append(np.nan)
            else:
                dClk[i*4+2].append((Clk1[ind1[j], i1*4+2] -
                                   Clk2[ind2[k], i2*4+2])*1e9)
    return ClkName, dClk


def DiffRnxClk1(fClkList1, fClkList2, ClkRefList, ClkList):
    '''
    Cal the diff of clock offsets from two sets of clock files by
    adopting the double-difference method.

    ClkRefList --- the reference clock list, take the first available one
 Return:
          dClk --- clock diff in nanosec
    '''

    ClkName = []
    dClk = []
    # Whether find a reference clock in both clock files
    lFound = False

    for i in range(len(ClkRefList)):
        ClkRef = ClkRefList[i]
        # Read the reference clock from the 1st clock set
        ClkName10, EpoStr10, Clk10 = ReadRnxClk(fClkList1, [ClkRef], 0)
        if len(ClkName10) != 1 or ClkName10[0] != ClkRef:
            print('Ref clk not found in the 1st set, '+ClkRef)
            continue
        # Read the reference clock from the 2nd clock set
        ClkName20, EpoStr20, Clk20 = ReadRnxClk(fClkList2, [ClkRef], 0)
        if len(ClkName20) != 1 or ClkName20[0] != ClkRef:
            print('Ref clk not found in the 2nd set, '+ClkRef)
            continue
        print('Found ref clk in both sets, '+ClkRef)
        lFound = True
        break
    if not lFound:
        print('Failed to find a ref clk')
        return '', ClkName, dClk

    ClkName1, EpoStr1, Clk1 = ReadRnxClk(fClkList1, ClkList, 0)
    ClkName2, EpoStr2, Clk2 = ReadRnxClk(fClkList2, ClkList, 0)
    for i2 in range(len(ClkName2)):
        # Skip the reference clock
        if ClkName2[i2] == ClkRef:
            continue
        # Not common clock
        if ClkName2[i2] not in ClkName1:
            continue
        i1 = ClkName1.index(ClkName2[i2])
        ClkName.append(ClkName2[i2])
        iClk = len(ClkName)-1
        # Epoch, clk diff
        dClk.append([])
        dClk.append([])

        # self-diff for the 1st clock set
        nEpo1 = len(Clk1[:, i1*4])
        nEpo10 = len(Clk10[:, 0])
        XClk1 = [[], []]
        for j in range(nEpo1):
            t1 = Clk1[j, i1*4]
            k = -1
            for l in range(nEpo10):
                t2 = Clk10[l, 0]
                dt = (t1-t2)*86400
                if dt > 0.5:
                    continue
                elif dt < -0.5:
                    break
                else:
                    k = l
                    break
            if k < 0:
                # Not found
                continue
            # epoch
            XClk1[0].append(Clk1[j, i1*4])
            # offset diff wrt the ref clock
            XClk1[1].append(Clk1[j, i1*4+1]-Clk10[k, 1])

        # self-diff for the 2nd clock set
        nEpo2 = len(Clk2[:, i2*4])
        nEpo20 = len(Clk20[:, 0])
        XClk2 = [[], []]
        for j in range(nEpo2):
            k = -1
            for l in range(nEpo20):
                if (Clk2[j, i2*4]-Clk20[l, 0])*86400 > 0.5:
                    continue
                elif (Clk2[j, i2*4]-Clk20[l, 0])*86400 < -0.5:
                    break
                else:
                    k = l
                    break
            if k < 0:
                # Not found
                continue
            # epoch
            XClk2[0].append(Clk2[j, i2*4])
            # offset diff wrt the ref clock
            XClk2[1].append(Clk2[j, i2*4+1]-Clk20[k, 1])

        # mutual-diff between those two clock sets
        nEpo1 = len(XClk1[0])
        nEpo2 = len(XClk2[0])
        for k in range(nEpo1):
            l = -1
            for m in range(nEpo2):
                if (XClk1[0][k]-XClk2[0][m])*86400 > 0.5:
                    continue
                elif (XClk1[0][k]-XClk2[0][m])*86400 < -0.5:
                    break
                else:
                    l = m
                    break
            if l < 0:
                # Not found
                continue
            # epoch
            dClk[iClk*2].append(XClk1[0][k])
            # sec -> nanosec
            dClk[iClk*2+1].append((XClk1[1][k]-XClk2[1][l])*1e9)
    return ClkRef, ClkName, dClk


def PlotClk10(fClkList, ClkList, OutFilePrefix, OutFileSuffix):
    '''
    Plot the clock offset, rate and rate drift series for specified clocks
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    ClkName, EpoStr, Clk = ReadRnxClk(fClkList, ClkList, 0)
    nClk = len(ClkName)
    ClkName1 = ClkName.copy()
    ClkName1.sort()
    if nClk == 0:
        sys.exit('No clk found!')

    fig, axs = plt.subplots(nClk, 3, sharex='col',
                            squeeze=False, figsize=(24, nClk*1.5))
    # fig.subplots_adjust(hspace=0.1)
    formatterx = mpl.ticker.StrMethodFormatter('{x:7.1f}')

    cAxs = ['Offset', 'Rate', 'Drift']
    yLab = ['[ns]', '[ps/s]', r'[ps/$s^2$]']
    for i in range(nClk):
        j = ClkName.index(ClkName1[i])
        for k in range(3):
            if k == 1:
                axs[i, k].set_ylim(bottom=-50, top=50)
            # sec -> pic sec
            axs[i, k].plot(Clk[:, j*4], Clk[:, j*4+1+k]*1e12, '.r', ms=2)
            axs[i, k].grid(which='major', axis='y',
                           color='darkgray', ls='--', lw=0.4)
            axs[i, k].set_axisbelow(True)
            axs[i, k].set_ylabel(yLab[k], fontname='Arial', fontsize=16)
            axs[i, k].ticklabel_format(
                axis='y', style='sci', useOffset=False, useMathText=True)
            for tl in axs[i, k].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            if k != 0:
                if np.count_nonzero(~np.isnan(Clk[:, j*4+1+k])) > 0:
                    Mea = np.nanmean(Clk[:, j*4+1+k]*1e12)
                    Sig = np.nanstd(Clk[:, j*4+1+k]*1e12)
                else:
                    Mea = 0
                    Sig = 0
                strTmp = '{:>7.4E}+/-{:>7.5E}'.format(Mea, Sig)
                axs[i, k].text(0.5, 1.0, strTmp, transform=axs[i, k].transAxes, ha='center', va='top',
                               fontdict={'fontsize': 12, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkred'})

            axs[i, k].text(0.02, 0.98, ClkName[j], transform=axs[i, k].transAxes, ha='left', va='top',
                           fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
            axs[i, k].text(0.98, 0.98, cAxs[k], transform=axs[i, k].transAxes, ha='right', va='top',
                           fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
    for k in range(3):
        axs[i, k].set_xlabel('Modified Julian Day',
                             fontname='Arial', fontsize=16)
        axs[i, k].xaxis.set_major_formatter(formatterx)
        for tl in axs[i, k].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotClk11(fClkList, ClkList, OutFilePrefix, OutFileSuffix):
    '''
    Plot the 1) estimated clock rate series as well as
             2) the clock rates calculated from the estimated
                clock offsets.
             3) their differences
             4) clock offset discontinuities at the piece boundary
    Specified clocks are plotted one by one.

    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    ClkName, EpoStr, Clk = ReadRnxClk(fClkList, ClkList, 0)
    nClk = len(ClkName)
    ClkName1 = ClkName.copy()
    ClkName1.sort()
    if nClk == 0:
        sys.exit('No clk found!')

    fig, axs = plt.subplots(nClk, 3, sharex='col',
                            squeeze=False, figsize=(24, nClk*3))
    # fig.subplots_adjust(hspace=0.1)
    formatterx = mpl.ticker.StrMethodFormatter('{x:7.1f}')

    for i in range(nClk):
        j = ClkName.index(ClkName1[i])
        nEpo = Clk[:, j*4].size
        t = np.zeros(nEpo-1)
        # Clock rates calculated from the estimated offsets
        VClk1 = np.zeros(nEpo-1)
        # Clock rate differences between the calculated and estimated
        dVClk = np.zeros(nEpo-1)
        # Clock offset differences at the piece boundary
        dClk = np.zeros(nEpo-1)
        for k in range(nEpo-1):
            t[k] = Clk[k, j*4]
            if np.isnan(Clk[k, j*4+1]):
                # Estimated clock offset not available for this epoch
                VClk1[k] = np.nan
                dVClk[k] = np.nan
                dClk[k] = np.nan
            elif np.isnan(Clk[k+1, j*4+1]):
                # Estimated clock offset not available for this next epoch
                VClk1[k] = np.nan
                dVClk[k] = np.nan
                dClk[k] = np.nan
            else:
                # Day -> Sec
                dt = (Clk[k+1, j*4]-Clk[k, j*4])*86400.0
                # Calculated clock rate, sec/s -> ps/sec
                VClk1[k] = (Clk[k+1, j*4+1]-Clk[k, j*4+1])/dt*1e12
                if np.isnan(Clk[k, j*4+2]):
                    # Estimated clock rate not available for this epoch
                    dVClk[k] = np.nan
                    dClk[k] = np.nan
                else:
                    # Clock rate difference, sec/sec -> ps/sec
                    dVClk[k] = VClk1[k] - Clk[k, j*4+2]*1e12
                    # Clock offset discontinuity, ps -> m
                    dClk[k] = dVClk[k] * dt * 1e-12 * 299792458

        axs[i, 0].plot(t, VClk1, '.b', ms=2)
        axs[i, 0].yaxis.set_major_formatter('{x: >4.0f}')
        strTmp = 'Cal {:>7.1f}+/-{:>7.2f}'.format(
            np.nanmean(VClk1), np.nanstd(VClk1))
        axs[i, 0].text(0.98, 0.98, strTmp, transform=axs[i, 0].transAxes, ha='right', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkblue'})
        axs[i, 0].set_ylabel('Rate [ps/s]', fontname='Arial', fontsize=16)

        if np.count_nonzero(~np.isnan(dVClk)) > 0:
            # the estimated clock rates
            axs[i, 0].plot(Clk[:, 4*j], Clk[:, 4*j+2]*1e12, '.g', ms=2)
            strTmp = 'Est {:>7.1f}+/-{:>7.2f}'.format(np.nanmean(
                Clk[:, 4*j+2]*1e12), np.nanstd(Clk[:, 4*j+2]*1e12))
            axs[i, 0].text(0.98, 0.02, strTmp, transform=axs[i, 0].transAxes, ha='right', va='bottom',
                           fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkgreen'})
            # Rate difference between that estimated and that calculated
            axs[i, 1].plot(t, dVClk, '.r', ms=2)
            axs[i, 1].yaxis.set_major_formatter('{x: >4.0f}')
            strTmp = '{:>7.1f}+/-{:>7.2f}'.format(
                np.nanmean(dVClk), np.nanstd(dVClk))
            axs[i, 1].text(0.98, 0.98, strTmp, transform=axs[i, 1].transAxes, ha='right', va='top',
                           fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkred'})
            axs[i, 1].set_ylabel(
                'Rate dis [ps/s]', fontname='Arial', fontsize=16)

            axs[i, 2].plot(t, dClk, '.r', ms=2)
            axs[i, 2].yaxis.set_major_formatter('{x: >4.0f}')
            strTmp = '{:>7.1f}+/-{:>7.2f}'.format(
                np.nanmean(dClk), np.nanstd(dClk))
            axs[i, 2].text(0.98, 0.98, strTmp, transform=axs[i, 2].transAxes, ha='right', va='top',
                           fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkred'})
            axs[i, 2].set_ylabel('Bias dis [m]', fontname='Arial', fontsize=16)

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

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotClk12(fClkList, ClkList, OutFilePrefix, OutFileSuffix):
    '''
    Similar to PlotClk10, but plot for each file in the specified file list
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    strTmp = os.path.join(OutFilePrefix, OutFileSuffix+'.pdf')
    with PdfPages(strTmp) as pdf:
        for fClk in fClkList:
            ClkName, EpoStr, Clk = ReadRnxClk([fClk], ClkList, 0)
            nClk = len(ClkName)
            ClkName1 = ClkName.copy()
            ClkName1.sort()
            if nClk == 0:
                print('No clk found in '+fClk)
                continue
            fig, axs = plt.subplots(
                nClk, 3, sharex='col', squeeze=False, figsize=(24, nClk*1.5))
            # fig.subplots_adjust(hspace=0.1)
            formatterx = mpl.ticker.StrMethodFormatter('{x:7.1f}')
            axs[0, 1].text(0.50, 1.00, fClk, transform=axs[0, 1].transAxes, ha='center', va='bottom',
                           fontdict={'fontsize': 10, 'fontname': 'Arial', 'fontweight': 'bold'})

            yLab = ['Offset', 'Rate', 'Drift']
            for i in range(nClk):
                j = ClkName.index(ClkName1[i])
                for k in range(3):
                    # sec -> nano sec
                    axs[i, k].plot(Clk[:, j*4], Clk[:, j*4+1+k]
                                   * 1e9, '.r', ms=2)
                    axs[i, k].grid(which='major', axis='y',
                                   color='darkgray', ls='--', lw=0.4)
                    axs[i, k].set_axisbelow(True)
                    axs[i, k].set_ylabel('[ns]', fontname='Arial', fontsize=16)
                    axs[i, k].ticklabel_format(
                        axis='y', style='sci', useOffset=False, useMathText=True)
                    for tl in axs[i, k].get_yticklabels():
                        tl.set_fontname('Arial')
                        tl.set_fontsize(14)
                    if k != 0:
                        if np.count_nonzero(~np.isnan(Clk[:, j*4+1+k])) > 0:
                            Mea = np.nanmean(Clk[:, j*4+1+k]*1e9)
                            Sig = np.nanstd(Clk[:, j*4+1+k]*1e9)
                        else:
                            Mea = 0
                            Sig = 0
                        strTmp = '{:>7.4E}+/-{:>7.5E}'.format(Mea, Sig)
                        axs[i, k].text(0.5, 1.0, strTmp, transform=axs[i, k].transAxes, ha='center', va='top',
                                       fontdict={'fontsize': 12, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkred'})

                    axs[i, k].text(0.02, 0.98, ClkName[j], transform=axs[i, k].transAxes, ha='left', va='top',
                                   fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
                    axs[i, k].text(0.98, 0.98, yLab[k], transform=axs[i, k].transAxes, ha='right', va='top',
                                   fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
            for k in range(3):
                axs[i, k].set_xlabel('Modified Julian Day',
                                     fontname='Arial', fontsize=16)
                axs[i, k].xaxis.set_major_formatter(formatterx)
                for tl in axs[i, k].get_xticklabels():
                    tl.set_fontname('Arial')
                    tl.set_fontsize(14)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def PlotClk20(fClkList, ClkList, iPlot, OutFilePrefix, OutFileSuffix):
    '''
    Plot boxplot for clock rate and/or rate drift for specified clocks

    iPlot --- Which to plot
              # 0, rate
              # 1, rate drift
              # 2, both
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    ClkName, EpoStr, Clk = ReadRnxClk(fClkList, ClkList, 0)
    nClk = len(ClkName)
    ClkName1 = ClkName.copy()
    ClkName1.sort()
    if nClk == 0:
        sys.exit('No clk found!')

    yLab = ['Clock rate [ps/s]', r'Clock rate drift [ps/$s^2$]']
    if iPlot != 2:
        if iPlot == 0:
            print('{: <8s} {: >5s} {: >8s}'.format('Clk', 'EpoN', 'Rate'))
        else:
            print('{: <8s} {: >5s} {: >8s}'.format('Clk', 'EpoN', 'Drift'))
        # Only rate or rate drift
        y = []
        for i in range(nClk):
            j = ClkName.index(ClkName1[i])
            y.append([])
            for k in range(Clk[:, j*4].size):
                if np.isnan(Clk[k, j*4+2+iPlot]):
                    continue
                # s -> ps
                y[i].append(Clk[k, j*4+2+iPlot]*1e12)
            if len(y[i]) > 0:
                strTmp = '{: <8s} {:>5d} {:>8.5e}'.format(
                    ClkName1[i], len(y[i]), np.median(y[i]))
            else:
                strTmp = '{: <8s} {:>5d} {:>8.5e}'.format(ClkName1[i], 0, 0)
            print(strTmp)

        fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(18, nClk*0.6))

        axs[0, 0].boxplot(y, notch=True, vert=False,
                          flierprops={'marker': 'o', 'ms': 4, 'mec': 'r', 'fillstyle': 'none'})
        axs[0, 0].set_ylim(bottom=0, top=nClk+1)
        axs[0, 0].set_yticks(np.arange(1, nClk+1))
        axs[0, 0].set_yticklabels(
            ClkName1, fontdict={'fontsize': 14, 'fontname': 'Arial'})

        axs[0, 0].grid(which='major', axis='y', color='darkgray',
                       ls='--', lw=0.4)
        axs[0, 0].grid(which='both', axis='x', color='darkgray',
                       ls='--', lw=0.4)
        axs[0, 0].set_axisbelow(True)

        axs[0, 0].set_xlabel(yLab[iPlot], fontname='Arial', fontsize=16)
        for tl in axs[0, 0].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
    else:
        # Both rate and rate drift
        fig, axs = plt.subplots(2, 1, squeeze=False,
                                figsize=(18, nClk*1.2+0.5))
        fig.subplots_adjust(hspace=0.1)
        for i in range(2):
            y = []
            for j in range(nClk):
                k = ClkName.index(ClkName1[j])
                y.append([])
                for l in range(Clk[:, k*4].size):
                    if np.isnan(Clk[l, k*4+2+i]):
                        continue
                    # s -> ps
                    y[j].append(Clk[l, k*4+2+i]*1e12)
            axs[i, 0].boxplot(y, notch=True, vert=False,
                              flierprops={'marker': 'o', 'ms': 4, 'mec': 'r', 'fillstyle': 'none'})
            axs[i, 0].set_ylim(bottom=0, top=nClk+1)
            axs[i, 0].set_yticks(np.arange(1, nClk+1))
            axs[i, 0].set_yticklabels(
                ClkName1, fontdict={'fontsize': 14, 'fontname': 'Arial'})

            axs[i, 0].grid(which='major', axis='y',
                           color='darkgray', ls='--', lw=0.4)
            axs[i, 0].grid(which='both', axis='x',
                           color='darkgray', ls='--', lw=0.4)
            axs[i, 0].set_axisbelow(True)

            axs[i, 0].set_xlabel(yLab[i], fontname='Arial', fontsize=16)
            for tl in axs[i, 0].get_xticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotClk21(fClkList, ClkList, iPlot, OutFilePrefix, OutFileSuffix):
    '''
    Plot mean && std for clock rate and/or rate drift for specified clocks

    iPlot --- Which to plot
              # 0, rate
              # 1, rate drift
              # 2, both
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    ClkName, EpoStr, Clk = ReadRnxClk(fClkList, ClkList, 0)
    nClk = len(ClkName)
    ClkName1 = ClkName.copy()
    ClkName1.sort()
    if nClk == 0:
        sys.exit('No clk found!')

    yLab = ['Clock rate [ps/s]', r'Clock rate drift [ps/$s^2$]']
    if iPlot != 2:
        # Only rate or rate drift
        if iPlot == 0:
            print('{: <8s} {: >7s} {: >7s}'.format('Clk', 'RMean', 'RSTD'))
        else:
            print('{: <8s} {: >7s} {: >7s}'.format('Clk', 'DMean', 'DSTD'))
        # Mean && STD for each satellite
        y = [[], []]
        for i in range(nClk):
            j = ClkName.index(ClkName1[i])
            if np.count_nonzero(~np.isnan(Clk[:, j*4+2+iPlot])) == 0:
                # No data
                y[0].append(np.nan)
                y[1].append(np.nan)
            else:
                # s -> ps
                y[0].append(np.nanmean(Clk[:, j*4+2+iPlot])*1e12)
                y[1].append(np.nanstd(Clk[:, j*4+2+iPlot])*1e12)
            strTmp = '{: <8s} {: >7.1f} {: >7.2f}'.format(
                ClkName1[i], y[0][i], y[1][i])
            print(strTmp)

        fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(nClk*0.6, 4))

        x = np.arange(nClk)
        axs[0, 0].set_xlim(left=-1, right=nClk)
        w = 1/(1+1)
        axs[0, 0].bar(x+(0-1/2)*w, y[0], width=w, yerr=y[1], align='edge',
                      error_kw=dict(ecolor='r', capsize=2, elinewidth=1, capthick=0.6))

        axs[0, 0].set_ylabel(yLab[iPlot], fontname='Arial', fontsize=16)
        for tl in axs[0, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        axs[0, 0].grid(which='major', axis='y', color='darkgray',
                       ls='--', lw=0.4)
        axs[0, 0].set_axisbelow(True)

        axs[0, 0].set_xlabel('Satellite PRNs', fontname='Arial', fontsize=16)
        axs[0, 0].set_xticks(x)
        axs[0, 0].set_xticklabels(
            ClkName1, fontdict={'fontsize': 14, 'fontname': 'Arial'})
    else:
        # Both rate and rate drift
        fig, axs = plt.subplots(
            2, 1, sharex='col', squeeze=False, figsize=(nClk*0.6, 8))
        x = np.arange(nClk)
        w = 1/(1+1)
        fig.subplots_adjust(hspace=0.1)

        for i in range(2):
            y = [[], []]
            for j in range(nClk):
                k = ClkName.index(ClkName1[j])
                if np.count_nonzero(~np.isnan(Clk[:, k*4+2+i])) == 0:
                    y[0].append(np.nan)
                    y[1].append(np.nan)
                else:
                    y[0].append(np.nanmean(Clk[:, k*4+2+i])*1e12)
                    y[1].append(np.nanstd(Clk[:, k*4+2+i])*1e12)

            axs[i, 0].bar(x+(0-1/2)*w, y[0], width=w, yerr=y[1], align='edge',
                          error_kw=dict(ecolor='r', capsize=2, elinewidth=1, capthick=0.6))

            axs[i, 0].set_ylabel(yLab[i], fontname='Arial', fontsize=16)
            for tl in axs[i, 0].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            axs[i, 0].grid(which='major', axis='y',
                           color='darkgray', ls='--', lw=0.4)
            axs[i, 0].set_axisbelow(True)

            axs[i, 0].set_ylim(bottom=0, top=nClk+1)
            axs[i, 0].set_yticks(np.arange(1, nClk+1))
            axs[i, 0].set_yticklabels(
                ClkName1, fontdict={'fontsize': 14, 'fontname': 'Arial'})
        axs[i, 0].set_xlim(left=-1, right=nClk)
        axs[i, 0].set_xlabel('Satellite PRNs', fontname='Arial', fontsize=16)
        axs[i, 0].set_xticks(x)
        axs[i, 0].set_xticklabels(
            ClkName1, fontdict={'fontsize': 14, 'fontname': 'Arial'})

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotClk22(fClkList, ClkList, nCol, iPlot, lFit, lPlotBad, OutFilePrefix, OutFileSuffix):
    '''
    Plot clock rate or rate drift series for specified clocks.
    Sigma clipping will be done on the overall data.

     nCol --- Number of columns of the figure if plotting for multiple clocks
    iPlot --- Which data to process/plot
              # 0, rate
              # 1, rate drift
     lFit --- Whether do a linear fitting
 lPlotBad --- Whether plot bad points
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    ClkName, EpoStr, Clk = ReadRnxClk(fClkList, ClkList, 0)
    nClk = len(ClkName)
    ClkName1 = ClkName.copy()
    ClkName1.sort()
    if nClk == 0:
        sys.exit('No clk found!')
    else:
        # Cal the number of row based on specified number of col
        nRow = math.ceil(nClk/nCol)

    fig, axs = plt.subplots(nRow, nCol, sharex='col', sharey='row',
                            squeeze=False, figsize=(nCol*8, nRow*3))

    yLab = ['Clock rate [s/s]', r'Clock rate drift [s/$s^2$]']
    if iPlot == 0:
        # Clock rate
        print('{: <8s} {: >5s} {: >5s} {: >5s} {: >12s} {: >12s} {: >12s}'.format(
            'Clk', 'EpoN', 'EpoX', 'EPOI', 'MeaR', 'MedR', 'SigR'))
    else:
        # Clock rate drift
        print('{: <8s} {: >5s} {: >5s} {: >5s} {: >12s} {: >12s} {: >12s}'.format(
            'Clk', 'EpoN', 'EpoX', 'EPOI', 'MeaD', 'MedD', 'SigD'))
    # Close warnings from Astropy
    warnings.filterwarnings('ignore', category=UserWarning, append=True)

    for i in range(nClk):
        j = ClkName.index(ClkName1[i])
        nEpo = np.count_nonzero((~np.isnan(Clk[:, j*4+2+iPlot])))
        if nEpo == 0:
            print('No data for clk '+ClkName1[i])
            continue
        else:
            # Cal the axis position, row-wise
            iRow = math.ceil((i+1)/nCol)-1
            iCol = i-iRow*nCol
        # Do the sigma clipping
        Ma = stats.sigma_clip(Clk[:, j*4+2+iPlot],
                              sigma=3, maxiters=5, masked=True)
        # Pick out bad && good points
        xDel = [[], []]
        xGod = [[], []]
        for k in range(Ma.size):
            if np.isnan(Clk[k, j*4+2+iPlot]):
                continue
            if Ma.mask[k]:
                # Clipped point
                xDel[0].append(Clk[k, j*4])
                xDel[1].append(Clk[k, j*4+2+iPlot])
            else:
                # Good point
                xGod[0].append(Clk[k, j*4])
                xGod[1].append(Clk[k, j*4+2+iPlot])
        nDel = len(xDel[0])
        nGod = len(xGod[0])
        # Get the robust statistics
        Mea = np.mean(xGod[1])
        Med = np.median(xGod[1])
        Sig = np.std(xGod[1])
        # Report to the terminal
        strTmp = '{: <8s} {:>5d} {:>5d} {:>5d} {:>12.5E} {:>12.5E} {:>12.5E}'.format(
                 ClkName1[i], nEpo, nDel, nGod, Mea, Med, Sig)
        print(strTmp)

        # Mea,Med,Sig=stats.sigma_clipped_stats(Clk[:,j*4+2+iPlot],sigma=3,maxiters=5,mask_value=np.nan)
        axs[iRow, iCol].plot(xGod[0], xGod[1], 'g.', ms=4)
        if lPlotBad:
            # Plot also the bad points
            axs[iRow, iCol].plot(xDel[0], xDel[1], 'rx', ms=4)
        axs[iRow, iCol].text(0.02, 0.98, ClkName1[i], transform=axs[iRow, iCol].transAxes, ha='left', va='top',
                             fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
        axs[iRow, iCol].text(0.98, 0.98, '{:>8.5E} ({:>8.5E}) +/- {:>8.5E}'.format(Mea, Med, Sig),
                             transform=axs[iRow, iCol].transAxes, ha='right', va='top',
                             fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
        axs[iRow, iCol].grid(which='major', axis='y',
                             color='darkgray', ls='--', lw=0.4)
        axs[iRow, iCol].set_axisbelow(True)

        if iCol == 0:
            axs[iRow, iCol].set_ylabel(
                yLab[iPlot], fontname='Arial', fontsize=16)
            # axs[iRow,iCol].ticklabel_format(axis='y',style='sci',useOffset=False,useMathText=True)
            axs[iRow, iCol].yaxis.set_major_formatter('{x: >.2E}')
            for tl in axs[iRow, iCol].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
        if iRow == (nRow-1):
            axs[iRow, iCol].xaxis.set_major_formatter('{x: >7.1f}')
            axs[iRow, iCol].set_xlabel(
                'Modified Julian Day', fontname='Arial', fontsize=16)
            for tl in axs[iRow, iCol].get_xticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotClk23(fClkList, ClkList, nCol, iPlot, lFit, lFitRes, lPlotBad, OutFilePrefix, OutFileSuffix):
    '''
    Plot clock rate or rate drift series for specified clocks.
    Similar to PlotClk22, but Sigma clipping will be done on the data
    from individual file first.

     nCol --- Number of columns of the figure if plotting for multiple clocks
    iPlot --- Which data to process/plot
              # 0, rate
              # 1, rate drift
     lFit --- Whether do a linear fitting
  lFitRes --- If a linear fitting done, plot the fitting residuals or
              still the original data points, i.e., the observations
 lPlotBad --- Whether plot bad points
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    # Close warnings from Astropy
    warnings.filterwarnings('ignore', category=UserWarning, append=True)

    # Global clock list and data table
    ClkName = []
    Clk = []
    for i in range(len(fClkList)):
        ClkName0, EpoStr0, Clk0 = ReadRnxClk([fClkList[i]], ClkList, 0)
        for j in range(len(ClkName0)):
            if ClkName0[j] not in ClkName:
                # New clock
                ClkName.append(ClkName0[j])
                # Good points for this clock
                Clk.append([])
                Clk.append([])
                # Bad points for this clock
                Clk.append([])
                Clk.append([])
            iClk = ClkName.index(ClkName0[j])
            # Do the sigma clipping for each clock
            Ma0 = stats.sigma_clip(
                Clk0[:, j*4+2+iPlot], sigma=3, maxiters=5, masked=True)
            # Pick out the bad && good points
            for k in range(Ma0.size):
                if np.isnan(Clk0[k, j*4+2+iPlot]):
                    continue
                if Ma0.mask[k]:
                    # Clipped point
                    Clk[iClk*4+2].append(Clk0[k, j*4])
                    Clk[iClk*4+3].append(Clk0[k, j*4+2+iPlot])
                else:
                    # Good point
                    Clk[iClk*4].append(Clk0[k, j*4])
                    Clk[iClk*4+1].append(Clk0[k, j*4+2+iPlot])
    nClk = len(ClkName)
    ClkName1 = ClkName.copy()
    ClkName1.sort()
    if nClk == 0:
        sys.exit('No clk found!')
    else:
        # Cal the number of row based on specified number of col
        nRow = math.ceil(nClk/nCol)

    fig, axs = plt.subplots(nRow, nCol, sharex='col',
                            squeeze=False, figsize=(nCol*8, nRow*3))

    yLab = ['Clock rate [s/s]', r'Clock rate drift [s/$s^2$]']
    if iPlot == 0:
        # Clock Rate
        print('{: <8s} {: >5s} {: >5s} {: >5s} {: >12s} {: >12s} {: >12s} {: >12s} {: >12s}'.format(
            'Clk', 'EpoN', 'EpoX', 'EPOI', 'MeaR', 'MedR', 'SigR[1e-13]', 'a0[1e-12]', 'a1[1e-19]'))
    else:
        # Clock rate Drift
        print('{: <8s} {: >5s} {: >5s} {: >5s} {: >12s} {: >12s} {: >12s}'.format(
            'Clk', 'EpoN', 'EpoX', 'EPOI', 'MeaD', 'MedD', 'SigD'))

    for i in range(nClk):
        j = ClkName.index(ClkName1[i])
        # Sum of bad and good poins
        nEpo = len(Clk[j*4]) + len(Clk[j*4+2])
        if nEpo == 0:
            print('No data for clk '+ClkName1[i])
            continue
        else:
            # Cal the axis position, row-wise
            iRow = math.ceil((i+1)/nCol)-1
            iCol = i-iRow*nCol

        # Do the overall sigma clipping again on the clean data?
        xDel = [[], []]
        xGod = [[], []]
        if False:
            Ma = stats.sigma_clip(Clk[j*4+1], sigma=3, maxiters=5, masked=True)
            # Pick out bad && good points
            for k in range(Ma.size):
                if Ma.mask[k]:
                    # Clipped point
                    xDel[0].append(Clk[j*4][k])
                    xDel[1].append(Clk[j*4+1][k])
                else:
                    # Good point
                    xGod[0].append(Clk[j*4][k])
                    xGod[1].append(Clk[j*4+1][k])
            # Add the before clipped data to the bad point list
            xDel[0] += Clk[j*4+2]
            xDel[1] += Clk[j*4+3]
        else:
            xGod[0] = Clk[j*4]
            xGod[1] = Clk[j*4+1]
            xDel[0] = Clk[j*4+2]
            xDel[1] = Clk[j*4+3]

        # The good and bad points list
        x1 = []
        y1 = []
        x2 = []
        y2 = []
        if not lFit:
            # If not fit, simply get plot the original data points and print some
            # statistics
            x1 = xGod[0]
            y1 = xGod[1]
            x2 = xDel[0]
            y2 = xDel[1]
            axs[iRow, iCol].plot(x1, y1, 'g.', ms=4, label='God')
            if lPlotBad:
                # Plot also the bad points
                axs[iRow, iCol].plot(x2, y2, 'rx', ms=4, label='Del')
            nGod = len(x1)
            nDel = len(x2)
            Mea = np.mean(y1)
            Med = np.median(y1)
            Sig = np.std(y1)
            # Report to the terminal
            strTmp = '{: <8s} {:>5d} {:>5d} {:>5d} {:>12.5E} {:>12.5E} {:>12.5E}'.format(
                ClkName1[i], nEpo, nDel, nGod, Mea, Med, Sig)
            print(strTmp)
            # Show on the axis (statistics)
            strTmp1 = '{:>8.5E} ({:>8.5E}) +/- {:>8.5E}'.format(Mea, Med, Sig)
        else:
            # Do the iterative fitting using sigma clipping
            ORFit = AsFit.FittingWithOutlierRemoval(AsFit.LinearLSQFitter(), stats.sigma_clip,
                                                    niter=3, sigma=3.0)
            # weights
            w = np.zeros(len(xGod[1]))
            w[:] = 1.0
            fm, ma = ORFit(AsMod.Linear1D(), np.array(
                xGod[0]), np.array(xGod[1]), weights=w)
            # The fitted model, for both good part and bad part
            x0 = np.array(xGod[0])
            y0 = fm(x0)
            yd = fm(xDel[0])

            # Pick out the good and bad points
            for k in range(ma.size):
                if ma[k]:
                    # Clipped point
                    x2.append(xGod[0][k])
                    if lFitRes:
                        y2.append(xGod[1][k]-y0[k])
                    else:
                        y2.append(xGod[1][k])
                else:
                    # Non-clipped, i.e. good points
                    x1.append(xGod[0][k])
                    if lFitRes:
                        y1.append(xGod[1][k]-y0[k])
                    else:
                        y1.append(xGod[1][k])
            # Also include the bad points detected at the beginning
            for k in range(len(xDel[0])):
                x2.append(xDel[0][k])
                if lFitRes:
                    y2.append(xDel[1][k]-yd[k])
                else:
                    y2.append(xDel[1][k])
            # Plot the data points, original or residuals
            axs[iRow, iCol].plot(x1, y1, 'g.', ms=4, label='God')
            if lPlotBad:
                # Plot also the bad points
                axs[iRow, iCol].plot(x2, y2, 'rx', ms=4, label='Del')
            if not lFitRes:
                # Plot the fitting model as well, only plot when not plotting for residuals
                # because otherwise, it is ugly.
                # As it is a line, better to sort them before plotting
                ind = np.argsort(x0)
                axs[iRow, iCol].plot(
                    x0[ind], y0[ind], 'k-', lw=2, label='Fitted')
            # Calculate the intercept at the start of the data not at the zero point as default
            a = fm.intercept.value + fm.slope.value*np.amin(x0)
            # Convert the rate from per day -> per sec
            b = fm.slope.value/86400

            # Get the statistics
            nGod = len(x1)
            nDel = len(x2)
            Mea = np.mean(y1)
            Med = np.median(y1)
            Sig = np.std(y1)
            # Report to the terminal
            strTmp = '{: <8s} {:>5d} {:>5d} {:>5d} {:>12.5E} {:>12.5E} {:>12.5E} {:>12.5E} {:>12.5E}'.format(
                ClkName1[i], nEpo, nDel, nGod, Mea, Med, Sig*1e13, a*1e12, b*1e19)
            print(strTmp)
            # Show on the axis (fitted model parameters)
            strTmp1 = 'a= {:>8.5E}, b= {:>8.5E}'.format(a, b)

        axs[iRow, iCol].text(0.02, 0.98, ClkName1[i], transform=axs[iRow, iCol].transAxes, ha='left', va='top',
                             fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
        axs[iRow, iCol].text(0.98, 0.98, strTmp1, transform=axs[iRow, iCol].transAxes, ha='right', va='top',
                             fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
        axs[iRow, iCol].grid(which='major', axis='y',
                             color='darkgray', ls='--', lw=0.4)
        axs[iRow, iCol].set_axisbelow(True)

        if iCol == 0:
            axs[iRow, iCol].set_ylabel(
                yLab[iPlot], fontname='Arial', fontsize=16)
        # axs[iRow,iCol].ticklabel_format(axis='y',style='sci',useOffset=False,useMathText=True)
        axs[iRow, iCol].yaxis.set_major_formatter('{x: >.2E}')
        for tl in axs[iRow, iCol].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        if iRow == (nRow-1):
            axs[iRow, iCol].xaxis.set_major_formatter('{x: >7.1f}')
            axs[iRow, iCol].set_xlabel(
                'Modified Julian Day', fontname='Arial', fontsize=16)
            for tl in axs[iRow, iCol].get_xticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotClkADev0(fClkList, ClkList, dIntv, OutFilePrefix, OutFileSuffix):
    '''
    Plot the Allan deviation of specified clocks one by one

    dIntv --- Interval of the clock data, in seconds
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    ClkName, EpoStr, Clk = ReadRnxClk(fClkList, ClkList, 0)

    nClk = len(ClkName)
    ClkName1 = ClkName.copy()
    ClkName1.sort()

    fig, axs = plt.subplots(nClk, 1, sharex='col',
                            squeeze=False, figsize=(4, nClk*4))

    for i in range(nClk):
        j = ClkName.index(ClkName1[i])
        # Only plot for continuous data
        if np.count_nonzero(np.isnan(Clk[:, j*4+1])) > 0:
            continue

        t = np.logspace(0, 4, 50)
        (t2, ad, ade, adn) = allantools.oadev(
            Clk[:, j*4+1], rate=1/dIntv, data_type="phase", taus=t)
        axs[i, 0].loglog(t2, ad)

        axs[i, 0].set_ylabel('Allan Dev [sec]', fontname='Arial', fontsize=16)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        axs[i, 0].text(0.98, 0.98, ClkName[j], transform=axs[i, 0].transAxes, ha='right', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})

    axs[i, 0].set_xlabel(r'Averaging Time $\tau$ [sec]',
                         fontname='Arial', fontsize=16)
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotClkADev1(fClkList, ClkList, dIntv, OutFilePrefix, OutFileSuffix):
    '''
    Plot the Allan deviation of specified clocks

    dIntv --- Interval of the clock data, in seconds
              Original clock data will be re-sampled to this
              sampling interval
    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    ClkName, EpoStr, Clk = ReadRnxClk(fClkList, ClkList, dIntv)
    nClk = len(ClkName)
    ClkName1 = ClkName.copy()
    ClkName1.sort()
    if nClk == 0:
        sys.exit('No clk found!')

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(8, 8))

    axs[0, 0].set_ylim(bottom=1e-15, top=1e-11)
    axs[0, 0].set_xlim(left=1e2, right=1e5)
    axs[0, 0].set_prop_cycle(color=['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r',
                                    'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g',
                                    'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'],
                             marker=['.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                     '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                     '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X'])
    print('{: <6s} {: >8s} {: >8s} {: >8s} {: >8s}'.format(
        'Clk', 'Tau[s]', 'Sig', 'SigE', '#'))
    for i in range(nClk):
        j = ClkName.index(ClkName1[i])
        # Only plot for continuous data
        if np.count_nonzero(np.isnan(Clk[:, j*4+1])) > 0:
            print('Non-continuous clk, '+ClkName1[i])
            continue

        t = np.logspace(0, 5, 50)
        (t2, ad, ade, adn) = allantools.oadev(
            Clk[:, j*4+1], rate=1/dIntv, data_type="phase", taus=t)
        axs[0, 0].loglog(t2, ad, label=ClkName[j], markevery=0.1, lw=1)
        # axs[0,0].set_xscale('log')
        # axs[0,0].set_yscale('log')
        # axs[0,0].errorbar(t2,ad,yerr=ade,label=ClkName[j],ls='--',lw=1)
        for k in range(len(t2)):
            strTmp = '{: <6s} {:>8.2f} {:>8.2e} {:>8.2e} {:>8.1f}'.format(ClkName1[i],
                                                                          t2[k], ad[k], ade[k], adn[k])
            print(strTmp)

    axs[0, 0].grid(which='both', axis='both',
                   alpha=0.5, ls='--', lw=0.8, color='darkgray')
    axs[0, 0].legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5), framealpha=0.6,
                     labelspacing=0.1, borderpad=0.1, prop={'family': 'Arial', 'size': 14})

    axs[0, 0].set_ylabel('Allan Dev [sec]', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    axs[0, 0].set_xlabel(r'Averaging Time $\tau$ [sec]',
                         fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotClkADev2(fClkList, ClkList, dIntv, AC, OutFilePrefix, OutFileSuffix):
    '''
    Plot the Allan deviation of specified clocks within each clock file

    dIntv --- Interval of the clock data, in seconds
       AC --- Analysis Center of the clock files. This is mainly for determining
              the filename of the plots.
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    strTmp = os.path.join(OutFilePrefix, OutFileSuffix+'.pdf')
    with PdfPages(strTmp) as pdf:
        for k in range(len(fClkList)):
            if AC == 'phb':
                YYYY = int(fClkList[k][-7:-3])
                DOY = int(fClkList[k][-3:])
            else:
                # IGS/MGEX clock
                WK = int(fClkList[k][-9:-5])
                WKD = int(fClkList[k][-5:-4])
                YYYY, DOY = GNSSTime.wkd2doy(WK, WKD)

            ClkName, EpoStr, Clk = ReadRnxClk([fClkList[k]], ClkList, 0)
            nClk = len(ClkName)
            ClkName1 = ClkName.copy()
            ClkName1.sort()
            if nClk == 0:
                print('No clk found in '+fClkList[k])
                continue

            fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(8, 8))
            axs[0, 0].text(0.50, 1.00, fClkList[k], transform=axs[0, 0].transAxes, ha='center', va='bottom',
                           fontdict={'fontsize': 10, 'fontname': 'Arial', 'fontweight': 'bold'})

            axs[0, 0].set_ylim(bottom=1e-15, top=1e-11)
            axs[0, 0].set_xlim(left=1e2, right=1e5)
            axs[0, 0].set_prop_cycle(color=['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r',
                                            'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g',
                                            'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'],
                                     marker=['.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                             '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                             '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X'])

            for i in range(nClk):
                j = ClkName.index(ClkName1[i])
                # Only plot for continuous data
                if np.count_nonzero(np.isnan(Clk[:, j*4+1])) > 0:
                    print('Non-continuous clk, ' +
                          ClkName1[i]+' in '+fClkList[k])
                    continue
                t = np.logspace(0, 4, 50)
                (t2, ad, ade, adn) = allantools.oadev(
                    Clk[:, j*4+1], rate=1/dIntv, data_type="phase", taus=t)
                axs[0, 0].loglog(t2, ad, label=ClkName[j], markevery=0.1, lw=1)
            axs[0, 0].grid(which='both', axis='both',
                           alpha=0.5, ls='--', lw=0.8, color='darkgray')
            axs[0, 0].legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5), framealpha=0.6,
                             labelspacing=0.1, borderpad=0.1, prop={'family': 'Arial', 'size': 14})

            axs[0, 0].set_ylabel(
                'Allan Dev [sec]', fontname='Arial', fontsize=16)
            for tl in axs[0, 0].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)

            axs[0, 0].set_xlabel(
                r'Averaging Time $\tau$ [sec]', fontname='Arial', fontsize=16)
            for tl in axs[0, 0].get_xticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def PlotClkADev3(h, OutFilePrefix, OutFileSuffix):
    '''
    Plot the Sigma-Tao plot based on a theoretical Allan Deviation Model

    Ref. Martoccia D, Bernstein H, Chan Y, Frueholz R, Wu A (1998)
         GPS satellite timing performance using the autonomous navigation(autonav).
         Paper presented at the ION 98,

      h --- the Allan deviation model coefficients
            # 0, the white frequency noise coefficient
            # 1, the flicker frequency noise coefficient
            # 2, random walk frequency coefficient
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(8, 8))

    # Average time in seconds (max ~11 days)
    t = np.logspace(0, 6, num=100)
    y = np.sqrt(h[0]*h[0]/t + h[1]*h[1] + h[2]*h[2]*t, dtype=np.float64)

    # Plot the x-axis (or abscissa) in days
    axs[0, 0].loglog(t/86400.0, y, 'o-', ms=3, lw=1)
    axs[0, 0].grid(which='both', axis='both', c='darkgray', ls='--', lw=0.4)
    axs[0, 0].set_axisbelow(True)

    axs[0, 0].set_ylabel('Allan Dev [sec]', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    # The model coefficients
    strTmp = r'$h_{-1}$='+'{: >8.2E}, '.format(h[0]) + r'$h_0$='+'{: >8.2E}, '.format(
        h[1]) + r'$h_1$='+'{: >8.2E}'.format(h[2])
    axs[0, 0].text(0.98, 0.98, strTmp, transform=axs[0, 0].transAxes, ha='right', va='top',
                   fontdict={'fontsize': 14, 'fontname': 'Arial'})

    axs[0, 0].set_xlabel(r'Averaging Time $\tau$ [day]',
                         fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotClkDif10(fClkList1, fClkList2, ClkList, nFit, OutFilePrefix, OutFileSuffix):
    '''
    Plot the (direct) diff time series of specific clocks from two sets of RINEX
    clock files

    NOTE: Sigma clipping is applied on the overall data as a whole for each clock.

    ClkList --- List of specified clocks
       nFit --- If non-negative, fitting the data to a polynomial of `nFit` degree
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    ClkName, dClk = DiffRnxClk0(fClkList1, fClkList2, ClkList)
    nClk = len(ClkName)
    ClkName1 = ClkName.copy()
    ClkName1.sort()
    if nClk == 0:
        sys.exit('No clk found!')
    # Close warnings from Astropy
    warnings.filterwarnings('ignore', category=UserWarning, append=True)

    fig, axs = plt.subplots(nClk, 1, sharex='col',
                            squeeze=False, figsize=(12, nClk*1.5))
    # Header line of the report
    print('{: <8s} {: >5s} {: >5s} {: >5s} {: >12s} {: >12s} {: >12s}'.format(
        'Clk', 'EpoN', 'EpoX', 'EPOI', 'Mea', 'Med', 'Sig'))
    for i in range(nClk):
        j = ClkName.index(ClkName1[i])
        # Epoch
        t = np.array(dClk[j*4])
        # Clock offset
        dx = np.array(dClk[j*4+1])
        # Clock rate
        dv = np.array(dClk[j*4+2])
        # Number of valid points
        nEpo = np.count_nonzero(~np.isnan(dClk[j*4+1]))
        # Do the sigma clipping for offset
        Ma = stats.sigma_clip(dx, sigma=5, maxiters=5, masked=True)
        # Pick out bad && good points
        xDel = [[], []]
        xGod = [[], []]
        for k in range(Ma.size):
            if np.isnan(dx[k]):
                continue
            if Ma.mask[k]:
                # Clipped point
                xDel[0].append(t[k])
                xDel[1].append(dx[k])
            else:
                # Good point
                xGod[0].append(t[k])
                xGod[1].append(dx[k])
        nDel = len(xDel[0])
        nGod = len(xGod[0])
        Mea = np.mean(xGod[1])
        Med = np.median(xGod[1])
        Sig = np.std(xGod[1])
        # Report to the terminal
        strTmp = '{: <8s} {:>5d} {:>5d} {:>5d} {:>12.5E} {:>12.5E} {:>12.5E}'.format(
                 ClkName1[i], nEpo, nDel, nGod, Mea, Med, Sig)
        print(strTmp)

        if nFit >= 0:
            # Fit with nFit-order polynomial
            c = np.polynomial.polynomial.polyfit(xGod[0], xGod[1], nFit)
            # Fitting residuals
            xGod[1] = xGod[1] - np.polynomial.polynomial.polyval(xGod[0], c)
        axs[i, 0].plot(xGod[0], xGod[1], '.r', ms=2, label='offset')
        axs[i, 0].grid(which='major', axis='y', color='darkgray',
                       ls='--', lw=0.4)
        axs[i, 0].set_axisbelow(True)
        # Show on the axis
        strTmp = '{:>6.2f}+/-{:>5.2f}'.format(
            np.mean(xGod[1]), np.std(xGod[1]))
        axs[i, 0].text(0.98, 0.02, strTmp, transform=axs[i, 0].transAxes, ha='right', va='bottom',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkred'})

        axs[i, 0].set_ylabel('[ns]', fontname='Arial', fontsize=16)
        axs[i, 0].ticklabel_format(
            axis='y', style='sci', useOffset=False, useMathText=True)
        axs[i, 0].yaxis.set_major_formatter('{x: >8.2f}')
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        axs[i, 0].text(0.02, 0.98, ClkName[i], transform=axs[i, 0].transAxes, ha='left', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
        # axs[i,0].legend(ncol=2,loc='upper center',prop={'family':'Arial','size':14})

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


def PlotClkDif11(YYYY0, DOY0, nDay, ClkPath1, AC1, ClkPath2, AC2, ClkList, nFit, nCol,
                 cExl, fAngList, iPlot, OutFilePrefix, OutFileSuffix):
    '''
    Plot the (direct) diff time series of specific clocks from two sets of RINEX
    clock files in the orbit beta-u system.

    NOTE: Similar to PlotClkDif10, but
    1) Sigma clipping is applied on indiviudal daily comparison for each clock first.
    2) Daily statistics of the comparison concerning the number of bad and good
       points, the mean, median && STD for each clock are output to a file
    3) Daily polynomial fitting can be required and the resulting trend would be
       removed from the differencing time series
    4) The accumulated differencing data points can be plotted either in the form
       of time series or using the orbit angular coordinates, i.e. the beta-u system
    5) Except for accumulated time series plots, STDs of the overall means which
       indicate the variation of daily means would be plotted as bar-plot for all clocks

    ClkList --- List of specified clocks
       nFit --- If non-negative, fitting the data to a polynomial of `nFit` degree
       nCol --- Number of columns of the output figure
       cExl ---
   fAngList --- List of orbital geometry angle files
      iPlot --- Which plot(s) is/are required
                # 0, only the differencing data point would be plotted
                # 1, only the STDs of means would be plotted
                # 2, both of the above mentioned would be plotted
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # Close warnings from Astropy
    warnings.filterwarnings('ignore', category=UserWarning, append=True)

    # Global clock name, data table
    ClkName = []
    dClk = []
    # Daily mean, std, number of valid points of each clock
    RMS = []
    for iDay in range(nDay):
        # Epoch for each day
        RMS.append([])
        RMS[iDay].append(0)

    YYYY = YYYY0
    DOY = DOY0
    for iDay in range(nDay):
        WK, WKD = GNSSTime.doy2wkd(YYYY, DOY)
        if AC1 == 'phb':
            fClk1 = os.path.join(
                ClkPath1, 'clk_{:04d}{:03d}'.format(YYYY, DOY))
        else:
            # IGS/MGEX clock
            fClk1 = os.path.join(
                ClkPath1, AC1+'{:04d}{:01d}.clk'.format(WK, WKD))
        if AC2 == 'phb':
            fClk2 = os.path.join(
                ClkPath2, 'clk_{:04d}{:03d}'.format(YYYY, DOY))
        else:
            # IGS/MGEX clock
            fClk2 = os.path.join(
                ClkPath2, AC2+'{:04d}{:01d}.clk'.format(WK, WKD))
        # Epoch of this day
        RMS[iDay][0] = GNSSTime.doy2mjd(YYYY, DOY)
        # Initialize the RMS table for current satellite list at this day
        for i in range(len(ClkName)):
            # Number of bad && good points
            RMS[iDay].append(0)
            RMS[iDay].append(0)
            # Mean, Median && STD
            RMS[iDay].append(np.nan)
            RMS[iDay].append(np.nan)
            RMS[iDay].append(np.nan)

        if os.path.isfile(fClk1) and os.path.isfile(fClk2):
            ClkName0, dClk0 = DiffRnxClk0([fClk1], [fClk2], ClkList)
            nClk0 = len(ClkName0)
            for i in range(nClk0):
                if ClkName0[i] not in ClkName:
                    # New satellite
                    ClkName.append(ClkName0[i])
                    # Expand the RMS table to account for the new sat
                    for k in range(nDay):
                        # Number of clipped and keeped points
                        RMS[k].append(0)
                        RMS[k].append(0)
                        # Mean, Median, STD
                        RMS[k].append(np.nan)
                        RMS[k].append(np.nan)
                        RMS[k].append(np.nan)
                    # Expand the data table to account for the new sat
                    # Clipped data points for this satellite
                    dClk.append([])
                    dClk.append([])
                    # Good data points for this satellite
                    dClk.append([])
                    dClk.append([])
                # Index of this sat in the global table
                j = ClkName.index(ClkName0[i])

                # Do sigma clipping for this clock offset
                Ma = stats.sigma_clip(
                    dClk0[i*4+1], sigma=5, maxiters=5, masked=True)
                # Pick out bad && good points
                xDel = [[], []]
                xGod = [[], []]
                for k in range(Ma.size):
                    if np.isnan(dClk0[i*4+1][k]):
                        continue
                    if Ma.mask[k]:
                        # Clipped point
                        xDel[0].append(dClk0[i*4][k])
                        xDel[1].append(dClk0[i*4+1][k])
                    else:
                        # Good point
                        xGod[0].append(dClk0[i*4][k])
                        xGod[1].append(dClk0[i*4+1][k])
                # Add to global table
                RMS[iDay][1+j*5] = len(xDel[0])
                RMS[iDay][1+j*5+1] = len(xGod[0])
                RMS[iDay][1+j*5+2] = np.mean(xGod[1])
                RMS[iDay][1+j*5+3] = np.median(xGod[1])
                RMS[iDay][1+j*5+4] = np.std(xGod[1])
                if nFit >= 0:
                    # Fit with nFit-order polynomial
                    c = np.polynomial.polynomial.polyfit(
                        xGod[0], xGod[1], nFit)
                    # Fitting residuals
                    xGod[1] = xGod[1] - \
                        np.polynomial.polynomial.polyval(xGod[0], c)
                # Gloabl data table
                for k in range(len(xDel[0])):
                    dClk[j*4].append(xDel[0][k])
                    dClk[j*4+1].append(xDel[1][k])
                for k in range(len(xGod[0])):
                    dClk[j*4+2].append(xGod[0][k])
                    dClk[j*4+3].append(xGod[1][k])
        elif not os.path.isfile(fClk1) and os.path.isfile(fClk2):
            print(fClk1+' not exist!')
        elif os.path.isfile(fClk1) and not os.path.isfile(fClk2):
            print(fClk2+' not exist!')
        else:
            print('Both '+fClk1+' and '+fClk2+' not exist!')
        # Next day
        DOY = DOY+1
        if GNSSTime.IsLeapYear(YYYY):
            if DOY > 366:
                DOY = DOY-366
                YYYY = YYYY+1
        else:
            if DOY > 365:
                DOY = DOY-365
                YYYY = YYYY+1
    nClk = len(ClkName)
    ClkName1 = ClkName.copy()
    ClkName1.sort()

    # Now do some plots
    # Firstly, for differencing points
    if nClk == 0:
        sys.exit('No clk found!')

    cPRNBDS = ['C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28',
               'C29', 'C30', 'C32', 'C33', 'C34', 'C35', 'C36', 'C37']
    cSVNBDS = ['C201', 'C202', 'C206', 'C205', 'C209', 'C210', 'C212', 'C211', 'C203', 'C204',
               'C207', 'C208', 'C213', 'C214', 'C216', 'C215', 'C218', 'C219']

    if iPlot == 0 or iPlot == 2:
        # Cal the number of row based on specified number of col
        nRow = math.ceil(nClk/nCol)
        if len(fAngList) > 0:
            # Get the SVN for clock
            cSVNClk = []
            for i in range(nClk):
                if ClkName1[i] not in cPRNBDS:
                    continue
                cSVNClk.append(cSVNBDS[cPRNBDS.index(ClkName1[i])])
            # Beta, E, Eps, u
            iRec = [8, 9, 10, 11]
            cSat, Ang = plotGeoAng.GetAng(fAngList, cSVNClk, 'ANG', iRec)
            # The SVN list of the available satellites
            cSVN = []
            for i in range(len(cSat)):
                cSVN.append(cSat[i][0:4])
            # Get the index of each clock in the ANG table
            iAng = []
            for i in range(nClk):
                if ClkName1[i] not in cPRNBDS:
                    iAng.append(-1)
                elif cSVNBDS[cPRNBDS.index(ClkName1[i])] not in cSVN:
                    iAng.append(-1)
                else:
                    iAng.append(cSVN.index(
                        cSVNBDS[cPRNBDS.index(ClkName1[i])]))

        fig, axs = plt.subplots(nRow, nCol, sharex='col', sharey='row',
                                squeeze=False, figsize=(nCol*8, nRow*1.5))

        for i in range(nClk):
            j = ClkName.index(ClkName1[i])
            # Number of good points
            nEpo = len(dClk[j*4+2])
            if nEpo == 0:
                print('No valid data points for clk '+ClkName1[i])
                continue
            else:
                # Cal the axis position, row-wise
                iRow = math.ceil((i+1)/nCol)-1
                iCol = i-iRow*nCol
            if len(fAngList) > 0 and iAng[i] >= 0:
                # Match the angle with data
                tAng = np.array(Ang[iAng[i]*(4+2)])
                xAng = np.array(Ang[iAng[i]*(4+2)+2:iAng[i]*(4+2)+6])
                y = [[], [], []]
                ind1 = np.argsort(np.array(dClk[j*4+2]))
                ind2 = np.argsort(tAng)
                for k in range(ind1.size):
                    for l in range(ind2.size):
                        if (dClk[j*4+2][ind1[k]]-tAng[ind2[l]])*86400 > 1:
                            continue
                        elif (dClk[j*4+2][ind1[k]]-tAng[ind2[l]])*86400 < -1:
                            break
                        else:
                            y[0].append(dClk[j*4+3][ind1[k]])
                            # Beta
                            y[1].append(xAng[0][ind2[l]])
                            # u
                            y[2].append(xAng[3][ind2[l]])
                sc = axs[iRow, iCol].scatter(y[2], y[1], c=y[0], s=2.5)
                # Label of x-/y-axis
                xLab = r'$\mu$ [deg]'
                yLab = r'$\beta$ [deg]'
                # Ticklabel Format of x-/y-axis
                xTF = '{x: >4.0f}'
                yTF = '{x: >3.0f}'
                cbar = fig.colorbar(sc, ax=axs[iRow, iCol])
                cbar.set_label('[ns]', loc='center',
                               fontname='Arial', fontsize=16)
                for tl in cbar.ax.get_yticklabels():
                    tl.set_fontname('Arial')
                    tl.set_fontsize(14)
                cbar.ax.yaxis.set_major_formatter('{x: >5.2f}')
            else:
                axs[iRow, iCol].plot(dClk[j*4+2], dClk[j*4+3], '.r', ms=2)
                xLab = 'Modified Julian Day'
                yLab = '[ns]'
                xTF = '{x: >7.1f}'
                yTF = '{x: >8.2f}'

            axs[iRow, iCol].text(0.02, 0.98, ClkName1[i], transform=axs[iRow, iCol].transAxes, ha='left', va='top',
                                 fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
            axs[iRow, iCol].grid(which='major', axis='y',
                                 color='darkgray', ls='--', lw=0.4)
            axs[iRow, iCol].set_axisbelow(True)
            strTmp = '{:>6.2f}+/-{:>5.2f}'.format(
                np.mean(dClk[j*4+3]), np.std(dClk[j*4+3]))
            axs[iRow, iCol].text(0.98, 0.02, strTmp, transform=axs[i, 0].transAxes, ha='right', va='bottom',
                                 fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkred'})

            if iCol == 0:
                axs[iRow, iCol].set_ylabel(yLab, fontname='Arial', fontsize=16)
                # axs[iRow,iCol].ticklabel_format(axis='y',style='sci',useOffset=False,useMathText=True)
                axs[iRow, iCol].yaxis.set_major_formatter(yTF)
                for tl in axs[iRow, iCol].get_yticklabels():
                    tl.set_fontname('Arial')
                    tl.set_fontsize(14)
            if iRow == (nRow-1):
                axs[iRow, iCol].xaxis.set_major_formatter(xTF)
                axs[iRow, iCol].set_xlabel(xLab, fontname='Arial', fontsize=16)
                for tl in axs[iRow, iCol].get_xticklabels():
                    tl.set_fontname('Arial')
                    tl.set_fontsize(14)

        strTmp = OutFilePrefix+OutFileSuffix+'_1.png'
        fig.savefig(strTmp, transparent=True, bbox_inches='tight')
        strTmp = OutFilePrefix+OutFileSuffix+'_1.svg'
        fig.savefig(strTmp, transparent=True, bbox_inches='tight')
        plt.close(fig)

    if iPlot == 1 or iPlot == 2:
        # Secondly, for daily statistics
        fOut = open(OutFilePrefix+os.path.splitext(OutFileSuffix)[0], 'w')
        # Valid days for each clock
        Avg = []
        # Header line
        strTmp = '{: <8s}'.format('MJDEpoch')
        for i in range(nClk):
            Avg.append([])
            Avg.append([])
            Avg.append([])
            Avg.append([])
            Avg.append([])
            strTmp = strTmp+' {: >8s} {: >8s} {: >8s} {: >8s} {: >8s}'.format(ClkName1[i]+'_nEx',
                                                                              ClkName1[i]+'_nIn', ClkName1[i]+'_Mea', ClkName1[i]+'_Med', ClkName1[i]+'_STD')
        fOut.write(strTmp+'\n')
        # Print for each day
        for iDay in range(nDay):
            strTmp = '{:>8.2f}'.format(RMS[iDay][0])
            for i in range(nClk):
                j = ClkName.index(ClkName1[i])
                lExcluded = False
                if RMS[iDay][1+j*5+1] < 0.5:
                    # No Valid points for this clock at this day
                    Mea = 99999.99
                    Med = 99999.99
                    STD = 9999.999
                    lExcluded = True
                else:
                    Mea = RMS[iDay][1+j*5+2]
                    Med = RMS[iDay][1+j*5+3]
                    STD = RMS[iDay][1+j*5+4]
                strTmp = strTmp+' {:>8.0f} {:>8.0f} {:>8.2f} {:>8.2f} {:>8.3f}'.format(RMS[iDay][1+j*5],
                                                                                       RMS[iDay][1+j*5+1], Mea, Med, STD)
                if lExcluded:
                    continue
                # Cal the average considering the possible excluded sateelites
                # Firstly, check if this day is registered for exlcuding
                for k in range(len(cExl)):
                    # The first element in each record list is the MJD
                    if math.fabs(float(cExl[k][0])-RMS[iDay][0])*86400 > 10:
                        continue
                    # From the second element, they are the clocks to be excluded
                    # for this MJD
                    if (cExl[k][1] == 'ALL') or (ClkName1[i] in cExl[k][1:]):
                        lExcluded = True
                        break
                if not lExcluded:
                    # Valid day for this clock
                    Avg[i*5].append(RMS[iDay][1+j*5])
                    Avg[i*5+1].append(RMS[iDay][1+j*5+1])
                    Avg[i*5+2].append(Mea)
                    Avg[i*5+3].append(Med)
                    Avg[i*5+4].append(STD)
            fOut.write(strTmp+'\n')
        # The Average line
        strTmp = '{: <8s}'.format('Average')
        for i in range(nClk):
            if len(Avg[i*5]) == 0:
                # No records for this clock
                strTmp = strTmp+' {:>8.0f} {:>8.0f} {:>8.2f} {:>8.2f} {:>8.3f}'.format(0, 0,
                                                                                       99999.99, 99999.99, 9999.999)
            else:
                strTmp = strTmp+' {:>8.1f} {:>8.0f} {:>8.2f} {:>8.2f} {:>8.3f}'.format(np.mean(Avg[i*5]),
                                                                                       np.mean(Avg[i*5+1]), np.mean(Avg[i*5+2]), np.mean(Avg[i*5+3]), np.mean(Avg[i*5+4]))
        fOut.write(strTmp+'\n')
        # The Sigma line
        strTmp = '{: <8s}'.format('Sigma')
        for i in range(nClk):
            if len(Avg[i*5]) == 0:
                # No records for this clock
                strTmp = strTmp+' {:>8.0f} {:>8.0f} {:>8.2f} {:>8.2f} {:>8.3f}'.format(0, 0,
                                                                                       99999.99, 99999.99, 9999.999)
            else:
                # Number of valid days, sum points
                strTmp = strTmp+' {:>8.0f} {:>8.0f} {:>8.2f} {:>8.2f} {:>8.3f}'.format(len(Avg[i*5]),
                                                                                       np.sum(Avg[i*5+1]), np.std(Avg[i*5+2]), np.std(Avg[i*5+3]), np.std(Avg[i*5+4]))
        fOut.write(strTmp+'\n')
        fOut.close()

        # Plot the STDs of daily mean
        x = np.arange(nClk)
        fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(nClk*0.6, 4))
        axs[0, 0].set_xlim(left=-1, right=nClk)
        # the width of the bars
        w = 1/(1+1)
        Sig = []
        for i in range(nClk):
            if len(Avg[i*5]) == 0:
                Sig.append(np.nan)
            else:
                # STD of mean values
                Sig.append(np.std(Avg[i*5+2]))
        axs[0, 0].bar(x+(0-1/2)*w, Sig, w, align='edge', label='STD')

        axs[0, 0].grid(which='major', axis='y', color='darkgray',
                       ls='--', lw=0.4)
        axs[0, 0].set_axisbelow(True)
        axs[0, 0].set_ylabel('STD [ns]', fontname='Arial', fontsize=16)
        axs[0, 0].set_xticks(x)
        axs[0, 0].set_xticklabels(
            ClkName1, fontdict={'fontsize': 14, 'fontname': 'Arial'})
        for tl in axs[0, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        strTmp = OutFilePrefix+OutFileSuffix+'_2.png'
        fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
        strTmp = OutFilePrefix+OutFileSuffix+'_2.svg'
        fig.savefig(strTmp, transparent=True, bbox_inches='tight')
        plt.close(fig)


def PlotClkDif12(cSer, fSer, lSTD, OutFilePrefix, OutFileSuffix):
    '''
    Plot the STD or Mean comparison for several solutions based on the output
    file from PlotClkDif11.

    NOTE: We assume the clock names are satellite PRNs !!!

    lSTD --- Plot the comparison of STDs ptherwise Means
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nSer = len(cSer)
    RMS = []
    for i in range(nSer*2):
        # STD && Mea for each ser
        RMS.append([])
    cSat = []
    for i in range(nSer):
        with open(fSer[i], mode='rt') as fOb:
            iSat = []
            lReadSat = False
            lReadAvg = False
            lReadSig = False
            for cLine in fOb:
                if cLine[0:8] == 'MJDEpoch':
                    # Read the satellite PRN list
                    cWords = cLine[8:].split()
                    for j in range(len(cWords)):
                        if cWords[j][3:7] != '_Mea':
                            continue
                        if cWords[j][0:3] in cSat:
                            iSat.append(cSat.index(cWords[j][0:3]))
                        else:
                            # New Satellite
                            cSat.append(cWords[j][0:3])
                            iSat.append(len(cSat)-1)
                            for k in range(nSer*2):
                                RMS[k].append(np.nan)
                    lReadSat = True
                elif cLine[0:7] == 'Average':
                    if not lReadSat:
                        sys.exit('Failed to read the sat line in '+fSer[i])
                    cWords = cLine[7:].split()
                    for j in range(0, len(cWords), 5):
                        # Average of Mean
                        RMS[i*2][iSat[j//5]] = float(cWords[j+2])
                    lReadAvg = True
                elif cLine[0:5] == 'Sigma':
                    if not lReadSat:
                        sys.exit('Failed to read the sat line in '+fSer[i])
                    cWords = cLine[5:].split()
                    for j in range(0, len(cWords), 5):
                        # STD of Mean
                        RMS[i*2+1][iSat[j//5]] = float(cWords[j+2])
                    lReadSig = True
                elif lReadSat and lReadAvg and lReadSig:
                    break
            if (not lReadAvg) or (not lReadSig):
                sys.exit('Failed to read the avg/sig line in '+fSer[i])
    nClk = len(cSat)
    ClkName0 = cSat.copy()
    ClkName0.sort()

    x = np.arange(nClk)
    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(nClk*0.6, 4))

    axs[0, 0].set_xlim(left=-1, right=nClk)
    # axs[0,0].set_ylim(bottom=0,top=0.6)

    # the width of the bars
    w = 1/(nSer+1)
    for i in range(nSer):
        Sig = []
        for j in range(nClk):
            k = cSat.index(ClkName0[j])
            if lSTD:
                # STD
                Sig.append(RMS[i*2+1][k])
            else:
                # Mean
                Sig.append(RMS[i*2][k])
        axs[0, 0].bar(x+(i-nSer/2)*w, Sig, w, align='edge', label=cSer[i])

    axs[0, 0].legend(ncol=nSer, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                     prop={'family': 'Arial', 'size': 14})

    axs[0, 0].grid(which='both', axis='y', color='darkgray', ls='--',
                   lw=0.8)
    axs[0, 0].set_axisbelow(True)
    if lSTD:
        axs[0, 0].set_ylabel('STD [ns]', fontname='Arial', fontsize=16)
    else:
        axs[0, 0].set_ylabel('Mean [ns]', fontname='Arial', fontsize=16)
    axs[0, 0].set_xlabel('Satellite PRNs', fontname='Arial', fontsize=16)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(
        ClkName0, fontdict={'fontsize': 14, 'fontname': 'Arial'})
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotClkDif20(fClkList, ClkPair, OutFilePrefix, OutFileSuffix):
    '''
    Plot the time series of clock offset diff between specified clock pairs

    ClkPair --- the list of specified clock pairs
    '''

    nPair = len(ClkPair)
    # Get list of clocks involved into the diff
    ClkList = []
    for i in range(nPair):
        cWords = ClkPair[i].split(sep='-')
        if cWords[0] not in ClkList:
            ClkList.append(cWords[0])
        if cWords[1] not in ClkList:
            ClkList.append(cWords[1])
    # Read data for required clocks
    ClkName, EpoStr, Clk = ReadRnxClk(fClkList, ClkList, 0)

    fig, axs = plt.subplots(nPair, 1, sharex='col',
                            squeeze=False, figsize=(8, nPair*2))
    # fig.subplots_adjust(hspace=0.1)
    formatterx = mpl.ticker.StrMethodFormatter('{x:8.2f}')

    for i in range(nPair):
        cWords = ClkPair[i].split(sep='-')
        if (cWords[0] not in ClkName) or (cWords[1] not in ClkName):
            continue
        i1 = ClkName.index(cWords[0])
        i2 = ClkName.index(cWords[1])
        # Cal the diff for each clock pair
        t = []
        dClk = []
        for j in range(len(EpoStr[i1])):
            if EpoStr[i1][j] not in EpoStr[i2]:
                continue
            k = EpoStr[i2].index(EpoStr[i1][j])
            t.append(Clk[j, i1*4])
            # clock offset diff, sec -> meter
            dClk.append((Clk[k, i2*4+1]-Clk[j, i1*4+1])*299792458)
        axs[i, 0].plot(t, dClk, '.r', ms=2)
        axs[i, 0].set_ylabel('[m]', fontname='Arial', fontsize=16)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        axs[i, 0].text(0.05, 0.95, ClkPair[i], transform=axs[i, 0].transAxes, ha='left', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})

    axs[i, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    axs[i, 0].xaxis.set_major_formatter(formatterx)
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotClkDif40(fClkList, ClkList, ClkRef, OutFilePrefix, OutFileSuffix):
    '''
    Plot the time series of clock offset diff for specified clocks
    with respect to a reference clock

    ClkList --- list of clocks of interest
     ClkRef --- the reference clock
    '''

    ClkName, EpoStr, Clk = ReadRnxClk(fClkList, ClkList, 0)
    ClkName0, EpoStr0, Clk0 = ReadRnxClk(fClkList, [ClkRef], 0)
    if len(ClkName0) != 1 or ClkName0[0] != ClkRef:
        sys.exit('Ref clk not found, '+ClkRef)

    ClkName1 = ClkName.copy()
    ClkName1.sort()
    if ClkRef in ClkName:
        nClk = len(ClkName) - 1
    else:
        nClk = len(ClkName)
    fig, axs = plt.subplots(nClk, 1, sharex='col',
                            squeeze=False, figsize=(12, nClk*1.5))
    formatterx = mpl.ticker.StrMethodFormatter('{x:7.1f}')

    i = -1
    for iClk in range(len(ClkName1)):
        if ClkName1[iClk] == ClkRef:
            continue
        # Index for the axse
        i = i+1
        j = ClkName.index(ClkName1[iClk])
        # Cal the diff offset
        t = []
        dXClk = []
        for k in range(len(EpoStr[j])):
            if EpoStr[j][k] not in EpoStr0[0]:
                continue
            l = EpoStr0[0].index(EpoStr[j][k])
            t.append(Clk[k, j*4])
            # sec -> nanosec
            dXClk.append((Clk[k, j*4+1]-Clk0[l, 1])*1e9)
        axs[i, 0].plot(t, dXClk, '.r', ms=2)
        axs[i, 0].set_ylabel('[ns]', fontname='Arial', fontsize=16)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        axs[i, 0].text(0.98, 0.98, ClkName[j], transform=axs[i, 0].transAxes, ha='right', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})

    axs[i, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    axs[i, 0].xaxis.set_major_formatter(formatterx)
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotClkDif50(YYYY0, DOY0, nDay, ClkPath1, AC1, ClkPath2, AC2, ClkList, ClkRefList,
                 lPrintMeaSTD, OutFilePrefix, OutFileSuffix):
    '''
    Plot the diff series of clocks between two sets of clock files after
    the double-difference, i.e
    1) firstly, within each set, all but the selected ref clock are differenced wrt the
       reference clock;
    2) then, resulted clocks from those two sets are differenced respectively.

         YYYY0 --- Year of start
          DOY0 --- DOY of start
          nDay --- Number of days
      ClkPath1 --- Path of the first clk solution
           AC1 --- AC of the first clk solution
      ClkPath2 --- Path of the second clk solution
           AC2 --- AC of the second clk solution
       ClkList --- Specified clk list
    ClkRefList --- Specified ref clk list (by priority)
  lPrintMeaSTD --- Whether print the mean and STD on the axis

    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    fClkList1 = []
    fClkList2 = []
    YYYY = YYYY0
    DOY = DOY0
    for iDay in range(nDay):
        WK, WKD = GNSSTime.doy2wkd(YYYY, DOY)
        # Start epoch
        if iDay == 0:
            rMJD1 = GNSSTime.doy2mjd(YYYY, DOY)
        # End epoch
        if iDay == nDay-1:
            # Assume they are daily files
            rMJD2 = GNSSTime.doy2mjd(YYYY, DOY) + 1
        if AC1 == 'phb':
            fClk1 = os.path.join(
                ClkPath1, 'clk_{:04d}{:03d}'.format(YYYY, DOY))
        else:
            # IGS/MGEX clock
            fClk1 = os.path.join(
                ClkPath1, AC1+'{:04d}{:01d}.clk'.format(WK, WKD))
        if AC2 == 'phb':
            fClk2 = os.path.join(
                ClkPath2, 'clk_{:04d}{:03d}'.format(YYYY, DOY))
        else:
            # IGS/MGEX clock
            fClk2 = os.path.join(
                ClkPath2, AC2+'{:04d}{:01d}.clk'.format(WK, WKD))
        if os.path.isfile(fClk1) and os.path.isfile(fClk2):
            fClkList1.append(fClk1)
            fClkList2.append(fClk2)
        elif not os.path.isfile(fClk1) and os.path.isfile(fClk2):
            print(fClk1+' not exist!')
        elif os.path.isfile(fClk1) and not os.path.isfile(fClk2):
            print(fClk2+' not exist!')
        else:
            print('Both '+fClk1+' and '+fClk2+' not exist!')
        # Next day
        DOY = DOY+1
        if GNSSTime.IsLeapYear(YYYY):
            if DOY > 366:
                DOY = DOY-366
                YYYY = YYYY+1
        else:
            if DOY > 365:
                DOY = DOY-365
                YYYY = YYYY+1

    ClkRef, ClkName, dClk = DiffRnxClk1(
        fClkList1, fClkList2, ClkRefList, ClkList)
    nClk = len(ClkName)
    ClkName1 = ClkName.copy()
    ClkName1.sort()
    if nClk == 0:
        sys.exit('No clk found!')

    fig, axs = plt.subplots(nClk, 1, sharex='col',
                            squeeze=False, figsize=(12, nClk*1.5))
    # fig.subplots_adjust(hspace=0.1)

    # Header line of the report
    print('{: <4s} {: >11s} {: >5s}'.format('Clk', 'Mean', 'STD'))
    for i in range(nClk):
        j = ClkName.index(ClkName1[i])
        # in nano sec
        Mea = np.nanmean(dClk[j*2+1])
        Sig = np.nanstd(dClk[j*2+1])
        # Remove mean
        axs[i, 0].plot(dClk[j*2], dClk[j*2+1]-Mea, '.r', ms=2)
        axs[i, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
        axs[i, 0].set_axisbelow(True)
        # Whether print the mean and STD on the axis
        if lPrintMeaSTD:
            strTmp = '{:>6.2f}+/-{:>5.2f}'.format(Mea, Sig)
            axs[i, 0].text(0.98, 0.98, strTmp, transform=axs[i, 0].transAxes, ha='right', va='top',
                           fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkred'})
        # Report to terminal
        strTmp = '{: <4s} {:>11.2f} {:>5.2f}'.format(ClkName[j], Mea, Sig)
        print(strTmp)

        axs[i, 0].set_ylabel('[ns]', fontname='Arial', fontsize=16)
        axs[i, 0].yaxis.set_major_formatter('{x:>6.2f}')
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        axs[i, 0].text(0.02, 0.98, ClkName[j]+' (Ref '+ClkRef+')', transform=axs[i, 0].transAxes, ha='left', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
    axs[i, 0].set_xlim(left=rMJD1, right=rMJD2)
    axs[i, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    axs[i, 0].xaxis.set_major_formatter('{x:>7.1f}')
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


def PlotClkDif51(YYYY0, DOY0, nDay, ClkPath1, AC1, ClkPath2, AC2, ClkList, ClkRefList,
                 OutFilePrefix, OutFileSuffix):
    '''
    Similar to PlotClkDif50, but plot for each day within a specified period

       YYYY0 --- Year of start
        DOY0 --- DOY of start
        nDay --- Number of days
    ClkPath1 --- Path of the first clk solution
         AC1 --- AC of the first clk solution
    ClkPath2 --- Path of the second clk solution
         AC2 --- AC of the second clk solution
     ClkList --- Specified clk list
  ClkRefList --- list of reference clocks. The first available one will
                 be used
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    strTmp = os.path.join(OutFilePrefix, OutFileSuffix+'.pdf')
    with PdfPages(strTmp) as pdf:
        YYYY = YYYY0
        DOY = DOY0
        for iDay in range(nDay):
            WK, WKD = GNSSTime.doy2wkd(YYYY, DOY)
            if AC1 == 'phb':
                fClk1 = os.path.join(
                    ClkPath1, 'clk_{:04d}{:03d}'.format(YYYY, DOY))
            else:
                # IGS/MGEX clock
                fClk1 = os.path.join(
                    ClkPath1, AC1+'{:04d}{:01d}.clk'.format(WK, WKD))
            if AC2 == 'phb':
                fClk2 = os.path.join(
                    ClkPath2, 'clk_{:04d}{:03d}'.format(YYYY, DOY))
            else:
                # IGS/MGEX clock
                fClk2 = os.path.join(
                    ClkPath2, AC2+'{:04d}{:01d}.clk'.format(WK, WKD))

            if os.path.isfile(fClk1) and os.path.isfile(fClk2):
                ClkRef, ClkName, dClk = DiffRnxClk1(
                    [fClk1], [fClk2], ClkRefList, ClkList)
                nClk = len(ClkName)
                ClkName1 = ClkName.copy()
                ClkName1.sort()
                if nClk == 0:
                    print('No diff for '+'{:4d}{:03d}'.format(YYYY, DOY))
                else:
                    fig, axs = plt.subplots(
                        nClk, 1, sharex='col', squeeze=False, figsize=(12, nClk*1.5))
                    # fig.suptitle('{:4d}{:03d}'.format(YYYY,DOY),fontfamily='Arial',fontsize=18,fontweight='bold')
                    axs[0, 0].text(0.5, 1.0, '{:4d}{:03d} (Ref {: <4s})'.format(YYYY, DOY, ClkRef),
                                   transform=axs[0, 0].transAxes, ha='center', va='bottom',
                                   fontdict={'fontsize': 18, 'fontname': 'Arial', 'fontweight': 'bold'})
                    for i in range(nClk):
                        j = ClkName.index(ClkName1[i])
                        # in nano sec
                        Mea = np.nanmean(dClk[j*2+1])
                        Sig = np.nanstd(dClk[j*2+1])
                        # Remove mean
                        axs[i, 0].plot(dClk[j*2], dClk[j*2+1]-Mea, '.r', ms=2)
                        axs[i, 0].text(0.98, 0.98, '{:>6.2f}+/-{:>5.2f}'.format(Mea, Sig),
                                       transform=axs[i, 0].transAxes, ha='right', va='top',
                                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkred'})

                        axs[i, 0].set_ylabel(
                            '[ns]', fontname='Arial', fontsize=16)
                        axs[i, 0].yaxis.set_major_formatter('{x:>6.2f}')
                        for tl in axs[i, 0].get_yticklabels():
                            tl.set_fontname('Arial')
                            tl.set_fontsize(14)

                        axs[i, 0].text(0.02, 0.98, ClkName[j], transform=axs[i, 0].transAxes, ha='left', va='top',
                                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})

                    axs[i, 0].set_xlabel(
                        'Modified Julian Day', fontname='Arial', fontsize=16)
                    axs[i, 0].xaxis.set_major_formatter('{x:>7.1f}')
                    for tl in axs[i, 0].get_xticklabels():
                        tl.set_fontname('Arial')
                        tl.set_fontsize(14)

                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close(fig)
            elif not os.path.isfile(fClk1) and os.path.isfile(fClk2):
                print(fClk1+' not exist!')
            elif os.path.isfile(fClk1) and not os.path.isfile(fClk2):
                print(fClk2+' not exist!')
            else:
                print('Both '+fClk1+' and '+fClk2+' not exist!')

            # Next day
            DOY = DOY+1
            if GNSSTime.IsLeapYear(YYYY):
                if DOY > 366:
                    DOY = DOY-366
                    YYYY = YYYY+1
            else:
                if DOY > 365:
                    DOY = DOY-365
                    YYYY = YYYY+1


def PlotClkDifRMS0(fClkList1, fClkList2, ClkList, lFit, OutFilePrefix, OutFileSuffix):
    '''
    Plot the (direct) diff RMS of specific clocks between two sets of clock files

    ClkList --- Specified clocks
       lFit --- Whether remove the trend from the diff series
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    ClkName, dClk = DiffRnxClk0(fClkList1, fClkList2, ClkList)
    nClk = len(ClkName)
    ClkName1 = ClkName.copy()
    ClkName1.sort()

    if nClk == 0:
        sys.exit('No clk found!')
    RMS = np.zeros((nClk, 2))

    for i in range(nClk):
        t = np.array(dClk[i*4])
        dx = np.array(dClk[i*4+1])
        dv = np.array(dClk[i*4+2])
        if lFit:
            # Remove the trend in residuals
            # ind=np.argsort(t)
            # Fit with a linear function
            c = np.polynomial.polynomial.polyfit(t, dx, 1)
            dx = dx - np.polynomial.polynomial.polyval(t, c)
        # Cal the RMS for this clock
        nObs1 = 0
        nObs2 = 0
        for k in range(t.size):
            # Offset
            if not np.isnan(dx[k]):
                nObs1 = nObs1+1
                RMS[i, 0] = RMS[i, 0] + dx[k]*dx[k]
            # Rate
            if not np.isnan(dv[k]):
                nObs2 = nObs2+1
                RMS[i, 1] = RMS[i, 1] + dv[k]*dv[k]
        if nObs1 > 0:
            RMS[i, 0] = np.sqrt(RMS[i, 0]/nObs1)
        else:
            RMS[i, 0] = np.nan
        if nObs2 > 0:
            RMS[i, 1] = np.sqrt(RMS[i, 1]/nObs2)
        else:
            RMS[i, 1] = np.nan

    x = np.arange(nClk)
    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 4))

    axs[0, 0].set_xlim(left=-1, right=nClk)
    axs[0, 0].set_ylim(bottom=0, top=1.0)

    # the width of the bars
    w = 1/(1+1)
    # Clock offset
    axs[0, 0].bar(x+(0-1/2)*w, RMS[:, 0], w, align='edge', label='Clk Offset')

    axs[0, 0].grid(which='both', axis='y', color='darkgray', ls='--', lw=0.8)
    axs[0, 0].set_axisbelow(True)
    axs[0, 0].set_ylabel('[ns]', fontname='Arial', fontsize=16)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(
        ClkName, fontdict={'fontsize': 14, 'fontname': 'Arial'})
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotClkDifRMS1(fClkList1, fClkList2, ClkList, ClkRefList, lFit, iPlot, OutFilePrefix, OutFileSuffix):
    '''
    Plot the mean/std/rms of clock offset diff between two sets of clock files for specified
    clocks by adopting the double-difference method

       ClkList --- list of specified clocks
    ClkRefList --- list of specified referenced clocks. The first available clock
                   will be used.
          lFit --- Whether remove the (linear) trend from the diff series
         iPlot --- Which info should be plotted
                   # 0, the mean
                   # 1, the std
                   # 2, the RMS
                   # 3, mean, std and RMS
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    ClkRef, ClkName, dClk = DiffRnxClk1(
        fClkList1, fClkList2, ClkRefList, ClkList)
    nClk = len(ClkName)
    ClkName1 = ClkName.copy()
    ClkName1.sort()

    if nClk == 0:
        sys.exit('No clk found!')
    # Mean, STD && RMS
    RMS = np.zeros((nClk, 3))

    for i in range(nClk):
        t = np.array(dClk[i*2])
        dx = np.array(dClk[i*2+1])
        if lFit:
            # Remove the trend in residuals
            # ind=np.argsort(t)
            # Fit with a linear function
            c = np.polynomial.polynomial.polyfit(t, dx, 1)
            dx = dx - np.polynomial.polynomial.polyval(t, c)
        # Cal the RMS for this clock
        RMS[i, 0] = np.mean(dx)
        RMS[i, 1] = np.std(dx)
        nObs1 = 0
        for k in range(t.size):
            # Offset
            if not np.isnan(dx[k]):
                nObs1 = nObs1+1
                RMS[i, 2] = RMS[i, 2] + dx[k]*dx[k]
        if nObs1 > 0:
            RMS[i, 2] = np.sqrt(RMS[i, 2]/nObs1)
        else:
            RMS[i, 2] = np.nan

    x = np.arange(nClk)
    w = 1/(1+1)
    yLab = ['Mean [ns]', 'STD [ns]', 'RMS [ns]']

    if iPlot != 3:
        fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(nClk*0.6, 4))
        axs[0, 0].set_xlim(left=-1, right=nClk)

        axs[0, 0].bar(x+(0-1/2)*w, RMS[:, iPlot], w, align='edge')
        axs[0, 0].grid(which='both', axis='y',
                       color='darkgray', ls='--', lw=0.8)
        axs[0, 0].set_axisbelow(True)
        axs[0, 0].set_ylabel(yLab[iPlot], fontname='Arial', fontsize=16)
        for tl in axs[0, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        axs[0, 0].set_xticks(x)
        axs[0, 0].set_xticklabels(
            ClkName, fontdict={'fontsize': 14, 'fontname': 'Arial'})

    else:
        fig, axs = plt.subplots(
            3, 1, sharex='col', squeeze=False, figsize=(nClk*0.6, 9))
        for i in range(3):
            axs[i, 0].bar(x+(0-1/2)*w, RMS[:, i], w, align='edge')

            axs[i, 0].grid(which='both', axis='y',
                           color='darkgray', ls='--', lw=0.8)
            axs[i, 0].set_axisbelow(True)
            axs[i, 0].set_ylabel(yLab[i], fontname='Arial', fontsize=16)
            axs[i, 0].xaxis.set_major_formatter('{x:>5.2f}')
            for tl in axs[i, 0].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
        axs[i, 0].set_xticks(x)
        axs[i, 0].set_xticklabels(
            ClkName, fontdict={'fontsize': 14, 'fontname': 'Arial'})

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotClkDifRMS2(fRefList, cSer, fSerList, ClkList, ClkRefList, OutFilePrefix, OutFileSuffix):
    '''
    Plot the comparison of std of clock offset (double-differenced) diff wrt a reference
    solution for different solutions.

    fRefList --- the reference solution of clock offsets
        cSer ---
    fSerList ---
     ClkList --- list of specified clocks
  ClkRefList --- list of specified referenced clocks. The first available clock
                 will be used.
    '''

    nSer = len(cSer)

    ClkName = []
    RMS = []
    for i in range(nSer):
        # Cal the diff RMS between each solution and the ref solution
        ClkRef, ClkName1, dClk1 = DiffRnxClk1(
            fRefList, fSerList[i], ClkRefList, ClkList)
        for j in range(len(ClkName1)):
            if ClkName1[j] not in ClkName:
                # New clock
                ClkName.append(ClkName1[j])
                iClk = len(ClkName)-1
                # Mean && STD list
                RMS.append([])
                RMS.append([])
                for k in range(nSer):
                    RMS[iClk*2].append(np.nan)
                    RMS[iClk*2+1].append(np.nan)
            else:
                iClk = ClkName.index(ClkName1[j])
            RMS[iClk*2][i] = np.nanmean(dClk1[j*2+1])
            RMS[iClk*2+1][i] = np.nanstd(dClk1[j*2+1])
    nClk = len(ClkName)
    ClkName0 = ClkName.copy()
    ClkName0.sort()

    x = np.arange(nClk)
    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(nClk*0.6, 4))

    axs[0, 0].set_xlim(left=-1, right=nClk)
    axs[0, 0].set_ylim(bottom=0, top=1.0)

    # the width of the bars
    w = 1/(nSer+1)
    for i in range(nSer):
        # STD
        Sig = []
        for j in range(nClk):
            k = ClkName.index(ClkName0[j])
            Sig.append(RMS[k*2+1][i])
        axs[0, 0].bar(x+(i-nSer/2)*w, Sig, w, align='edge', label=cSer[i])

    axs[0, 0].legend(ncol=nSer, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                     prop={'family': 'Arial', 'size': 14})

    axs[0, 0].grid(which='both', axis='y', color='darkgray', ls='--', lw=0.8)
    axs[0, 0].set_axisbelow(True)
    axs[0, 0].set_ylabel('STD [ns]', fontname='Arial', fontsize=16)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(
        ClkName0, fontdict={'fontsize': 14, 'fontname': 'Arial'})
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotClkDifRMS3(YYYY0, DOY0, nDay, ClkPath1, AC1, ClkPath2, AC2, ClkList, ClkRefList, iPlot, yLim,
                   lReport, cExl, OutFilePrefix, OutFileSuffix):
    '''
    Plot daily mean or/and std series of clock offset (double-differenced) diff between two solutions

        YYYY0 --- Year of start
         DOY0 --- DOY of start
         nDay --- Number of days
     ClkPath1 --- Path to the clock files of the first solution
          AC1 --- AC of the first clk solution
     ClkPath2 --- Path to the clock files of the second solution
          AC2 --- AC of the second clk solution
      ClkList --- list of specified clocks
   ClkRefList --- list of specified referenced clocks. The first available clock
                  will be used.
        iPlot --- Which info should be plotted
                  # 0, the mean series
                  # 1, the std series
                  # 2, both the mean && std series
         yLim --- bottom and top limits of the mean && std axes
                  if yLim[0]==999, do not set the limits for the mean axes
                  if yLim[2]==999, do not set the limits for the std axes
      lReport --- Whether report the daily mean && std of each satellite to a file
         cExl --- Specify the date and satellites to be excluded when ploting or reporting.
                  Each element is a list specifying the to-be-excluded clock list on a specified
                  day. And the first element of this list is the day for excluding. E.g.,
                  cExl = [['58827', 'ALL'],
                          ['58845', 'C36']]
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    ClkName = []
    RMS = []
    rEpo = []
    YYYY = YYYY0
    DOY = DOY0
    for i in range(nDay):
        rEpo.append(GNSSTime.doy2mjd(YYYY, DOY))
        WK, WKD = GNSSTime.doy2wkd(YYYY, DOY)
        if AC1 == 'phb':
            fClk1 = os.path.join(
                ClkPath1, 'clk_{:04d}{:03d}'.format(YYYY, DOY))
        else:
            fClk1 = os.path.join(
                ClkPath1, AC1+'{:04d}{:01d}.clk'.format(WK, WKD))
        if AC2 == 'phb':
            fClk2 = os.path.join(
                ClkPath2, 'clk_{:04d}{:03d}'.format(YYYY, DOY))
        else:
            fClk2 = os.path.join(
                ClkPath2, AC2+'{:04d}{:01d}.clk'.format(WK, WKD))
        if os.path.isfile(fClk1) and os.path.isfile(fClk2):
            # Do double-difference
            ClkRef, ClkName1, dClk1 = DiffRnxClk1(
                [fClk1], [fClk2], ClkRefList, ClkList)
            # Cal the mean && std for each clock
            for j in range(len(ClkName1)):
                if ClkName1[j] not in ClkName:
                    # New clock
                    ClkName.append(ClkName1[j])
                    iClk = len(ClkName)-1
                    # Mean && STD list
                    RMS.append([])
                    RMS.append([])
                    for k in range(nDay):
                        # Mean for each day
                        RMS[iClk*2].append(np.nan)
                        # STD for each day
                        RMS[iClk*2+1].append(np.nan)
                else:
                    iClk = ClkName.index(ClkName1[j])
                # Mean for each day
                RMS[iClk*2][i] = np.nanmean(dClk1[j*2+1])
                # STD for each day
                RMS[iClk*2+1][i] = np.nanstd(dClk1[j*2+1])
        elif not os.path.isfile(fClk1) and os.path.isfile(fClk2):
            print(fClk1+' does not exist!')
        elif os.path.isfile(fClk1) and not os.path.isfile(fClk2):
            print(fClk2+' does not exist!')
        else:
            print('Both '+fClk1+' and '+fClk2+' do not exist!')

        DOY = DOY+1
        if GNSSTime.IsLeapYear(YYYY):
            if DOY > 366:
                DOY = DOY-366
                YYYY = YYYY+1
        else:
            if DOY > 365:
                DOY = DOY-365
                YYYY = YYYY+1
    nClk = len(ClkName)
    ClkName0 = ClkName.copy()
    ClkName0.sort()

    # Exclude specified satellites
    for iDay in range(nDay):
        for i in range(nClk):
            j = ClkName.index(ClkName0[i])
            for k in range(len(cExl)):
                # The first element in each record list is the MJD
                if math.fabs(float(cExl[k][0])-rEpo[iDay])*86400 > 10:
                    continue
                # From the second element on, they are the clocks to be excluded
                # for this MJD
                if (cExl[k][1] == 'ALL') or (ClkName0[i] in cExl[k][1:]):
                    RMS[j*2][iDay] = np.nan
                    RMS[j*2+1][iDay] = np.nan
                    break

    # Report to a file
    if lReport:
        fOut = open(OutFilePrefix+os.path.splitext(OutFileSuffix)[0], 'w')
        # Header line
        strTmp = '# Excluded option:'
        fOut.write(strTmp+'\n')
        for i in range(len(cExl)):
            strTmp = '#                :'
            for j in range(len(cExl[i])):
                strTmp = strTmp+' '+cExl[i][j]
            fOut.write(strTmp+'\n')
        strTmp = '{: <8s}'.format('MJDEpoch')
        for i in range(nClk):
            strTmp = strTmp + ' {: >8s} {: >8s}'.format(
                ClkName0[i]+'_Mea', ClkName0[i]+'_STD')
        fOut.write(strTmp+'\n')
        # Print the mean && std for each satellite at each day as well as
        # their averages over the whole period
        Avg = np.zeros(nClk*3)
        for iDay in range(nDay):
            strTmp = '{:>8.2f}'.format(rEpo[iDay])
            for i in range(nClk):
                j = ClkName.index(ClkName0[i])
                lExcluded = False
                if np.isnan(RMS[j*2][iDay]) or np.isnan(RMS[j*2+1][iDay]):
                    Mea = 99999.99
                    STD = 9999.999
                    lExcluded = True
                else:
                    Mea = RMS[j*2][iDay]
                    STD = RMS[j*2+1][iDay]
                strTmp = strTmp+' {:>8.2f} {:>8.3f}'.format(Mea, STD)
                if not lExcluded:
                    # Number of counted days
                    Avg[i*3] = Avg[i*3]+1
                    Avg[i*3+1] = Avg[i*3+1]+Mea
                    Avg[i*3+2] = Avg[i*3+2]+STD
            fOut.write(strTmp+'\n')
        # The Average line
        strTmp = '{: <8s}'.format('Average')
        for i in range(nClk):
            if Avg[i*3] < 0.5:
                Avg[i*3+1] = 99999.99
                Avg[i*3+2] = 9999.999
            else:
                Avg[i*3+1] = Avg[i*3+1]/Avg[i*3]
                Avg[i*3+2] = Avg[i*3+2]/Avg[i*3]
            strTmp = strTmp+' {:>8.2f} {:>8.3f}'.format(Avg[i*3+1], Avg[i*3+2])
        fOut.write(strTmp+'\n')
        # The number of Valid Days
        strTmp = '{: <8s}'.format('ValidDay')
        for i in range(nClk):
            strTmp = strTmp+' {:>8.0f} {:>8.0f}'.format(Avg[i*3], Avg[i*3])
        fOut.write(strTmp+'\n')
        fOut.close()

    if iPlot != 2:
        # Only Mean or STD series
        fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 6))
        axs[0, 0].set_prop_cycle(color=['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r',
                                        'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g',
                                        'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'],
                                 marker=['.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                         '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                         '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X'])
        if iPlot == 0:
            axs[0, 0].set_ylabel('Mean [ns]', fontname='Arial', fontsize=16)
            if yLim[0] != 999:
                axs[0, 0].set_ylim(bottom=yLim[0], top=yLim[1])
        else:
            axs[0, 0].set_ylabel('STD [ns]', fontname='Arial', fontsize=16)
            if yLim[2] != 999:
                axs[0, 0].set_ylim(bottom=yLim[2], top=yLim[3])
        for i in range(nClk):
            j = ClkName.index(ClkName0[i])
            if iPlot == 0:
                # Mean series
                axs[0, 0].plot(rEpo, RMS[j*2], ls='--',
                               lw=1, label=ClkName0[i])
            else:
                # STD series
                axs[0, 0].plot(rEpo, RMS[j*2+1], ls='--',
                               lw=1, label=ClkName0[i])
        axs[0, 0].grid(which='major', axis='y',
                       color='darkgray', ls='--', lw=0.4)
        axs[0, 0].legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5), framealpha=0.6,
                         labelspacing=0.1, borderpad=0.1, prop={'family': 'Arial', 'size': 14})
        for tl in axs[0, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        axs[0, 0].set_xlabel('Modified Julian Day',
                             fontname='Arial', fontsize=16)
        for tl in axs[0, 0].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        axs[0, 0].xaxis.set_major_formatter('{x:7.1f}')
    else:
        # Both Mean and STD series
        fig, axs = plt.subplots(
            2, 1, sharex='col', squeeze=False, figsize=(12, 10))
        # First, mean
        axs[0, 0].set_prop_cycle(color=['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r',
                                        'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g',
                                        'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'],
                                 marker=['.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                         '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                         '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X'])
        axs[0, 0].set_ylabel('Mean [ns]', fontname='Arial', fontsize=16)
        if yLim[0] != 999:
            axs[0, 0].set_ylim(bottom=yLim[0], top=yLim[1])
        for i in range(nClk):
            j = ClkName.index(ClkName0[i])
            # Mean series
            axs[0, 0].plot(rEpo, RMS[j*2], ls='--', lw=1, label=ClkName0[i])
        axs[0, 0].grid(which='major', axis='y',
                       color='darkgray', ls='--', lw=0.4)
        axs[0, 0].set_axisbelow(True)
        axs[0, 0].legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.0), framealpha=0.6,
                         labelspacing=0.1, borderpad=0.1, prop={'family': 'Arial', 'size': 14})
        for tl in axs[0, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        # Then, STD
        axs[1, 0].set_prop_cycle(color=['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r',
                                        'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g',
                                        'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'],
                                 marker=['.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                         '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                         '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X'])
        axs[1, 0].set_ylabel('STD [ns]', fontname='Arial', fontsize=16)
        if yLim[2] != 999:
            axs[1, 0].set_ylim(bottom=yLim[2], top=yLim[3])
        for i in range(nClk):
            j = ClkName.index(ClkName0[i])
            # STD series
            axs[1, 0].plot(rEpo, RMS[j*2+1], ls='--', lw=1, label=ClkName0[i])
        axs[1, 0].grid(which='major', axis='y',
                       color='darkgray', ls='--', lw=0.4)
        axs[1, 0].set_axisbelow(True)
        for tl in axs[1, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        for tl in axs[1, 0].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        axs[1, 0].xaxis.set_major_formatter('{x:7.1f}')
        axs[1, 0].set_xlabel('Modified Julian Day',
                             fontname='Arial', fontsize=16)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotClkDifRMS4(cSer, fSer, ClkList0, yMax, OutFilePrefix, OutFileSuffix):
    '''
    Plot the comparison of average STD or Mean of several solutions based on the output
    files from PlotClkDifRMS3.

    NOTE: We assume the clock names are satellite PRNs !
          Currently, only STD comparison can be plotted.

    ClkList0 --- List of to-be-plotted clocks
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nSer = len(cSer)
    X = []
    for i in range(nSer*2):
        # STD && Mea for each ser
        X.append([])
    ClkName0 = []
    for i in range(nSer):
        with open(fSer[i], mode='rt') as fOb:
            iClk = []
            # Whether have read the clock name list
            lReadClk = False
            # Whether have read the average
            lReadAvg = False
            for cLine in fOb:
                if cLine[0:8] == 'MJDEpoch':
                    # Read the header line
                    cWords = cLine[8:].split()
                    for j in range(len(cWords)):
                        # Assume these are satellite clocks identified by PRNs
                        if cWords[j][3:7] != '_Mea':
                            continue
                        if cWords[j][0:3] in ClkName0:
                            iClk.append(ClkName0.index(cWords[j][0:3]))
                        else:
                            # New Satellite
                            ClkName0.append(cWords[j][0:3])
                            iClk.append(len(ClkName0)-1)
                            for k in range(nSer*2):
                                X[k].append(np.nan)
                    lReadClk = True
                elif cLine[0:7] == 'Average':
                    if not lReadClk:
                        sys.exit('Failed to read the sat line in '+fSer[i])
                    cWords = cLine[7:].split()
                    for j in range(0, len(cWords), 2):
                        # Mean
                        X[i*2][iClk[j//2]] = float(cWords[j])
                        # STD
                        X[i*2+1][iClk[j//2]] = float(cWords[j+1])
                    lReadAvg = True
                elif lReadClk and lReadAvg:
                    break
            if not lReadAvg:
                sys.exit('Failed to read the avg line in '+fSer[i])
    # Exclude some clocks if required
    if ClkList0[0] != 'ALL':
        ClkName = []
        RMS = []
        for i in range(nSer*2):
            # STD && Mea for each ser
            RMS.append([])
        for i in range(len(ClkName0)):
            if ClkName0[i] not in ClkList0:
                continue
            ClkName.append(ClkName0[i])
            for j in range(nSer):
                RMS[j*2].append(X[j*2][i])
                RMS[j*2+1].append(X[j*2+1][i])
    else:
        ClkName = ClkName0
        RMS = X

    nClk = len(ClkName)
    ClkName0 = ClkName.copy()
    ClkName0.sort()

    x = np.arange(nClk)
    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(nClk*0.6, 4))

    axs[0, 0].set_xlim(left=-1, right=nClk)
    if yMax[0] < yMax[1]:
        axs[0, 0].set_ylim(bottom=yMax[0], top=yMax[1])

    # the width of the bars
    w = 1/(nSer+1)
    for i in range(nSer):
        # STD
        Sig = []
        for j in range(nClk):
            k = ClkName.index(ClkName0[j])
            Sig.append(RMS[i*2+1][k])
        axs[0, 0].bar(x+(i-nSer/2)*w, Sig, w, align='edge', label=cSer[i])

    # Number of Cols for legend
    if nSer <= 5:
        nColLG = nSer
    else:
        nColLG = 5
    axs[0, 0].legend(ncol=nColLG, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                     framealpha=0.6, prop={'family': 'Arial', 'size': 14})

    axs[0, 0].grid(which='both', axis='y', color='darkgray', ls='--', lw=0.8)
    axs[0, 0].set_axisbelow(True)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    axs[0, 0].set_ylabel('STD [ns]', fontname='Arial', fontsize=16)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(
        ClkName0, fontdict={'fontsize': 14, 'fontname': 'Arial'})
    axs[0, 0].set_xlabel('Satellite PRNs', fontname='Arial', fontsize=16)

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

    # BDS-3
    cPRN3 = ['C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28',
             'C29', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35', 'C36', 'C37']

    cSer = []
    fSerList = []

    fClkList = glob.glob(os.path.join(
        cWrkPre0, r'MGEX/CLK/gbm_300s/gbm20843.clk'))

    # InFilePrefix=os.path.join(cWrkPre0,r'MGEX/CLK/gbm/')
    # InFilePrefix=os.path.join(cWrkPre0,r'PRO_2019001_2020366_WORK/I060/WORK2019335/')
    InFilePrefix = os.path.join(
        cWrkPre0, r'PRO_2019001_2020366_WORK/I060/2019/SATCLK_POD/')
    # InFilePrefix=r'D:/Code/PROJECT/WORK2019336_ERROR/'
    fClkList2 = glob.glob(InFilePrefix+'clk_2019352')
    # fClkList2=glob.glob(InFilePrefix+'gbm20820.clk')

    OutFilePrefix = os.path.join(cDskPre0, r'PRO_2019001_2020366/D650/CLK/')
    # OutFilePrefix=r'D:/Code/PROJECT/WORK_Clk/'

    # h=[9.48e-12,0,2.5e-17]
    # OutFileSuffix='AllanDev_test'
    # PlotClkADev3(h,OutFilePrefix,OutFileSuffix)

    # cSer.append('None')
    # InFilePrefix=r'Y:/PRO_2019001_2020366_WORK/I0_1/2019/SATCLK_POD/'
    # fResList=glob.glob(InFilePrefix+'clk_2019335')
    # fSerList.append(fResList)

    # cSer.append('Model 1')
    # InFilePrefix=r'Y:/PRO_2019001_2020366_WORK/I0_1_1/2019/SATCLK_POD/'
    # fResList=glob.glob(InFilePrefix+'clk_2019335')
    # fSerList.append(fResList)

    # cSer.append('Model 2')
    # InFilePrefix=r'Y:/PRO_2019001_2020366_WORK/I0_1_2/2019/SATCLK_POD/'
    # fResList=glob.glob(InFilePrefix+'clk_2019335')
    # fSerList.append(fResList)

    # cSer.append('Model 3')
    # InFilePrefix=r'Y:/PRO_2019001_2020366_WORK/I0_1_3/2019/SATCLK_POD/'
    # fResList=glob.glob(InFilePrefix+'clk_2019335')
    # fSerList.append(fResList)

    # OutFileSuffix='ClkDiffRMS_clk_to_wum_2019335_Comp.png'
    # PlotClkDifRMS2(fClkList,cSer,fSerList,['CXX'],['C20'],OutFilePrefix,OutFileSuffix)

    # OutFileSuffix='Clk_20193335_2019365'
    # PlotClk10(fClkList2,['CXX'],OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='Clk_2019335_2019365'
    # PlotClk12(fClkList2,['CXX'],OutFilePrefix,OutFileSuffix)

    # OutFileSuffix='VClk_2019335'
    # PlotClk11(fClkList2,['CXX'],OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='Rate_2019335'
    # PlotClk21(fClkList2,['CXX'],0,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='VClk_2019335_box'
    # PlotClk20(fClkList2,['CXX'],0,OutFilePrefix,OutFileSuffix)
    OutFileSuffix = 'Rate_2019335_2019365_3_Fit_new'
    # PlotClk23(fClkList2, ['C20'], 1, 0, True, True,
    #           True, OutFilePrefix, OutFileSuffix)
    # OutFileSuffix='Adev_2019335_gbm_GPS_300'
    # PlotClkADev1(fClkList2,['GXX'],300,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='Adev_2019335_phb_GPS'
    # PlotClkADev2(fClkList2,['GXX'],300,'phb',OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='RelClk_2019336_wum.png'
    # PlotClkDif40(fClkList,['CXX'],'C19',OutFilePrefix,OutFileSuffix)

    # OutFileSuffix='Diff_C01_2019336'
    # PlotClkDif10(fClkList2,fClkList,cPRN3,False,OutFilePrefix,OutFileSuffix)
    ClkPath1 = os.path.join(
        cWrkPre0, r'PRO_2019001_2020366_WORK/I060/2019/SATCLK_POD')
    # # # ClkPath1=os.path.join(cWrkPre0,r'MGEX/CLK/gbm_300s')
    ClkPath2 = os.path.join(
        cWrkPre0, r'PRO_2019001_2020366_WORK/C01/2019/SATCLK_POD')
    # # ClkPath2=os.path.join(cWrkPre0,r'MGEX/CLK/wum')
    # fAngList=glob.glob(os.path.join(cDskPre0,r'GNSS/PROJECT/OrbGeometry/ANG/')+'GeoAng_2019335_C2?')
    # OutFileSuffix='Diff_C01_2019335_2019365_1_test'
    # PlotClkDif11(2019,335,1,ClkPath1,'phb',ClkPath2,'phb',['C20','C21'],0,1,
    #              [],fAngList,0,OutFilePrefix,OutFileSuffix)
    # cSer=[]; fSer=[]
    # cSer.append('None')   ; fSer.append(os.path.join(cDskPre0,r'PRO_2019001_2020366/I060/HD/Diff_C01_2019335_2019365_1'))
    # cSer.append('Model 1'); fSer.append(os.path.join(cDskPre0,r'PRO_2019001_2020366/I061/HD/Diff_C01_2019335_2019365_1'))
    # cSer.append('Model 2'); fSer.append(os.path.join(cDskPre0,r'PRO_2019001_2020366/I062/HD/Diff_C01_2019335_2019365_1'))
    # cSer.append('Model 3'); fSer.append(os.path.join(cDskPre0,r'PRO_2019001_2020366/I063/HD/Diff_C01_2019335_2019365_1'))
    # OutFileSuffix='Diff_C01_2019335_2019365_STD_Comp'
    # PlotClkDif12(cSer,fSer,True,OutFilePrefix,OutFileSuffix)

    # OutFileSuffix='ClkDiffRMS_wum_to_phb_2019335.png'
    # PlotClkDifRMS0(fClkList,fClkList2,['CXX'],True,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='DiffRMS_wum_2019355_igs'
    # PlotClkDifRMS1(fClkList2,fClkList,['GXX'],['G01'],False,3,OutFilePrefix,OutFileSuffix)

    # ClkPath1 = os.path.join(
    #     cWrkPre0, r'PRO_2019001_2020366_WORK/I060/2019/SATCLK_POD')
    ClkPath1 = os.path.join(cWrkPre0, r'MGEX/CLK/gbm_300s')
    # ClkPath1 = os.path.join(cWrkPre0, r'MGEX/CLK/wum')
    ClkPath2 = os.path.join(
        cWrkPre0, r'PRO_2019001_2020366_WORK/I648/2019/SATCLK_POD')
    # ClkPath2 = os.path.join(cWrkPre0, r'MGEX/CLK/gbm_300s')
    # ClkPath2 = os.path.join(cWrkPre0, r'MGEX/CLK/wum')
    # cExl = [['58827', 'ALL'], ['58845', 'C36']]
    # cExl = [['58819', 'C23'], ['58827', 'ALL'], ['58845', 'C36']]
    # cExl = [['58819', 'C23'], ['58827', 'ALL']]
    # cExl = [['58819', 'C23']]
    cExl = []
    OutFileSuffix = 'Diff_gbm_2019335_2019365_phb_Mean_STD'
    # PlotClkDifRMS3(2019, 335, 31, ClkPath1, 'gbm', ClkPath2, 'phb', cPRN3, ['C21', 'C20'], 2,
    #                [999, 999, 0, 1.3], True, cExl, OutFilePrefix, OutFileSuffix)

    cSer = []
    fSer = []
    cSer.append('No ISL')
    fSer.append(os.path.join(
        cDskPre0, r'PRO_2019001_2020366/C01/CLK/Diff_wum_2019335_2019365_phb_Mean_STD'))
    cSer.append('ISL Rng')
    fSer.append(os.path.join(
        cDskPre0, r'PRO_2019001_2020366/I648/CLK/Diff_wum_2019335_2019365_phb_Mean_STD'))
    cSer.append('ISL Clk')
    fSer.append(os.path.join(
        cDskPre0, r'PRO_2019001_2020366/J642/CLK/Diff_wum_2019335_2019365_phb_Mean_STD'))
    cSer.append('ISL Rng+Clk')
    fSer.append(os.path.join(
        cDskPre0, r'PRO_2019001_2020366/D650/CLK/Diff_wum_2019335_2019365_phb_Mean_STD'))
    # cSer.append('12 cm')
    # fSer.append(os.path.join(
    #     cDskPre0, r'PRO_2019001_2020366/J647/CLK/Diff_gbm_2019335_2019365_phb_Mean_STD'))
    # cSer.append('15 cm')
    # fSer.append(os.path.join(
    #     cDskPre0, r'PRO_2019001_2020366/J648/CLK/Diff_gbm_2019335_2019365_phb_Mean_STD'))
    # cSer.append('18 cm')
    # fSer.append(os.path.join(
    #     cDskPre0, r'PRO_2019001_2020366/J651/CLK/Diff_gbm_2019335_2019365_phb_Mean_STD'))
    # cSer.append('20 cm')
    # fSer.append(os.path.join(
    #     cDskPre0, r'PRO_2019001_2020366/J653/CLK/Diff_gbm_2019335_2019365_phb_Mean_STD'))
    ClkList = ['C20', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28',
               'C29', 'C30', 'C32', 'C33', 'C34', 'C35', 'C36', 'C37']
    OutFileSuffix = 'Diff_wum_2019335_2019365_phb_STD_Comp'
    PlotClkDifRMS4(cSer, fSer, ClkList, [0, 0.5], OutFilePrefix, OutFileSuffix)

    # ClkPath1 = os.path.join(
    #     cWrkPre0, r'PRO_2019001_2020366_WORK/I060/2019/SATCLK_POD')
    ClkPath1 = os.path.join(cWrkPre0, r'MGEX/CLK/gbm_300s')
    # ClkPath1 = os.path.join(cWrkPre0, r'MGEX/CLK/wum')
    ClkPath2 = os.path.join(
        cWrkPre0, r'PRO_2019001_2020366_WORK/C01/2019/SATCLK_POD')
    # ClkPath2 = os.path.join(cWrkPre0, r'MGEX/CLK/wum')
    # ClkPath2 = os.path.join(cWrkPre0, r'MGEX/CLK/gbm_300s')

    ClkList = ['C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28',
               'C29', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35', 'C36', 'C37']

    # OutFileSuffix = 'Diff_wum_2019362_C36'
    # PlotClkDif50(2019, 362, 1, ClkPath1, 'wum', ClkPath2, 'phb',
    #              ['C36'], ['C21'], True, OutFilePrefix, OutFileSuffix)

    OutFileSuffix = 'Diff_gbm_2019335_2019365_phb'
    # PlotClkDif51(2019, 335, 31, ClkPath1, 'gbm', ClkPath2, 'phb',
    #              ['CXX'], ['C21', 'C20'], OutFilePrefix, OutFileSuffix)

    # InFilePrefix=r'E:/ISL/'
    # fISLList=glob.glob(InFilePrefix+'ISL_2019001')
    # OutFileSuffix='LinkClk_2019001.png'
    # cLink=ISL.PlotLinkClk(fISLList,['ALL-ALL'],True,False,1,0.0417,0.0417,OutFilePrefix,OutFileSuffix)

    # OutFileSuffix='wum_clkdif_2019001.png'
    # PlotClkDif20(fClkList,cLink,OutFilePrefix,OutFileSuffix)

    # InFilePrefix=r'D:/Code/PROJECT/WORK2019001_ERROR/'
    # fClkList=glob.glob(InFilePrefix+'clk_2019001')
    # OutFileSuffix='brd_clkdif_2019001.png'
    # PlotClkDif20(fClkList,cLink,OutFilePrefix,OutFileSuffix)
