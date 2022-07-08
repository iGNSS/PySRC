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
import datetime
import math

# Related third party imports
import numpy as np
from numpy import dtype, linalg as NupLA, matmul
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata
from scipy import linalg as SciLA

from astropy.timeseries import LombScargle

# Local application/library specific imports
from PySRC.miscellaneous import GNSSTime


def GetOmCISL(fOmcList, cSat1, cSat2, lAnisotropic):
    '''
    Extract ISL OmC (in cm) for links between cSat1 and cSat2.

    lAnisotropic --- Whether discriminate recv against tran end
    '''

    rEpo = []
    OmC = [[], []]
    for i in range(len(fOmcList)):
        with open(fOmcList[i], mode='rt') as fOb:
            for cLine in fOb:
                if 'ISL' not in cLine:
                    continue
                cWords = cLine.split()
                if lAnisotropic:
                    if cSat1 != 'ALL' and cSat1 != cWords[3]:
                        continue
                    if cSat2 != 'ALL' and cSat2 != cWords[4]:
                        continue
                else:
                    if cSat1 != 'ALL' and cSat1 not in cLine:
                        continue
                    if cSat2 != 'ALL' and cSat2 not in cLine:
                        continue
                rEpo.append(int(cWords[0]) + float(cWords[1]) / 86400.0)
                # Relative range && clock, cm
                OmC[0].append(float(cWords[11]) * 100)
                OmC[1].append(float(cWords[12]) * 100)

    x = np.array(rEpo)
    y = np.array(OmC)

    return x, y


def GetResISL(fResList, cSat1, cSat2, lAnisotropic):
    '''
    Extract ISL res for links between cSat1 and cSat2

        fResList --- residual file list
    lAnisotropic --- Whether discriminate recv against tran end
    '''

    rEpo = []
    Res = [[], [], [], [], [], []]
    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                if lAnisotropic:
                    if cSat1 != 'ALL' and cSat1 != cWords[2]:
                        continue
                    if cSat2 != 'ALL' and cSat2 != cWords[3]:
                        continue
                else:
                    if cSat1 != 'ALL' and (cSat1 != cWords[2] and cSat1 != cWords[3]):
                        continue
                    if cSat2 != 'ALL' and (cSat2 != cWords[2] and cSat2 != cWords[3]):
                        continue
                # epoch time, rMJD
                rEpo.append(int(cWords[14]) + float(cWords[15]) / 86400.0)
                if cSat1 == cWords[2] or cSat2 == cWords[3]:
                    # 'cSat1-ALL', 'ALL-cSat2' or 'cSat1-cSat2'
                    # Nadir for cSat1, cSat2
                    Res[1].append(float(cWords[5]))
                    Res[2].append(float(cWords[7]))
                    # Azim for cSat1, cSat2
                    Res[3].append(float(cWords[4]))
                    Res[4].append(float(cWords[6]))
                elif cSat1 == cWords[3] or cSat2 == cWords[2]:
                    # 'ALL-cSat1', 'cSat2-ALL' or 'cSat2-cSat1'
                    Res[1].append(float(cWords[7]))
                    Res[2].append(float(cWords[5]))
                    Res[3].append(float(cWords[6]))
                    Res[4].append(float(cWords[4]))
                else:
                    # 'ALL-ALL'
                    Res[1].append(float(cWords[5]))
                    Res[2].append(float(cWords[7]))
                    Res[3].append(float(cWords[4]))
                    Res[4].append(float(cWords[6]))
                # Relative range, m -> cm
                if cWords[8] == 'T':
                    Res[0].append(float(cWords[9]) * 100)
                else:
                    Res[0].append(np.nan)
                # Relative clock, m -> cm
                if cWords[11] == 'T':
                    Res[5].append(float(cWords[12]) * 100)
                else:
                    Res[5].append(np.nan)

    x = np.array(rEpo)
    y = np.array(Res)

    return x, y


def GetCorISL(fCor):
    '''
    Read the Lomb-Scargle model for ISL link correction, which was generated during the
    calculation of Lomb-Scargle periodogram of the ISL residuals.

    NOTE: Here we assume that for the Lomb-Scargle model, `nterm=2`
    '''

    cLink = []
    X = [[], [], [], [], [], [], []]
    with open(fCor, mode='rt') as fOb:
        lReach = False
        for cLine in fOb:
            if cLine[0:5] != '#Link' and (not lReach):
                continue
            elif cLine[0:5] == '#Link':
                lReach = True
            elif len(cLine) < 10:
                continue
            else:
                cWords = cLine.split()
                cLink.append(cWords[0])
                # Offset
                X[0].append(float(cWords[1]))
                # Frequency
                X[1].append(float(cWords[2]))
                # theta 0~4
                X[2].append(float(cWords[4]))
                X[3].append(float(cWords[5]))
                X[4].append(float(cWords[6]))
                X[5].append(float(cWords[7]))
                X[6].append(float(cWords[8]))
    return cLink, X


def GetResGNS(fResList, cSta0, cSat0, iObsType, rUnit):
    '''
    Extract GNS res of one specific obs type between cSta0 and cSat0

       cSta0 --- Specified station or 'ALL'
       cSat0 --- Specified satellite or 'ALL'
    iObsType --- Index of the obs type to be extracted, start from 1

    Return:
    '''

    cSta = []
    cSat = []
    Res = [[], [], []]

    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'RnxRES':
                    continue
                if cSta0 != 'ALL ' and cSta0 not in cLine:
                    continue
                if cSat0 != 'ALL' and cSat0 not in cLine:
                    continue
                cWords = cLine.split()
                if cWords[2] not in cSta:
                    cSta.append(cWords[2])
                if cWords[3] not in cSat:
                    cSat.append(cWords[3])
                # Number of obs types
                nObsType = int(cWords[8])
                # Only read valid observations (Flag<=2)
                Flag = int(cWords[8 + 3 * iObsType - 2])
                if Flag > 2:
                    continue
                # Epoch
                rMJD = float(cWords[9 + 3 * nObsType]) + \
                    float(cWords[10 + 3 * nObsType]) / 86400.0
                Res[0].append(rMJD)
                # Residual
                Res[1].append(float(cWords[8 + 3 * iObsType - 1]) * rUnit)
                # Elevation
                Res[2].append(float(cWords[5]))

    return np.array(Res), cSta, cSat


def PlotRes0(cSer, fSerList, OutFilePrefix, OutFileSuffix):
    '''
    Plot the post-priori sigma series for different solutions. The sigma
    of unit weight of one is taken as the benchmark for all solutions
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    nSer = len(cSer)

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(8, 4))
    # axs[0,0].set_yscale('log')

    Sig = []
    for i in range(nSer):
        # epoch and post-priori sigma
        Sig.append([])
        Sig.append([])
        nFile = len(fSerList[i])
        for j in range(nFile):
            # Get the date of this file from the file name
            YYYY = int(os.path.basename(fSerList[i][j])[4:8])
            DOY = int(os.path.basename(fSerList[i][j])[8:11])
            Sig[i * 2].append(GNSSTime.doy2mjd(YYYY, DOY))
            with open(fSerList[i][j], mode='rt') as fOb:
                for cLine in fOb:
                    if cLine[0:32] != 'Post-priori sigma of unit weight':
                        continue
                    else:
                        cWords = cLine.split(sep=':')
                        Sig[i * 2 + 1].append(float(cWords[1]))
                        break
        axs[0, 0].plot(Sig[i * 2], Sig[i * 2 + 1],
                       'o--', ms=3, lw=1, label=cSer[i])

    axs[0, 0].axhline(y=1, color='black', linestyle='-', lw=0.6, zorder=1.0)
    axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
    lg = axs[0, 0].legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5), framealpha=0.6,
                          prop={'family': 'Arial', 'size': 14})
    # # Get the width of the widest label, since every label will need to shift by this
    # # amount after we aligh to the right
    # rd=fig.canvas.get_renderer()
    # shift=max([lgt.get_window_extent(rd).width for lgt in lg.get_texts()])
    # for lgt in lg.get_texts():
    #     # lgt.set_ha('right')
    #     lgt.set_position((shift,0))

    # axs[0,0].ticklabel_format(axis='y',style='sci',useOffset=False,useMathText=True)
    # axs[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axs[0, 0].set_ylabel(r'Post-Priori $\sigma_0$'+' [m]',
                         fontname='Arial', fontsize=16)
    # axs[0,0].set_yticks([0.001,0.01,0.1,0.2,0.5,1,2,3,4,5])
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    axs[0, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
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


def PlotRes1(cSer, fSerList, OutFilePrefix, OutFileSuffix):
    '''
    Similar to PlotRes0, but the post-priori sigmas from the first solution
    are taken as the benchmark for all other solutions
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    nSer = len(cSer)

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(8, 4))
    # axs[0,0].set_yscale('log')

    for i in range(nSer):
        # epoch and post-priori sigma for this solution
        Sig = [[], []]
        nFile = len(fSerList[i])
        for j in range(nFile):
            # Get the date of this file from the file name
            YYYY = int(os.path.basename(fSerList[i][j])[4:8])
            DOY = int(os.path.basename(fSerList[i][j])[8:11])
            # Epoch in MJD
            Sig[0].append(GNSSTime.doy2mjd(YYYY, DOY))
            with open(fSerList[i][j], mode='rt') as fOb:
                for cLine in fOb:
                    if cLine[0:32] != 'Post-priori sigma of unit weight':
                        continue
                    else:
                        cWords = cLine.split(sep=':')
                        Sig[1].append(float(cWords[1]))
                        break
        if i > 0:
            Sig1 = np.array(Sig)
            # Cal the ratio to the first solution
            ind1 = np.argsort(Sig1[0])
            X = [[], []]
            for j in range(ind1.size):
                for k in range(ind0.size):
                    if (Sig1[0][ind1[j]] - Sig0[0][ind0[k]]) > 0.1:
                        continue
                    elif (Sig1[0][ind1[j]] - Sig0[0][ind0[k]]) < -0.1:
                        break
                    else:
                        X[0].append(Sig1[0][ind1[j]])
                        X[1].append(Sig1[1][ind1[j]] / Sig0[1][ind0[k]])
            axs[0, 0].plot(X[0], X[1], 'o--', ms=3, lw=0.8, label=cSer[i])
        else:
            # For the first solution
            Sig0 = np.array(Sig)
            # Sort for the first solution
            ind0 = np.argsort(Sig0[0])
            axe = axs[0, 0].twinx()
            axe.plot(Sig0[0][ind0], Sig0[1][ind0],
                     's--r', ms=3, lw=0.8, label=cSer[0])

    axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
    lg = axs[0, 0].legend(ncol=nSer - 1, loc='lower center', bbox_to_anchor=(0.5, 1.0), framealpha=0.6,
                          prop={'family': 'Arial', 'size': 14})
    axs[0, 0].set_ylabel(r'Ratio of Post-Priori $\sigma_0$',
                         fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    axe.set_ylabel(r'Post-Priori $\sigma_0$ [m]',
                   fontname='Arial', fontsize=16, color='r')
    for tl in axe.get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
        tl.set_color('r')

    axs[0, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
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


def PlotOmCISL1(fOmCList, lRelClk, lAnisotropic, OutFilePrefix, OutFileSuffix):
    '''
    Plot ISL OmC of all links one by one in a multi-page PDF file

         lRelClk --- Whether also plot for relative clock observations
    lAnisotropic --- Whether discriminate recv against tran end
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # Find all links
    cLink = []
    # Get the start && end epoch
    rMJD1 = 99999
    rMJD2 = -99999
    for i in range(len(fOmCList)):
        with open(fOmCList[i], mode='rt') as fOb:
            for cLine in fOb:
                if 'ISL' not in cLine:
                    continue
                cWords = cLine.split()
                rMJD = int(cWords[0]) + float(cWords[1])/86400
                if rMJD < rMJD1:
                    rMJD1 = rMJD
                if rMJD > rMJD2:
                    rMJD2 = rMJD
                if lAnisotropic:
                    cTmp = cWords[3] + '-' + cWords[4]
                else:
                    if cWords[3] > cWords[4]:
                        cTmp = cWords[4] + '-' + cWords[3]
                    else:
                        cTmp = cWords[3] + '-' + cWords[4]
                if cTmp not in cLink:
                    cLink.append(cTmp)
    cLink.sort()
    nLink = len(cLink)
    rMJD1 = np.floor(rMJD1)
    rMJD2 = np.ceil(rMJD2)

    strTmp = os.path.join(OutFilePrefix, OutFileSuffix + '.pdf')
    mpl.use('Agg')
    with PdfPages(strTmp) as pdf:
        for i in range(nLink):
            rEpo, OmC = GetOmCISL(
                fOmCList, cLink[i][0:3], cLink[i][4:7], lAnisotropic)
            nOmC = OmC[0].size

            # Get the mean (in meter) and std (in cm)
            Mea0 = np.mean(OmC[0]) / 100
            Sig0 = np.std(OmC[0])

            fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 1.5))
            axs[0, 0].set_xlim(left=rMJD1, right=rMJD2)
            # Remove the mean
            axs[0, 0].plot(rEpo, OmC[0] - Mea0 * 100, '.r', ms=2)
            axs[0, 0].text(0.98, 0.98, '{:>7.1f}+/-{:>6.1f}'.format(Mea0, Sig0),
                           transform=axs[0, 0].transAxes, ha='right', va='top',
                           fontdict={'fontsize': 14, 'fontname': 'Arial'})
            if lRelClk:
                Mea1 = np.mean(OmC[1]) / 100
                Sig1 = np.std(OmC[1])
                axs[0, 0].plot(rEpo, OmC[1] - Mea1 * 100, '.g', ms=2)
                axs[0, 0].text(0.98, 0.02, '{:>7.1f}+/-{:>6.1f}'.format(Mea1, Sig1),
                               transform=axs[0, 0].transAxes, ha='right', va='bottom',
                               fontdict={'fontsize': 14, 'fontname': 'Arial'})
            axs[0, 0].set_ylabel('[cm]', fontname='Arial', fontsize=16)
            for tl in axs[0, 0].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            axs[0, 0].text(0.05, 0.95, cLink[i], transform=axs[0, 0].transAxes, ha='left', va='top',
                           fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
            axs[0, 0].set_xlabel('Modified Julian Day',
                                 fontname='Arial', fontsize=16)
            axs[0, 0].xaxis.set_major_formatter('{x:7.1f}')
            for tl in axs[0, 0].get_xticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def PlotResISL00(fResList, lEpo, cLink0, iPlot, OutFilePrefix, OutFileSuffix):
    '''
    Plot ISL range, clock or both residual series of specific links one by one,
    i.e., each link takes an axis but all links share the figure.

      lEpo --- Whether plot residuals VS epoch time or VS nadir angle
    cLink0 --- List of specified ISL links, set cLink0[0]='ALL-ALL' to plot all links
     iPlot --- Info to plot
               # 0, range
               # 1, clock
               # 2, both
    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    cLink = []
    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                if cWords[8] == 'F' and cWords[11] == 'F':
                    continue
                if cWords[3] > cWords[2]:
                    cTmp = cWords[2] + '-' + cWords[3]
                else:
                    cTmp = cWords[3] + '-' + cWords[2]
                if cLink0[0] != 'ALL-ALL' and cTmp not in cLink0:
                    continue
                if cTmp not in cLink:
                    cLink.append(cTmp)
    cLink.sort()
    nLink = len(cLink)

    if lEpo:
        fig, axs = plt.subplots(nLink, 1, sharex='col',
                                squeeze=False, figsize=(12, nLink * 1.5))
    else:
        fig, axs = plt.subplots(nLink, 1, squeeze=False,
                                figsize=(12, nLink * 1.5))
    # fig.subplots_adjust(hspace=0.3)

    for i in range(nLink):
        rEpo, Res = GetResISL(fResList, cLink[i][0:3], cLink[i][4:7], False)
        nRes = Res[0].size
        if nRes == 0:
            print('Link ' + cLink[i] + ' not found!')
            continue
        # Get the mean, std && rms for range and clock respectively
        nObs1 = 0
        RMS1 = 0.0
        nObs2 = 0
        RMS2 = 0.0
        for j in range(nRes):
            if not np.isnan(Res[0, j]):
                nObs1 = nObs1 + 1
                RMS1 = RMS1 + Res[0, j] * Res[0, j]
            if not np.isnan(Res[5, j]):
                nObs2 = nObs2 + 1
                RMS2 = RMS2 + Res[5, j] * Res[5, j]
        if nObs1 > 0:
            Mea1 = np.nanmean(Res[0])
            Sig1 = np.nanstd(Res[0])
            RMS1 = np.sqrt(RMS1 / nObs1)
        if nObs2 > 0:
            Mea2 = np.nanmean(Res[5])
            Sig2 = np.nanstd(Res[5])
            RMS2 = np.sqrt(RMS2 / nObs2)

        # axs[i,0].set_ylim(bottom=-25,top=25)
        if lEpo:
            # residual VS epoch time
            if iPlot == 0 or iPlot == 2:
                axs[i, 0].plot(rEpo, Res[0], '.r', ms=2, label='Range')
            if iPlot == 1 or iPlot == 2:
                axs[i, 0].plot(rEpo, Res[5], '.b', ms=2, label='Clock')
        else:
            # residual VS nadir
            if iPlot == 0 or iPlot == 2:
                axs[i, 0].plot(Res[1], Res[0], '.r', ms=2, label='Range')
            if iPlot == 1 or iPlot == 2:
                axs[i, 0].plot(Res[1], Res[5], '.b', ms=2, label='Clock')
        axs[i, 0].axhline(color='darkgray', linestyle='dashed', alpha=0.5)
        axs[i, 0].text(0.02, 0.98, cLink[i], transform=axs[i, 0].transAxes, ha='left', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
        if nObs1 > 0 and (iPlot == 0 or iPlot == 2):
            strTmp = '{:>7.1f}+/-{:>6.1f}, RMS={:>7.1f}, #={:>6d}'.format(
                Mea1, Sig1, RMS1, nObs1)
            axs[i, 0].text(0.98, 0.98, strTmp, transform=axs[i, 0].transAxes, ha='right', va='top',
                           fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkred'})
        if nObs2 > 0 and (iPlot == 1 or iPlot == 2):
            strTmp = '{:>7.1f}+/-{:>6.1f}, RMS={:>7.1f}, #={:>6d}'.format(
                Mea2, Sig2, RMS2, nObs2)
            axs[i, 0].text(0.98, 0.02, strTmp, transform=axs[i, 0].transAxes, ha='right', va='bottom',
                           fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkblue'})
        axs[i, 0].set_ylabel('[cm]', fontname='Arial', fontsize=16)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
    if lEpo:
        axs[i, 0].set_xlabel('Modified Julian Day',
                             fontname='Arial', fontsize=16)
    else:
        axs[i, 0].set_xlabel('Nadir angle [deg]',
                             fontname='Arial', fontsize=16)
    axs[i, 0].xaxis.set_major_formatter('{x:7.1f}')
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix + OutFileSuffix + '.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResISL01(fResList, lEpo, cLink0, lAnisotropic, iPlot, OutFilePrefix, OutFileSuffix):
    '''
    Plot ISL range, clock or both residual series of specific links one by one,
    i.e., each link takes an axis but all links share a multi-page PDF.

        lEpo --- Whether plot residuals VS epoch time or VS nadir angle
      cLink0 --- List of specified ISL links, set cLink0[0]='ALL-ALL' to plot all links
lAnisotropic --- Whether discriminate recv against tran end
       iPlot --- Info to plot
                 # 0, range
                 # 1, clock
                 # 2, both
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    cLink = []
    # Get the start && end epoch
    rMJD1 = 99999
    rMJD2 = -99999
    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                rMJD = int(cWords[14]) + float(cWords[15]) / 86400
                if rMJD < rMJD1:
                    rMJD1 = rMJD
                if rMJD > rMJD2:
                    rMJD2 = rMJD
                if cWords[8] == 'F' and cWords[11] == 'F':
                    continue
                if lAnisotropic:
                    cTmp = cWords[2] + '-' + cWords[3]
                    if cLink0[0] != 'ALL-ALL':
                        if cLink0[0][0:3] == 'ALL':
                            if cLink0[0][4:7] != cWords[3]:
                                continue
                        elif cLink0[0][4:7] == 'ALL':
                            if cLink0[0][0:3] != cWords[2]:
                                continue
                        else:
                            if cTmp not in cLink0:
                                continue
                else:
                    if cWords[3] > cWords[2]:
                        cTmp = cWords[2] + '-' + cWords[3]
                    else:
                        cTmp = cWords[3] + '-' + cWords[2]
                    if cLink0[0] != 'ALL-ALL':
                        if cLink0[0][0:3] == 'ALL':
                            if cLink0[0][4:7] != cWords[2] and cLink0[0][4:7] != cWords[3]:
                                continue
                        elif cLink0[0][4:7] == 'ALL':
                            if cLink0[0][0:3] != cWords[2] and cLink0[0][0:3] != cWords[3]:
                                continue
                        else:
                            cTmp1 = cWords[2] + '-' + cWords[3]
                            cTmp2 = cWords[3] + '-' + cWords[2]
                            if (cTmp1 not in cLink0) and (cTmp2 not in cLink0):
                                continue
                if cTmp not in cLink:
                    cLink.append(cTmp)
    cLink.sort()
    nLink = len(cLink)

    strTmp = os.path.join(OutFilePrefix, OutFileSuffix + '.pdf')
    mpl.use('Agg')
    with PdfPages(strTmp) as pdf:
        for i in range(nLink):
            rEpo, Res = GetResISL(
                fResList, cLink[i][0:3], cLink[i][4:7], lAnisotropic)
            nRes = Res[0].size
            if nRes == 0:
                print('Link ' + cLink[i] + ' not found!')
                continue
            # Get the mean, std && rms for range and clock respectively
            nObs1 = 0
            RMS1 = 0.0
            nObs2 = 0
            RMS2 = 0.0
            for j in range(nRes):
                if not np.isnan(Res[0, j]):
                    nObs1 = nObs1 + 1
                    RMS1 = RMS1 + Res[0, j] * Res[0, j]
                if not np.isnan(Res[5, j]):
                    nObs2 = nObs2 + 1
                    RMS2 = RMS2 + Res[5, j] * Res[5, j]
            if nObs1 > 0:
                Mea1 = np.nanmean(Res[0])
                Sig1 = np.nanstd(Res[0])
                RMS1 = np.sqrt(RMS1 / nObs1)
            if nObs2 > 0:
                Mea2 = np.nanmean(Res[5])
                Sig2 = np.nanstd(Res[5])
                RMS2 = np.sqrt(RMS2 / nObs2)

            fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 1.5))
            # axs[0,0].set_ylim(bottom=-25,top=25)
            if lEpo:
                # residual VS epoch time
                if iPlot == 0 or iPlot == 2:
                    axs[0, 0].plot(rEpo, Res[0], '.r', ms=2, label='Range')
                if iPlot == 1 or iPlot == 2:
                    axs[0, 0].plot(rEpo, Res[5], '.b', ms=2, label='Clock')
                axs[0, 0].set_xlim(left=rMJD1, right=rMJD2)
            else:
                # residual VS nadir
                if iPlot == 0 or iPlot == 2:
                    axs[0, 0].plot(Res[1], Res[0], '.r', ms=2, label='Range')
                if iPlot == 1 or iPlot == 2:
                    axs[0, 0].plot(Res[1], Res[5], '.b', ms=2, label='Clock')
            # axs[0, 0].axhline(color='darkgray', linestyle='dashed', alpha=0.5)
            axs[0, 0].text(0.02, 0.98, cLink[i], transform=axs[0, 0].transAxes, ha='left', va='top',
                           fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
            if nObs1 > 0 and (iPlot == 0 or iPlot == 2):
                strTmp = '{:>7.1f}+/-{:>6.1f}, RMS={:>7.1f}, #={:>6d}'.format(
                    Mea1, Sig1, RMS1, nObs1)
                axs[0, 0].text(0.98, 0.98, strTmp, transform=axs[0, 0].transAxes, ha='right', va='top',
                               fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkred'})
            if nObs2 > 0 and (iPlot == 1 or iPlot == 2):
                strTmp = '{:>7.1f}+/-{:>6.1f}, RMS={:>7.1f}, #={:>6d}'.format(
                    Mea2, Sig2, RMS2, nObs2)
                axs[0, 0].text(0.98, 0.02, strTmp, transform=axs[0, 0].transAxes, ha='right', va='bottom',
                               fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkblue'})
            axs[0, 0].set_ylabel('[cm]', fontname='Arial', fontsize=16)
            for tl in axs[0, 0].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            if lEpo:
                axs[0, 0].set_xlabel('Modified Julian Day',
                                     fontname='Arial', fontsize=16)
                axs[0, 0].xaxis.set_major_formatter('{x:7.1f}')
            else:
                axs[0, 0].set_xlabel('Nadir angle [deg]',
                                     fontname='Arial', fontsize=16)
                axs[0, 0].xaxis.set_major_formatter('{x:4.1f}')
            for tl in axs[0, 0].get_xticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def PlotResISL10(fLogList, cLink0, iPlot, OutFilePrefix, OutFileSuffix):
    '''
    Plot ISL link corrections with the model apllied during reading ISL data into
    the lsq estimator. Those corrections are output to the lsq log files.

    cLink0 --- List of specified ISL links, set cLink0[0]='ALL-ALL' to plot all links
     iPlot --- Info to plot
               # 0, range
               # 1, clock
               # 2, both
    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    cLink = []
    for i in range(len(fLogList)):
        with open(fLogList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:19] != '***TIP(ReadObsISL):':
                    continue
                cWords = cLine.split()
                if iPlot == 0 and int(cWords[4]) != 1:
                    # Only range
                    continue
                if iPlot == 1 and int(cWords[4]) != 2:
                    # Only clock
                    continue
                if cWords[5][0:3] > cWords[5][4:7]:
                    cTmp = cWords[5][4:7] + '-' + cWords[5][0:3]
                else:
                    cTmp = cWords[5][0:3] + '-' + cWords[5][4:7]
                if cLink0[0] != 'ALL-ALL' and cTmp not in cLink0:
                    continue
                if cTmp not in cLink:
                    cLink.append(cTmp)
    cLink.sort()
    nLink = len(cLink)

    fig, axs = plt.subplots(nLink, 1, sharex='col',
                            squeeze=False, figsize=(12, nLink * 1.5))
    for i in range(nLink):
        # Read the corr for each link
        Res = [[], [], [], []]
        for j in range(len(fLogList)):
            with open(fLogList[j], mode='rt') as fOb:
                for cLine in fOb:
                    if cLine[0:19] != '***TIP(ReadObsISL):':
                        continue
                    cWords = cLine.split()
                    if iPlot == 0 and int(cWords[4]) != 1:
                        # Only range
                        continue
                    if iPlot == 1 and int(cWords[4]) != 2:
                        # Only clock
                        continue
                    if cWords[5][0:3] + '-' + cWords[5][4:7] != cLink[i] and \
                            cWords[5][4:7] + '-' + cWords[5][0:3] != cLink[i]:
                        continue
                    if int(cWords[4]) == 1:
                        # Epoch, rMJD
                        Res[0].append(int(cWords[6]) +
                                      float(cWords[7]) / 86400.0)
                        # Range correction, in cm
                        Res[1].append(float(cWords[9]))
                    else:
                        # Clock correction, in cm
                        Res[2].append(int(cWords[6]) +
                                      float(cWords[7]) / 86400.0)
                        # Range correction, in cm
                        Res[3].append(float(cWords[9]))
        # residual VS epoch time
        if iPlot == 0 or iPlot == 2:
            axs[i, 0].plot(Res[0], Res[1], '.r', ms=2, label='Range')
        if iPlot == 1 or iPlot == 2:
            axs[i, 0].plot(Res[2], Res[3], '.b', ms=2, label='Clock')

        axs[i, 0].axhline(color='darkgray', linestyle='dashed', alpha=0.5)
        axs[i, 0].text(0.02, 0.98, cLink[i], transform=axs[i, 0].transAxes, ha='left', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})

        axs[i, 0].set_ylabel('[cm]', fontname='Arial', fontsize=16)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

    axs[i, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)

    axs[i, 0].xaxis.set_major_formatter('{x:7.1f}')
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix + OutFileSuffix + '.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResISL20(fResList, lEpo, cLink0, lAnisotropic, iPlot, OutFilePrefix, OutFileSuffix):
    '''
    Plot ISL range, clock or both residual series of specific links within a single axis

        lEpo --- Whether plot residuals VS epoch time or VS nadir angle
      cLink0 --- Specified ISL links, set cLink0[0]='ALL-ALL' to plot all links
lAnisotropic --- Whether discriminate recv against tran end
       iPlot --- Info to plot
                 # 0, range
                 # 1, clock
                 # 2, both
    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    cLink = []
    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                if cWords[8] == 'F' and cWords[11] == 'F':
                    continue
                if lAnisotropic:
                    cTmp = cWords[2] + '-' + cWords[3]
                    if cLink0[0] != 'ALL-ALL':
                        if cLink0[0][0:3] == 'ALL':
                            if cLink0[0][4:7] != cWords[3]:
                                continue
                        elif cLink0[0][4:7] == 'ALL':
                            if cLink0[0][0:3] != cWords[2]:
                                continue
                        else:
                            if cTmp not in cLink0:
                                continue
                else:
                    if cWords[3] > cWords[2]:
                        cTmp = cWords[2] + '-' + cWords[3]
                    else:
                        cTmp = cWords[3] + '-' + cWords[2]
                    if cLink0[0] != 'ALL-ALL':
                        if cLink0[0][0:3] == 'ALL':
                            if cLink0[0][4:7] != cWords[2] and cLink0[0][4:7] != cWords[3]:
                                continue
                        elif cLink0[0][4:7] == 'ALL':
                            if cLink0[0][0:3] != cWords[2] and cLink0[0][0:3] != cWords[3]:
                                continue
                        else:
                            cTmp1 = cWords[2] + '-' + cWords[3]
                            cTmp2 = cWords[3] + '-' + cWords[2]
                            if (cTmp1 not in cLink0) and (cTmp2 not in cLink0):
                                continue
                if cTmp not in cLink:
                    cLink.append(cTmp)
    nLink = len(cLink)

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 3))

    for i in range(nLink):
        if i == 0:
            rEpo, Res = GetResISL(
                fResList, cLink[i][0:3], cLink[i][4:7], lAnisotropic)
        else:
            rEpo0, Res0 = GetResISL(
                fResList, cLink[i][0:3], cLink[i][4:7], lAnisotropic)
            rEpo = np.append(rEpo, rEpo0, axis=0)
            Res = np.append(Res, Res0, axis=1)

    nRes = Res[0].size
    # Get the mean, std && rms for range and clock respectively
    nObs1 = 0
    RMS1 = 0.0
    nObs2 = 0
    RMS2 = 0.0
    for j in range(nRes):
        if not np.isnan(Res[0, j]):
            nObs1 = nObs1 + 1
            RMS1 = RMS1 + Res[0, j] * Res[0, j]
        if not np.isnan(Res[5, j]):
            nObs2 = nObs2 + 1
            RMS2 = RMS2 + Res[5, j] * Res[5, j]
    if nObs1 > 0:
        Mea1 = np.nanmean(Res[0])
        Sig1 = np.nanstd(Res[0])
        RMS1 = np.sqrt(RMS1 / nObs1)
    if nObs2 > 0:
        Mea2 = np.nanmean(Res[5])
        Sig2 = np.nanstd(Res[5])
        RMS2 = np.sqrt(RMS2 / nObs2)

    if lEpo:
        # residual VS epoch time
        if iPlot == 0 or iPlot == 2:
            axs[0, 0].plot(rEpo, Res[0], '.r', ms=2, label='Range')
        if iPlot == 1 or iPlot == 2:
            axs[0, 0].plot(rEpo, Res[5], '.b', ms=2, label='Clock')
    else:
        # residual VS nadir
        if iPlot == 0 or iPlot == 2:
            axs[0, 0].plot(Res[1], Res[0], '.r', ms=2, label='Range')
        if iPlot == 1 or iPlot == 2:
            axs[0, 0].plot(Res[1], Res[5], '.b', ms=2, label='Clock')
    if nObs1 > 0 and (iPlot == 0 or iPlot == 2):
        strTmp = '{:>7.1f}+/-{:>6.1f}, RMS={:>7.1f}, #={:>6d}'.format(
            Mea1, Sig1, RMS1, nObs1)
        axs[0, 0].text(0.98, 0.98, strTmp, transform=axs[0, 0].transAxes, ha='right', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkred'})
    if nObs2 > 0 and (iPlot == 1 or iPlot == 2):
        strTmp = '{:>7.1f}+/-{:>6.1f}, RMS={:>7.1f}, #={:>6d}'.format(
            Mea2, Sig2, RMS2, nObs2)
        axs[0, 0].text(0.98, 0.02, strTmp, transform=axs[0, 0].transAxes, ha='right', va='bottom',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkblue'})
    axs[0, 0].set_ylabel('[cm]', fontname='Arial', fontsize=16)
    # axs[0,0].axhline(color='darkgray',linestyle='dashed',alpha=0.5,)
    axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
    axs[0, 0].set_axisbelow(True)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    if lEpo:
        axs[0, 0].set_xlabel('Modified Julian Day',
                             fontname='Arial', fontsize=16)
    else:
        axs[0, 0].set_xlabel('Nadir angle [deg]',
                             fontname='Arial', fontsize=16)
    axs[0, 0].xaxis.set_major_formatter('{x:7.1f}')
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


def PlotResISL21(fResList, lEpo, cLink0, iPlot, OutFilePrefix, OutFileSuffix):
    '''
    Similar as PlotResISL20, but plot for each file
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    strTmp = os.path.join(OutFilePrefix, OutFileSuffix + '.pdf')
    with PdfPages(strTmp) as pdf:
        for fRes in fResList:
            cLink = []
            with open(fRes, mode='rt') as fOb:
                for cLine in fOb:
                    if cLine[0:6] != 'ISLRES':
                        continue
                    cWords = cLine.split()
                    if cWords[8] == 'F' and cWords[11] == 'F':
                        continue
                    if cWords[3] > cWords[2]:
                        cTmp = cWords[2] + '-' + cWords[3]
                    else:
                        cTmp = cWords[3] + '-' + cWords[2]
                    if cLink0[0] != 'ALL-ALL' and cTmp not in cLink0:
                        continue
                    if cTmp not in cLink:
                        cLink.append(cTmp)
            nLink = len(cLink)

            fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 3))
            axs[0, 0].text(0.50, 1.00, fRes, transform=axs[0, 0].transAxes, ha='center', va='bottom',
                           fontdict={'fontsize': 10, 'fontname': 'Arial', 'fontweight': 'bold'})

            for i in range(nLink):
                if i == 0:
                    rEpo, Res = GetResISL(
                        [fRes], cLink[i][0:3], cLink[i][4:7], False)
                else:
                    rEpo0, Res0 = GetResISL(
                        [fRes], cLink[i][0:3], cLink[i][4:7], False)
                    rEpo = np.append(rEpo, rEpo0, axis=0)
                    Res = np.append(Res, Res0, axis=1)
            nRes = Res[0].size
            # Get the mean, std && rms for range and clock respectively
            nObs1 = 0
            RMS1 = 0.0
            nObs2 = 0
            RMS2 = 0.0
            for j in range(nRes):
                if not np.isnan(Res[0, j]):
                    nObs1 = nObs1 + 1
                    RMS1 = RMS1 + Res[0, j] * Res[0, j]
                if not np.isnan(Res[5, j]):
                    nObs2 = nObs2 + 1
                    RMS2 = RMS2 + Res[5, j] * Res[5, j]
            if nObs1 > 0:
                Mea1 = np.nanmean(Res[0])
                Sig1 = np.nanstd(Res[0])
                RMS1 = np.sqrt(RMS1 / nObs1)
            if nObs2 > 0:
                Mea2 = np.nanmean(Res[5])
                Sig2 = np.nanstd(Res[5])
                RMS2 = np.sqrt(RMS2 / nObs2)

            if lEpo:
                # residual VS epoch time
                if iPlot == 0 or iPlot == 2:
                    axs[0, 0].plot(rEpo, Res[0], '.r', ms=2, label='Range')
                if iPlot == 1 or iPlot == 2:
                    axs[0, 0].plot(rEpo, Res[5], '.b', ms=2, label='Clock')
            else:
                # residual VS nadir
                if iPlot == 0 or iPlot == 2:
                    axs[0, 0].plot(Res[1], Res[0], '.r', ms=2, label='Range')
                if iPlot == 1 or iPlot == 2:
                    axs[0, 0].plot(Res[1], Res[5], '.b', ms=2, label='Clock')
            if nObs1 > 0 and (iPlot == 0 or iPlot == 2):
                strTmp = '{:>7.1f}+/-{:>6.1f}, RMS={:>7.1f}, #={:>6d}'.format(
                    Mea1, Sig1, RMS1, nObs1)
                axs[0, 0].text(0.98, 0.98, strTmp, transform=axs[0, 0].transAxes, ha='right', va='top',
                               fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkred'})
            if nObs2 > 0 and (iPlot == 1 or iPlot == 2):
                strTmp = '{:>7.1f}+/-{:>6.1f}, RMS={:>7.1f}, #={:>6d}'.format(
                    Mea2, Sig2, RMS2, nObs2)
                axs[0, 0].text(0.98, 0.02, strTmp, transform=axs[0, 0].transAxes, ha='right', va='bottom',
                               fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold',
                                         'color': 'darkblue'})
            axs[0, 0].set_ylabel('[cm]', fontname='Arial', fontsize=16)
            # axs[0,0].axhline(color='darkgray',linestyle='dashed',alpha=0.5)
            axs[0, 0].grid(which='major', axis='y',
                           color='darkgray', linestyle='--', linewidth=0.4)
            axs[0, 0].set_axisbelow(True)
            for tl in axs[0, 0].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            if lEpo:
                axs[0, 0].set_xlabel('Modified Julian Day',
                                     fontname='Arial', fontsize=16)
            else:
                axs[0, 0].set_xlabel('Nadir angle [deg]',
                                     fontname='Arial', fontsize=16)
            axs[0, 0].xaxis.set_major_formatter('{x:7.1f}')
            for tl in axs[0, 0].get_xticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def PlotResISL22(fResList, cLink0, iPlot, OutFilePrefix, OutFileSuffix):
    '''
    Plot histogram for ISL range and clock residuals of specific links in a single axis

    cLink0 --- Specified ISL links, set cLink0[0]='ALL-ALL' to plot all links
     iPlot --- Info to plot
               # 0, range
               # 1, clock
               # 2, both
    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    cLink = []
    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                if cWords[8] == 'F' and cWords[11] == 'F':
                    continue
                if cWords[3] > cWords[2]:
                    cTmp = cWords[2] + '-' + cWords[3]
                else:
                    cTmp = cWords[3] + '-' + cWords[2]
                if cLink0[0] != 'ALL-ALL' and cTmp not in cLink0:
                    continue
                if cTmp not in cLink:
                    cLink.append(cTmp)
    nLink = len(cLink)

    for i in range(nLink):
        if i == 0:
            rEpo, Res = GetResISL(
                fResList, cLink[i][0:3], cLink[i][4:7], False)
        else:
            rEpo0, Res0 = GetResISL(
                fResList, cLink[i][0:3], cLink[i][4:7], False)
            rEpo = np.append(rEpo, rEpo0, axis=0)
            Res = np.append(Res, Res0, axis=1)

    nRes = Res[0].size
    # Get the mean, std && rms for range and clock respectively
    nObs1 = 0
    RMS1 = 0.0
    nObs2 = 0
    RMS2 = 0.0
    for j in range(nRes):
        if not np.isnan(Res[0, j]):
            nObs1 = nObs1 + 1
            RMS1 = RMS1 + Res[0, j] * Res[0, j]
        if not np.isnan(Res[5, j]):
            nObs2 = nObs2 + 1
            RMS2 = RMS2 + Res[5, j] * Res[5, j]
    if nObs1 > 0:
        Mea1 = np.nanmean(Res[0])
        Sig1 = np.nanstd(Res[0])
        RMS1 = np.sqrt(RMS1 / nObs1)
    if nObs2 > 0:
        Mea2 = np.nanmean(Res[5])
        Sig2 = np.nanstd(Res[5])
        RMS2 = np.sqrt(RMS2 / nObs2)

    if iPlot == 0:
        # Range
        fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(6, 3))
        n, bins, pat = axs[0, 0].hist(Res[0], bins=100, density=True)
        if nObs1 > 0:
            y = ((1 / (np.sqrt(2 * np.pi) * Sig1)) *
                 np.exp(-0.5 * (1 / Sig1 * (bins - Mea1)) ** 2))
            axs[0, 0].plot(bins, y, 'r--', lw=1)
            strTmp = '{:>7.1f}+/-{:>6.1f}'.format(Mea1, Sig1)
            axs[0, 0].text(0.98, 0.98, strTmp, transform=axs[0, 0].transAxes, ha='right', va='top',
                           fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkred'})

        axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
        axs[0, 0].set_axisbelow(True)

        axs[0, 0].set_ylabel('Density', fontname='Arial', fontsize=16)
        for tl in axs[0, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        axs[0, 0].set_xlabel('Residuals [cm]', fontname='Arial', fontsize=16)
        for tl in axs[0, 0].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

    elif iPlot == 1:
        # Clock
        fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(6, 3))
        n, bins, pat = axs[0, 0].hist(Res[5], bins=100, density=True)
        if nObs2 > 0:
            y = ((1 / (np.sqrt(2 * np.pi) * Sig2)) *
                 np.exp(-0.5 * (1 / Sig2 * (bins - Mea2)) ** 2))
            axs[0, 0].plot(bins, y, 'r--', lw=1)
            strTmp = '{:>7.1f}+/-{:>6.1f}'.format(Mea2, Sig2)
            axs[0, 0].text(0.98, 0.98, strTmp, transform=axs[0, 0].transAxes, ha='right', va='top',
                           fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkred'})

        axs[0, 0].grid(which='major', axis='y', color='darkgray',
                       linestyle='--', linewidth=0.4)
        axs[0, 0].set_axisbelow(True)

        axs[0, 0].set_ylabel('Density', fontname='Arial', fontsize=16)
        for tl in axs[0, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        axs[0, 0].set_xlabel('Residuals [cm]', fontname='Arial', fontsize=16)
        for tl in axs[0, 0].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

    else:
        # Range && Clock
        sys.exit('Not supported yet!')

    strTmp = OutFilePrefix + OutFileSuffix + '.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResISL23(fResList, cLink0, iPlot, OutFilePrefix, OutFileSuffix):
    '''
    Similar to PlotResISL22, but plot for each file
    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    strTmp = os.path.join(OutFilePrefix, OutFileSuffix + '.pdf')
    with PdfPages(strTmp) as pdf:
        for fRes in fResList:
            cLink = []
            with open(fRes, mode='rt') as fOb:
                for cLine in fOb:
                    if cLine[0:6] != 'ISLRES':
                        continue
                    cWords = cLine.split()
                    if cWords[8] == 'F' and cWords[11] == 'F':
                        continue
                    if cWords[3] > cWords[2]:
                        cTmp = cWords[2] + '-' + cWords[3]
                    else:
                        cTmp = cWords[3] + '-' + cWords[2]
                    if cLink0[0] != 'ALL-ALL' and cTmp not in cLink0:
                        continue
                    if cTmp not in cLink:
                        cLink.append(cTmp)
            nLink = len(cLink)

            for i in range(nLink):
                if i == 0:
                    rEpo, Res = GetResISL(
                        [fRes], cLink[i][0:3], cLink[i][4:7], False)
                else:
                    rEpo0, Res0 = GetResISL(
                        [fRes], cLink[i][0:3], cLink[i][4:7], False)
                    rEpo = np.append(rEpo, rEpo0, axis=0)
                    Res = np.append(Res, Res0, axis=1)

            nRes = Res[0].size
            # Get the mean, std && rms for range and clock respectively
            nObs1 = 0
            RMS1 = 0.0
            nObs2 = 0
            RMS2 = 0.0
            for j in range(nRes):
                if not np.isnan(Res[0, j]):
                    nObs1 = nObs1 + 1
                    RMS1 = RMS1 + Res[0, j] * Res[0, j]
                if not np.isnan(Res[5, j]):
                    nObs2 = nObs2 + 1
                    RMS2 = RMS2 + Res[5, j] * Res[5, j]
            if nObs1 > 0:
                Mea1 = np.nanmean(Res[0])
                Sig1 = np.nanstd(Res[0])
                RMS1 = np.sqrt(RMS1 / nObs1)
            if nObs2 > 0:
                Mea2 = np.nanmean(Res[5])
                Sig2 = np.nanstd(Res[5])
                RMS2 = np.sqrt(RMS2 / nObs2)

            if iPlot == 0:
                # Range
                fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(6, 3))
                axs[0, 0].text(0.50, 1.00, fRes, transform=axs[0, 0].transAxes, ha='center', va='bottom',
                               fontdict={'fontsize': 10, 'fontname': 'Arial', 'fontweight': 'bold'})
                n, bins, pat = axs[0, 0].hist(Res[0], bins=100, density=True)
                if nObs1 > 0:
                    y = ((1 / (np.sqrt(2 * np.pi) * Sig1)) *
                         np.exp(-0.5 * (1 / Sig1 * (bins - Mea1)) ** 2))
                    axs[0, 0].plot(bins, y, 'r--', lw=1)
                    strTmp = '{:>7.1f}+/-{:>6.1f}'.format(Mea1, Sig1)
                    axs[0, 0].text(0.98, 0.98, strTmp, transform=axs[0, 0].transAxes, ha='right', va='top',
                                   fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold',
                                             'color': 'darkred'})

                axs[0, 0].grid(which='major', axis='y',
                               color='darkgray', linestyle='--', linewidth=0.4)
                axs[0, 0].set_axisbelow(True)

                axs[0, 0].set_ylabel('Density', fontname='Arial', fontsize=16)
                for tl in axs[0, 0].get_yticklabels():
                    tl.set_fontname('Arial')
                    tl.set_fontsize(14)

                axs[0, 0].set_xlabel(
                    'Residuals [cm]', fontname='Arial', fontsize=16)
                for tl in axs[0, 0].get_xticklabels():
                    tl.set_fontname('Arial')
                    tl.set_fontsize(14)

            elif iPlot == 1:
                # Clock
                fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(6, 3))
                axs[0, 0].text(0.50, 1.00, fRes, transform=axs[0, 0].transAxes, ha='center', va='bottom',
                               fontdict={'fontsize': 10, 'fontname': 'Arial', 'fontweight': 'bold'})
                n, bins, pat = axs[0, 0].hist(Res[5], bins=100, density=True)
                if nObs2 > 0:
                    y = ((1 / (np.sqrt(2 * np.pi) * Sig2)) *
                         np.exp(-0.5 * (1 / Sig2 * (bins - Mea2)) ** 2))
                    axs[0, 0].plot(bins, y, 'r--', lw=1)
                    strTmp = '{:>7.1f}+/-{:>6.1f}'.format(Mea2, Sig2)
                    axs[0, 0].text(0.98, 0.98, strTmp, transform=axs[0, 0].transAxes, ha='right', va='top',
                                   fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold',
                                             'color': 'darkred'})

                axs[0, 0].grid(which='major', axis='y',
                               color='darkgray', linestyle='--', linewidth=0.4)
                axs[0, 0].set_axisbelow(True)

                axs[0, 0].set_ylabel('Density', fontname='Arial', fontsize=16)
                for tl in axs[0, 0].get_yticklabels():
                    tl.set_fontname('Arial')
                    tl.set_fontsize(14)

                axs[0, 0].set_xlabel(
                    'Residuals [cm]', fontname='Arial', fontsize=16)
                for tl in axs[0, 0].get_xticklabels():
                    tl.set_fontname('Arial')
                    tl.set_fontsize(14)

            else:
                # Range && Clock
                sys.exit('Not supported yet!')

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)


def PlotResISL30(fResList, cPRN0, lRng, OutFilePrefix, OutFileSuffix):
    '''
    Create a pseudocolor plot for Mean/STD/RMS of ISL link residuals

    cPRN0 ---
     lRng ---
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
    # Color for the label of each orbit plane
    cPlanC = ['r', 'g', 'b', 'c', 'm', 'y', 'k']

    # Get satellite list for each orbit plane
    cSatP = []
    for i in range(nPlane):
        cSatP.append([])
    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                if cPRN0[0] != 'ALL' and (cWords[2] not in cPRN0 or cWords[3] not in cPRN0):
                    continue
                if (cWords[8] == 'F' and lRng) or (cWords[11] == 'F' and not lRng):
                    continue
                for j in range(nPlane):
                    if (cWords[2] in cPlane[j]) and (cWords[2] not in cSatP[j]):
                        cSatP[j].append(cWords[2])
                    if (cWords[3] in cPlane[j]) and (cWords[3] not in cSatP[j]):
                        cSatP[j].append(cWords[3])
    cSat = []
    for i in range(nPlane):
        if len(cSatP[i]) == 0:
            continue
        cSat0 = cSatP[i].copy()
        cSat0.sort()
        for j in range(len(cSat0)):
            cSat.append(cSat0[j])
    nSat = len(cSat)

    # Mean
    M = np.zeros((nSat, nSat))
    M[:, :] = np.nan
    # STD
    S = np.zeros((nSat, nSat))
    S[:, :] = np.nan
    # RMS
    R = np.zeros((nSat, nSat))
    R[:, :] = np.nan
    if lRng:
        # Range residuals
        iObs = 0
    else:
        # Clock residuals
        iObs = 5

    # Get the mean/std/rms for each link
    for i in range(nSat):
        for j in range(i + 1, nSat):
            rEpo, Res = GetResISL(fResList, cSat[i], cSat[j], False)
            nRes = np.count_nonzero(~np.isnan(Res[iObs]))
            if nRes == 0:
                continue
            M[i, j] = np.nanmean(Res[iObs])
            S[i, j] = np.nanstd(Res[iObs])
            # Get the RMS
            R[i, j] = 0.0
            for k in range(Res[iObs].size):
                if np.isnan(Res[iObs, k]):
                    continue
                R[i, j] = R[i, j] + Res[iObs, k] * Res[iObs, k]
            R[i, j] = np.sqrt(R[i, j] / nRes)

    fig, axs = plt.subplots(1, 1, squeeze=False,
                            figsize=(nSat * 0.55, nSat * 0.5))
    x = np.arange(-0.5, nSat, 1)
    y = np.arange(-0.5, nSat, 1)

    # Link RMS
    qm = axs[0, 0].pcolormesh(x, y, R, shading='flat')
    axs[0, 0].grid(which='both', axis='both', c='darkgray', ls='--', lw=0.4)
    axs[0, 0].set_axisbelow(True)
    axs[0, 0].xaxis.set_major_locator(MultipleLocator(1))
    axs[0, 0].yaxis.set_major_locator(MultipleLocator(1))
    axs[0, 0].set_xticks(range(nSat))
    axs[0, 0].set_xticklabels(cSat, family='Arial', size=14, rotation=90)
    for tl in axs[0, 0].get_xticklabels():
        # Color the tick label according to its orbit plane
        for i in range(nPlane):
            if tl.get_text() not in cPlane[i]:
                continue
            tl.set_color(cPlanC[i])
            break
    axs[0, 0].set_yticks(range(nSat))
    axs[0, 0].set_yticklabels(cSat, family='Arial', size=14)
    for tl in axs[0, 0].get_yticklabels():
        # Color the tick label according to its orbit plane
        for i in range(nPlane):
            if tl.get_text() not in cPlane[i]:
                continue
            tl.set_color(cPlanC[i])
            break
    cbar = fig.colorbar(qm, ax=axs[0, 0], location='right', pad=0.01,
                        aspect=30, anchor=(0.0, 0.5), panchor=(1.0, 0.5))
    cbar.set_label('RMS [cm]', loc='center', fontname='Arial', fontsize=16)
    for tl in cbar.ax.get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix + OutFileSuffix + '.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResISL40(fResList, iX, lFit, cSat0, OutFilePrefix, OutFileSuffix):
    '''
    Plot ISL residual series for specified satellites

       iX --- argument of the X axis
              0 --- time
              1 --- nadir angle
              2 --- azimuth angle
     lFit --- Whether do fitting to the residuals
              For res VS time, Lomb-Scarlge periodogram
              For res VS nadir, polynomial fitting
    cSat0 --- Specified satellite(s)
    '''

    cSat = []
    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                if cWords[8] == 'F' and cWords[11] == 'F':
                    continue
                if cWords[2] not in cSat and (cSat0[0] == 'ALL' or cWords[2] in cSat0):
                    cSat.append(cWords[2])
                if cWords[3] not in cSat and (cSat0[0] == 'ALL' or cWords[3] in cSat0):
                    cSat.append(cWords[3])
    cSat.sort()
    nSat = len(cSat)

    fig, axs = plt.subplots(nSat, 1, sharex='col',
                            squeeze=False, figsize=(12, nSat * 3))
    fig.subplots_adjust(hspace=0.1)

    for i in range(nSat):
        rEpo, Res = GetResISL(fResList, cSat[i], 'ALL', False)
        nRes = Res[0].size
        # Get the mean, std && rms for range and clock respectively
        nObs1 = 0
        RMS1 = 0.0
        nObs2 = 0
        RMS2 = 0.0
        for j in range(nRes):
            if not np.isnan(Res[0, j]):
                nObs1 = nObs1 + 1
                RMS1 = RMS1 + Res[0, j] * Res[0, j]
            if not np.isnan(Res[5, j]):
                nObs2 = nObs2 + 1
                RMS2 = RMS2 + Res[5, j] * Res[5, j]
        if nObs1 > 0:
            Mea1 = np.nanmean(Res[0])
            Sig1 = np.nanstd(Res[0])
            RMS1 = np.sqrt(RMS1 / nObs1)
        if nObs2 > 0:
            Mea2 = np.nanmean(Res[5])
            Sig2 = np.nanstd(Res[5])
            RMS2 = np.sqrt(RMS2 / nObs2)

        axs[i, 0].set_ylim(bottom=-30, top=30)
        if iX == 0:
            # residual VS epoch time
            axs[i, 0].plot(rEpo, Res[0], '.r', ms=2, label='Range')
            axs[i, 0].plot(rEpo, Res[5], '.b', ms=2, label='Clock')
            if lFit:
                # Lomb-Scarlge periodogram
                ls = LombScargle(rEpo, Res[0])
                f, p = ls.autopower(minimum_frequency=0.1,
                                    maximum_frequency=24, samples_per_peak=10)
                # #False Alarm Probability
                # fap=ls.false_alarm_probability(p.max())
                t_fit = np.linspace(np.amax(rEpo), np.amin(rEpo))
                y_fit = ls.model(t_fit, f[np.argmax(p)])
                axs[i, 0].plot(t_fit, y_fit, 'c', lw=2)

                # Creat an inset axis for the periodogram
                axins = inset_axes(axs[i, 0], width='60%',
                                   height="80%", loc=1, borderpad=0)
                axins.set_ylim(0, 1)
                axins.plot(f, p)

                # axins.set_xlabel('Frequency [cpr]',fontname='Arial',fontsize=14)
                axins.set_xticks([0, 1, 2, 3, 4, 5, 10, 15, 20])
                # axins.set_xticklabels(['0','1','2','3','4','5','10','15','20'])
                axins.set_xticklabels([])
                for tl in axins.get_xticklabels():
                    tl.set_fontname('Arial')
                    tl.set_fontsize(12)
                # axins.set_ylabel('Power',fontname='Arial',fontsize=14)
                axins.set_yticklabels([])
                for tl in axins.get_yticklabels():
                    tl.set_fontname('Arial')
                    tl.set_fontsize(12)
        elif iX == 1:
            # residual VS nadir angle
            axs[i, 0].plot(Res[1], Res[0], '.r', ms=2, label='Range')
            axs[i, 0].plot(Res[1], Res[5], '.b', ms=2, label='Clock')
            # Fit using polynomial
            if lFit:
                c = np.polynomial.polynomial.polyfit(Res[1], Res[0], 5)
                ind = np.argsort(Res[1])
                V = np.polynomial.polynomial.polyval(Res[1][ind], c)
                axs[i, 0].plot(Res[1][ind], V, '-', lw=2)
                # dV=Res[0]-V
                # axs[i,0].plot(Res[1],dV,'.b',ms=4)
                # np.polynomial.set_default_printstyle('unicode')
                p = np.polynomial.polynomial.Polynomial(c)
        else:
            # residual VS azimuth angle
            axs[i, 0].plot(Res[3], Res[0], '.r', ms=2, label='Range')
            axs[i, 0].plot(Res[3], Res[5], '.b', ms=2, label='Clock')
        axs[i, 0].axhline(color='darkgray', linestyle='dashed', alpha=0.5)
        axs[i, 0].text(0.02, 0.98, cSat[i] + '-ALL', transform=axs[i, 0].transAxes, ha='left', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
        if nObs1 > 0:
            strTmp = '{:>7.1f}+/-{:>6.1f}, RMS={:>7.1f}, #={:>6d}'.format(
                Mea1, Sig1, RMS1, nObs1)
            axs[i, 0].text(0.98, 0.98, strTmp, transform=axs[i, 0].transAxes, ha='right', va='top',
                           fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkred'})
        if nObs2 > 0:
            strTmp = '{:>7.1f}+/-{:>6.1f}, RMS={:>7.1f}, #={:>6d}'.format(
                Mea2, Sig2, RMS2, nObs2)
            axs[i, 0].text(0.98, 0.02, strTmp, transform=axs[i, 0].transAxes, ha='right', va='bottom',
                           fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkblue'})
        axs[i, 0].set_ylabel('[cm]', fontname='Arial', fontsize=16)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
    if iX == 0:
        axs[i, 0].set_xlabel('Modified Julian Day',
                             fontname='Arial', fontsize=16)
    elif iX == 1:
        axs[i, 0].set_xlabel('Nadir angle [deg]',
                             fontname='Arial', fontsize=16)
    else:
        axs[i, 0].set_xlabel('Azimuth angle [deg]',
                             fontname='Arial', fontsize=16)
    axs[i, 0].xaxis.set_major_formatter('{x:7.1f}')
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix + OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResISL41(fResList, lRect, cSat0, lClk, nCol, OutFilePrefix, OutFileSuffix):
    '''
    Plot ISL residual series w.r.t. its Azim && Elev for specified satellites
    one by one using the `scatter`

    lRect --- Whether plot in rectangular projection instead of polar projection
    cSat0 --- Specified satellite(s)
     lClk --- Whether for clock obs, otherwise for range obs
     nCol --- Number of columns for the output figure
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    cSat = []
    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                if cWords[8] == 'F' and cWords[11] == 'F':
                    continue
                if cWords[2] not in cSat and (cSat0[0] == 'ALL' or cWords[2] in cSat0):
                    cSat.append(cWords[2])
                if cWords[3] not in cSat and (cSat0[0] == 'ALL' or cWords[3] in cSat0):
                    cSat.append(cWords[3])
    cSat.sort()
    nSat = len(cSat)
    # Cal the number of row based on specified number of col
    nRow = math.ceil(nSat / nCol)

    if not lRect:
        fig, axs = plt.subplots(nRow, nCol, squeeze=False, figsize=(nCol * 6.5, nRow * 6),
                                subplot_kw={'projection': 'polar'})
    else:
        fig, axs = plt.subplots(nRow, nCol, squeeze=False,
                                figsize=(nCol * 6.5, nRow * 6))
    # fig.subplots_adjust(hspace=0.3)

    # Header line of the report
    print('PRN {: >6s} {: >6s} {: >6s} {: >6s}'.format(
        'MinA', 'MaxA', 'MinN', 'MaxN'))
    for i in range(nSat):
        rEpo, Res = GetResISL(fResList, cSat[i], 'ALL', False)
        nRes = Res[0].size
        if nRes == 0:
            continue
        else:
            # Cal the axis position, row-wise
            iRow = math.ceil((i + 1) / nCol) - 1
            iCol = i - iRow * nCol
        # Get the mean, std && rms
        nObs = 0
        RMS = 0.0
        Mea = 0.0
        Sig = 0.0
        if not lClk:
            for j in range(nRes):
                if not np.isnan(Res[0, j]):
                    nObs = nObs + 1
                    RMS = RMS + Res[0, j] * Res[0, j]
            if nObs > 0:
                Mea = np.nanmean(Res[0])
                Sig = np.nanstd(Res[0])
                RMS = np.sqrt(RMS / nObs)
            sc = axs[iRow, iCol].scatter(np.deg2rad(
                Res[3]), Res[1], c=Res[0], s=2.5, vmin=-30, vmax=30)
        else:
            for j in range(nRes):
                if not np.isnan(Res[5, j]):
                    nObs = nObs + 1
                    RMS = RMS + Res[5, j] * Res[5, j]
            if nObs > 0:
                Mea = np.nanmean(Res[5])
                Sig = np.nanstd(Res[5])
                RMS = np.sqrt(RMS / nObs)
            sc = axs[iRow, iCol].scatter(np.deg2rad(
                Res[3]), Res[1], c=Res[5], s=2.5, vmin=-30, vmax=30)
        # Min && Max azimuth
        xMin = np.amin(Res[3])
        xMax = np.amax(Res[3])
        # Min && Max nadir
        yMin = np.amin(Res[1])
        yMax = np.amax(Res[1])
        print(
            cSat[i] + ' {:>6.2f} {:>6.2f} {:>6.2f} {:>6.2f}'.format(xMin, xMax, yMin, yMax))

        if not lRect:
            axs[iRow, iCol].set_rmax(70)
            axs[iRow, iCol].set_rmin(0)
            axs[iRow, iCol].set_rgrids((10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60),
                                       labels=('10', '', '20', '', '',
                                               '', '40', '', '', '', '60'),
                                       family='Arial', size=12)
        for tl in axs[iRow, iCol].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(12)

        axs[iRow, iCol].grid(True, c='darkgray', ls='--', lw=0.4)
        # axs[iRow,iCol].set_axisbelow(True)
        cbar = fig.colorbar(sc, ax=axs[iRow, iCol])
        cbar.set_label('[cm]', loc='center', fontname='Arial', fontsize=16)
        for tl in cbar.ax.get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        strTmp = cSat[i] + \
            ' {:>7.1f}+/-{:>6.1f}, RMS={:>7.1f}, #={:>6d}'.format(
            Mea, Sig, RMS, nRes)
        axs[iRow, iCol].set_title(strTmp, va='bottom', fontdict={
            'fontfamily': 'Arial', 'fontsize': 16})
    # Hide the remaining axises at the last row if existed
    iCol = nSat - (nRow - 1) * nCol
    for j in range(iCol, nCol):
        axs[nRow - 1, j].set_axis_off()

    strTmp = OutFilePrefix + OutFileSuffix + '.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResISL42(fResList, lRect, cSat0, lClk, nCol, OutFilePrefix, OutFileSuffix):
    '''
    Plot ISL residual series w.r.t. its Azim && Elev for specified satellites
    one by one using the `pcolormesh`

    lRect --- Whether plot in rectangular projection instead of polar projection
    cSat0 --- Specified satellite(s)
     lClk --- Whether for clock obs, otherwise for range obs
     nCol --- Number of columns for the output figure
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    cSat = []
    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                if cWords[8] == 'F' and cWords[11] == 'F':
                    continue
                if cWords[2] not in cSat and (cSat0[0] == 'ALL' or cWords[2] in cSat0):
                    cSat.append(cWords[2])
                if cWords[3] not in cSat and (cSat0[0] == 'ALL' or cWords[3] in cSat0):
                    cSat.append(cWords[3])
    cSat.sort()
    nSat = len(cSat)
    # Cal the number of row based on specified number of col
    nRow = math.ceil(nSat / nCol)

    if not lRect:
        fig, axs = plt.subplots(nRow, nCol, squeeze=False, figsize=(nCol * 6.5, nRow * 6),
                                subplot_kw={'projection': 'polar'})
    else:
        fig, axs = plt.subplots(nRow, nCol, squeeze=False,
                                figsize=(nCol * 6.5, nRow * 6))
    # fig.subplots_adjust(hspace=0.3)

    # Header line of the report
    print('PRN {: >6s} {: >6s} {: >6s} {: >6s}'.format(
        'MinA', 'MaxA', 'MinN', 'MaxN'))
    for i in range(nSat):
        rEpo, Res = GetResISL(fResList, cSat[i], 'ALL', False)
        nRes = Res[0].size
        if nRes == 0:
            continue
        else:
            # Cal the axis position, row-wise
            iRow = math.ceil((i + 1) / nCol) - 1
            iCol = i - iRow * nCol
        # Get the mean, std && rms
        nObs = 0
        RMS = 0.0
        Mea = 0.0
        Sig = 0.0
        if not lClk:
            for j in range(nRes):
                if not np.isnan(Res[0, j]):
                    nObs = nObs + 1
                    RMS = RMS + Res[0, j] * Res[0, j]
            if nObs > 0:
                Mea = np.nanmean(Res[0])
                Sig = np.nanstd(Res[0])
                RMS = np.sqrt(RMS / nObs)
        else:
            for j in range(nRes):
                if not np.isnan(Res[5, j]):
                    nObs = nObs + 1
                    RMS = RMS + Res[5, j] * Res[5, j]
            if nObs > 0:
                Mea = np.nanmean(Res[5])
                Sig = np.nanstd(Res[5])
                RMS = np.sqrt(RMS / nObs)
        # Min && Max azimuth
        xMin = np.amin(Res[3])
        xMax = np.amax(Res[3])
        # Min && Max nadir
        yMin = np.amin(Res[1])
        yMax = np.amax(Res[1])
        print(
            cSat[i] + ' {:>6.2f} {:>6.2f} {:>6.2f} {:>6.2f}'.format(xMin, xMax, yMin, yMax))

        X, Y = np.meshgrid(np.linspace(xMin, xMax, 360),
                           np.linspace(yMin, yMax, 100))
        # Using interpolation to re-sample the data
        if not lClk:
            C = griddata((Res[3], Res[1]), Res[0], (X, Y), method='linear')
        else:
            C = griddata((Res[3], Res[1]), Res[5], (X, Y), method='linear')
        sc = axs[iRow, iCol].pcolormesh(np.deg2rad(X), Y, C, shading='gouraud',
                                        vmin=-30, vmax=30, rasterized=True)
        if not lRect:
            axs[iRow, iCol].set_rmax(70)
            axs[iRow, iCol].set_rmin(0)
            axs[iRow, iCol].set_rgrids((10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60),
                                       labels=('10', '', '20', '', '',
                                               '', '40', '', '', '', '60'),
                                       family='Arial', size=12)
        for tl in axs[iRow, iCol].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(12)

        axs[iRow, iCol].grid(True, color='darkgray',
                             linestyle='--', linewidth=0.8, alpha=0.3)
        axs[iRow, iCol].set_axisbelow(False)
        cbar = fig.colorbar(sc, ax=axs[iRow, iCol])
        cbar.set_label('[cm]', loc='center', fontname='Arial', fontsize=16)
        for tl in cbar.ax.get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        strTmp = cSat[i] + \
            ' {:>7.1f}+/-{:>6.1f}, RMS={:>7.1f}, #={:>6d}'.format(
            Mea, Sig, RMS, nRes)
        axs[iRow, iCol].set_title(strTmp, va='bottom', fontdict={
            'fontfamily': 'Arial', 'fontsize': 16})
    # Hide the remaining axises at the last row if existed
    iCol = nSat - (nRow - 1) * nCol
    for j in range(iCol, nCol):
        axs[nRow - 1, j].set_axis_off()

    strTmp = OutFilePrefix + OutFileSuffix + '.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResISL5(fResList, OutFilePrefix, OutFileSuffix):
    '''
    Plot ISL residuals link by link, but always put closed link together
    '''

    # Find all links
    cLink = []
    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                if cWords[3] > cWords[2]:
                    cTmp = cWords[2] + '-' + cWords[3]
                else:
                    cTmp = cWords[3] + '-' + cWords[2]
                if cTmp not in cLink:
                    cLink.append(cTmp)
    cLink.sort()
    nLink = len(cLink)

    cSat1 = []
    for i in range(nLink):
        if cLink[i][0:3] not in cSat1:
            cSat1.append(cLink[i][0:3])
    nSat1 = len(cSat1)

    cSat2 = [[] for i in range(nSat1)]
    nSat2 = np.zeros(nSat1, dtype=np.uint8)
    for i in range(nSat1):
        for j in range(nLink):
            if cLink[j][0:3] != cSat1[i]:
                continue
            if cLink[j][4:7] not in cSat2[i]:
                cSat2[i].append(cLink[j][4:7])
        nSat2[i] = len(cSat2[i])
    # Creat a multipage pdf, with each page for one satellite
    strTmp = OutFilePrefix + OutFileSuffix

    with PdfPages(strTmp) as pdf:
        for i in range(nSat1):
            cLink0 = []
            for j in range(nSat2[i] - 1):
                k = j + 1
                while k <= nSat2[i] - 1:
                    cTmp = cSat2[i][j] + '-' + cSat2[i][k]
                    if cTmp in cLink:
                        cLink0.append(cSat1[i] + '-' + cSat2[i][j])
                        cLink0.append(cSat1[i] + '-' + cSat2[i][k])
                        cLink0.append(cTmp)
                    k = k + 1
            nLink0 = len(cLink0)
            if nLink0 == 0:
                continue
            fig, axs = plt.subplots(
                nLink0, 1, sharex='col', squeeze=False, figsize=(8, nLink0 * 1.5))
            fig.subplots_adjust(hspace=0.1)
            for j in range(nLink0):
                rEpo, Res = GetResISL(
                    fResList, cLink0[j][0:3], cLink0[j][4:7], False)
                nRes = Res[0].size
                Mea = np.mean(Res[0])
                Sig = np.std(Res[0])
                RMS = 0.0
                for k in range(nRes):
                    RMS = RMS + Res[0, k] * Res[0, k]
                RMS = np.sqrt(RMS / nRes)
                # residual VS epoch time
                axs[j, 0].set_ylim(bottom=-30, top=30)
                axs[j, 0].plot(rEpo, Res[0], '.r', label=cLink0[j])
                axs[j, 0].text(0.05, 0.95, cLink0[j],
                               transform=axs[j, 0].transAxes, ha='left', va='top')
                axs[j, 0].text(0.95, 0.95,
                               'Mea={:>7.1f}, STD={:>6.1f}, RMS={:>7.1f}, #={:>6d}'.format(
                                   Mea, Sig, RMS, nRes),
                               transform=axs[j, 0].transAxes, ha='right', va='top')
                axs[j, 0].set_ylabel('[cm]')
            axs[j, 0].set_xlabel('Modified Julian Day')
            axs[j, 0].xaxis.set_major_formatter('{x:8.2f}')
            # Save the figure into the pdf
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
            print('Finish page {:2d}, {:3d} links'.format(
                i, nLink0) + ', ' + cSat1[i])


def PlotResISL60(fRes1, fRes2, OutFilePrefix, OutFileSuffix):
    '''
    Plot the difference of ISL residuals in fRes1 and fRes2
    '''

    # Find the common links
    cLink1 = []
    with open(fRes1, mode='rt') as fOb:
        for cLine in fOb:
            if cLine[0:6] != 'ISLRES':
                continue
            cWords = cLine.split()
            if cWords[3] > cWords[2]:
                cTmp = cWords[2] + '-' + cWords[3]
            else:
                cTmp = cWords[3] + '-' + cWords[2]
            if cTmp not in cLink1:
                cLink1.append(cTmp)
    cLink1.sort()
    cLink = []
    with open(fRes2, mode='rt') as fOb:
        for cLine in fOb:
            if cLine[0:6] != 'ISLRES':
                continue
            cWords = cLine.split()
            if cWords[3] > cWords[2]:
                cTmp = cWords[2] + '-' + cWords[3]
            else:
                cTmp = cWords[3] + '-' + cWords[2]
            if (cTmp in cLink1) and (cTmp not in cLink):
                cLink.append(cTmp)
    cLink.sort()
    nLink = len(cLink)

    fig, axs = plt.subplots(nLink, 1, sharex='col',
                            squeeze=False, figsize=(8, nLink * 1.5))
    fig.subplots_adjust(hspace=0.1)
    # Formatting tick label

    for i in range(nLink):
        rEpo1, Res1 = GetResISL(fRes1, cLink[i][0:3], cLink[i][4:7], False)
        rEpo2, Res2 = GetResISL(fRes1, cLink[i][0:3], cLink[i][4:7], False)
        x = []
        y = []
        for j in range(len(rEpo1)):
            lFound = False
            for k in range(len(rEpo2)):
                if abs(rEpo1[j] - rEpo2[k]) * 86400 < 0.1:
                    lFound = True
                    break
            if lFound:
                x.append(rEpo1[j])
                # Difference in mm
                y.append((Res1[0][j] - Res2[0][k]) * 10)
        rEpo = np.array(x)
        Res = np.array(y)

        nRes = Res.size
        Mea = np.mean(Res)
        Sig = np.std(Res)
        RMS = 0.0
        for j in range(nRes):
            RMS = RMS + Res[j] * Res[j]
        RMS = np.sqrt(RMS / nRes)

        axs[i, 0].plot(rEpo, Res, '.r', label=cLink[i])
        axs[i, 0].text(0.05, 0.95, cLink[i], transform=axs[i,
                                                           0].transAxes, ha='left', va='top')
        axs[i, 0].text(0.95, 0.95, 'Mea={:>7.1f}, STD={:>6.1f}, RMS={:>7.1f}, #={:>6d}'.format(Mea, Sig, RMS, nRes),
                       transform=axs[i, 0].transAxes, ha='right', va='top')
        axs[i, 0].set_ylabel('[mm]')
    # Only for the last subplot
    axs[i, 0].set_xlabel('Modified Julian Day')
    axs[i, 0].xaxis.set_major_formatter('{x:8.2f}')

    strTmp = OutFilePrefix + OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResISL61(fResList, rMJD1, rMJD2, OutFilePrefix, OutFileSuffix):
    '''
    Re-construct the normal equation of ISL original observations when
    estimating recv && tran HD parameters with piece-wise constant model

    rMJD1 --- start time of the considered piece
    rMJD2 --- end time of the considered piece

    '''

    cPRN = ['C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27',
            'C28', 'C29', 'C30', 'C32', 'C33', 'C34', 'C35', 'C36', 'C37',
            'C38', 'C39', 'C40', 'C41', 'C42', 'C43', 'C44', 'C45', 'C46']

    nSat = len(cPRN)
    nPar = nSat*2-1
    # Number of obs for each par
    nObs = np.zeros(nPar, dtype=np.int32)
    # the overall design matrix
    A = []
    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                rMJD = int(cWords[14]) + float(cWords[15]) / 86400.0
                if rMJD < rMJD1 or rMJD >= rMJD2:
                    continue
                if cWords[8] == 'F':
                    continue
                # design vector for this obs
                A0 = []
                for j in range(nPar):
                    A0.append(0)
                # index of recv sat
                iSat1 = cPRN.index(cWords[2])
                # index of tran sat
                iSat2 = cPRN.index(cWords[3])
                # Recv HD par
                if iSat1 == 0:
                    # No recv HD par for the first sat (reference sat)
                    iPar1 = -1
                else:
                    iPar1 = 2 * iSat1
                # Tran HD
                if iSat2 == 0:
                    # No recv HD par for the first sat (reference sat)
                    iPar2 = 0
                else:
                    iPar2 = 2 * iSat2 - 1

                if iPar1 >= 0:
                    A0[iPar1] = 1
                    nObs[iPar1] = nObs[iPar1]+1
                A0[iPar2] = 1
                nObs[iPar2] = nObs[iPar2]+1
                A.append(A0)
    B = np.array(A)
    # Normal matrx
    N = np.matmul(np.transpose(B), B)

    # Rank && Conditional number of the normal matrix
    print('Rank: ', NupLA.matrix_rank(N))
    print('Cond: ', NupLA.cond(N))
    # Determinant of the normal matrix
    print('Det:  ', SciLA.det(N))

    fOut = open(OutFilePrefix + os.path.splitext(OutFileSuffix)[0], 'w')

    # Top header line
    StrTmp = '             '
    for i in range(nPar):
        StrTmp = StrTmp + '        {: >6d}        '.format(i)
    fOut.write(StrTmp + '\n')
    # The normal matrix
    for i in range(nPar):
        StrTmp = '{: >6d} {: >6d}'.format(nObs[i], i)
        for j in range(nPar):
            StrTmp = StrTmp + ' {: >21.14E}'.format(N[i, j])
        fOut.write(StrTmp + '\n')


def PlotResISL70(fResList, cLink0, lRng, nT, nCol, dIntv, OutFilePrefix, OutFileSuffix):
    '''
    Plot Lomb-Scargle periodogram of ISL residuals for specific
    links one by one. Addtionally, the fitting model will also be plotted against the
    residual series.

    cLink0 --- Specified ISL links, set cLink0[0]='ALL-ALL' to plot all links
      lRng --- Whether for range obs. Otherwise, for clock obs
        nT --- Number of terms to use in the Fourier fit for LS modeling
      nCol ---
     dIntv --- the interval of sampling for the fitting model, in days
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    if nT <= 0:
        nT = 1

    cLink = []
    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                if lRng and cWords[8] == 'F':
                    continue
                if (not lRng) and cWords[11] == 'F':
                    continue
                if cWords[3] > cWords[2]:
                    cTmp = cWords[2] + '-' + cWords[3]
                else:
                    cTmp = cWords[3] + '-' + cWords[2]
                if cLink0[0] != 'ALL-ALL' and cTmp not in cLink0:
                    continue
                if cTmp not in cLink:
                    cLink.append(cTmp)
    cLink.sort()
    nLink = len(cLink)
    # Cal the number of row based on specified number of col
    nRow = math.ceil(nLink / nCol)
    # index of obs residuals
    if lRng:
        iObs = 0
    else:
        iObs = 5

    fig, axs = plt.subplots(nRow, nCol, sharex='col', sharey='row',
                            squeeze=False, figsize=(nCol * 8, nRow * 2.5))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)

    for i in range(nLink):
        rEpo, Res = GetResISL(fResList, cLink[i][0:3], cLink[i][4:7], False)
        nRes = Res[iObs].size
        if nRes == 0:
            print('Link ' + cLink[i] + ' not found!')
            continue
        else:
            # Cal the axis position, row-wise
            iRow = math.ceil((i + 1) / nCol) - 1
            iCol = i - iRow * nCol
        # Get the mean, std && rms
        nObs = 0
        RMS1 = 0.0
        Mea1 = 0.0
        Sig1 = 0.0
        for j in range(nRes):
            if not np.isnan(Res[iObs, j]):
                nObs = nObs + 1
                RMS1 = RMS1 + Res[iObs, j] * Res[iObs, j]
        if nObs > 0:
            Mea1 = np.nanmean(Res[iObs])
            Sig1 = np.nanstd(Res[iObs])
            RMS1 = np.sqrt(RMS1 / nObs)

        # Lomb-Scarlge periodogram
        ls = LombScargle(rEpo, Res[iObs], fit_mean=True,
                         center_data=True, nterms=nT)
        f, p = ls.autopower(minimum_frequency=0.1,
                            maximum_frequency=24, samples_per_peak=10)
        if nT == 1:
            # False Alarm Probability
            fap = ls.false_alarm_probability(p.max())
            axs[iRow, iCol].axhline(
                y=fap, color='darkgray', linestyle='dashed')
        axs[iRow, iCol].plot(f, p, 'b', label=cLink[i])

        # Creat an inset axis for the time-series
        axins = inset_axes(axs[iRow, iCol], width='60%',
                           height="80%", loc=1, borderpad=0)
        # Plot the orignial residuals
        axins.plot(rEpo, Res[iObs], '.r', ms=2)
        axins.grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
        axins.set_axisbelow(True)
        # Plot the fitted model
        t0 = np.arange(np.amin(rEpo), np.amax(rEpo), dIntv)
        y0 = ls.model(t0, f[np.argmax(p)])
        axins.plot(t0, y0, '.-g', ms=2)
        # RMS before fitting
        strTmp = 'RMS={:>7.1f}'.format(RMS1)
        axins.text(0.02, 0.98, strTmp, transform=axins.transAxes, ha='left', va='top',
                   fontdict={'fontsize': 10, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'red'})
        # RMS after fitting
        y_fit = ls.model(rEpo, f[np.argmax(p)])
        RMS2 = 0.0
        Mea2 = 0.0
        Sig2 = 0.0
        for j in range(nRes):
            if not np.isnan(Res[iObs, j]):
                RMS2 = RMS2 + (Res[iObs, j] - y_fit[j]) * \
                    (Res[iObs, j] - y_fit[j])
        if nObs > 0:
            Mea2 = np.nanmean(Res[iObs] - y_fit)
            Sig2 = np.nanstd(Res[iObs] - y_fit)
            RMS2 = np.sqrt(RMS2 / nObs)
        # RMS decrease percentage if removing fitted model
        dP = (RMS1 - RMS2) / RMS1 * 100
        strTmp = 'RMS={:>7.1f} ({:6.1f}% '.format(RMS2, dP) + r'$\downarrow$)'
        axins.text(0.02, 0.02, strTmp, transform=axins.transAxes, ha='left', va='bottom',
                   fontdict={'fontsize': 10, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'green'})

        axins.set_xticklabels([])
        axins.set_ylabel('[cm]', fontname='Arial', fontsize=14)
        for tl in axins.get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(12)

        axs[iRow, iCol].set_ylim(bottom=0, top=1)
        axs[iRow, iCol].set_xlim(left=0, right=25)
        axs[iRow, iCol].set_xticks([0, 1, 2, 3, 4, 5, 10, 15, 20])
        axs[iRow, iCol].set_xticklabels(
            ['0', '1', '2', '3', '4', '5', '10', '15', '20'])

        axs[iRow, iCol].text(0.02, 0.98, cLink[i], transform=axs[iRow, iCol].transAxes, ha='left', va='top',
                             fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
        # Label the best-fit frequency
        axs[iRow, iCol].text(f[np.argmax(p)], p[np.argmax(p)], '{:>5.2f}'.format(f[np.argmax(p)]),
                             fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'blue'})
        if iCol == 0:
            axs[iRow, iCol].set_ylabel(
                'LS Power', fontname='Arial', fontsize=16)
            for tl in axs[iRow, iCol].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
        if iRow == (nRow - 1):
            # Only for the last subplot
            axs[iRow, iCol].set_xlabel(
                'Frequency [cycle per day]', fontname='Arial', fontsize=16)
            for tl in axs[iRow, iCol].get_xticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)

    strTmp = OutFilePrefix + OutFileSuffix + '.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResISL71(cSer, fSerList, cLink0, lRng, lByCol, OutFilePrefix, OutFileSuffix):
    '''
    Plot Lomb-Scargle periodogram of ISL residuals for specific
    links one by one.
    This is similiar to PlotResISL70 but plotting for multipl solutions
    column by column or row by row for the purpose of comparison.

    cLink0 --- Specified ISL links, set cLink0[0]='ALL-ALL' to plot all links
      lRng --- Whether for range obs. Otherwise, for clock obs
    lByCol --- All links from each individual solution are arranged as a single
               column
    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nSer = len(cSer)
    # Get the common link set among all solutions
    cLink1 = []
    iLink = []
    for iSer in range(nSer):
        nFile = len(fSerList[iSer])
        for i in range(nFile):
            with open(fSerList[iSer][i], mode='rt') as fOb:
                for cLine in fOb:
                    if cLine[0:6] != 'ISLRES':
                        continue
                    cWords = cLine.split()
                    # Invalid range obs
                    if cWords[8] == 'F' and lRng:
                        continue
                    # Invalid clock obs
                    if cWords[11] == 'F' and (not lRng):
                        continue
                    if cWords[3] > cWords[2]:
                        cTmp = cWords[2] + '-' + cWords[3]
                    else:
                        cTmp = cWords[3] + '-' + cWords[2]
                    if cLink0[0] != 'ALL-ALL' and cTmp not in cLink0:
                        continue
                    if cTmp not in cLink1:
                        # Globally new link
                        cLink1.append(cTmp)
                        # list of solutions this link showed up
                        iLink.append([])
                        j = len(cLink1) - 1
                        iLink[j].append(iSer)
                    else:
                        j = cLink1.index(cTmp)
                        if iSer not in iLink[j]:
                            # New link for this solution
                            iLink[j].append(iSer)
    # Only keep those links that shows up in every solution
    cLink = []
    for i in range(len(cLink1)):
        if len(iLink[i]) != nSer:
            continue
        cLink.append(cLink1[i])
    cLink.sort()
    nLink = len(cLink)

    # index of obs residuals
    if lRng:
        iObs = 0
    else:
        iObs = 5

    if lByCol:
        fig, axs = plt.subplots(nLink, nSer, sharex='col', sharey='row',
                                squeeze=False, figsize=(8 * nSer, nLink * 2.5))
    else:
        fig, axs = plt.subplots(nSer, nLink, sharex='col', sharey='row',
                                squeeze=False, figsize=(8 * nLink, nSer * 2.5))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)

    # Plot for each solution
    for iSer in range(nSer):
        for iLnk in range(nLink):
            rEpo, Res = GetResISL(
                fSerList[iSer], cLink[iLnk][0:3], cLink[iLnk][4:7], False)
            nRes = Res[iObs].size
            # Get the mean, std && rms
            nObs = 0
            RMS1 = 0.0
            Mea1 = 0.0
            Sig1 = 0.0
            for j in range(nRes):
                if not np.isnan(Res[iObs, j]):
                    nObs = nObs + 1
                    RMS1 = RMS1 + Res[iObs, j] * Res[iObs, j]
            Mea1 = np.nanmean(Res[iObs])
            Sig1 = np.nanstd(Res[iObs])
            RMS1 = np.sqrt(RMS1 / nObs)
            if lByCol:
                iRow = iLnk
                iCol = iSer
            else:
                iRow = iSer
                iCol = iLnk

            # Lomb-Scarlge periodogram
            ls = LombScargle(rEpo, Res[iObs])
            f, p = ls.autopower(minimum_frequency=0.1,
                                maximum_frequency=24, samples_per_peak=10)

            # False Alarm Probability
            fap = ls.false_alarm_probability(p.max())
            axs[iRow, iCol].axhline(
                y=fap, color='darkgray', linestyle='dashed')
            axs[iRow, iCol].plot(f, p, 'b', label=cLink[iLnk])

            # Creat an inset axis
            axins = inset_axes(axs[iRow, iCol], width='60%',
                               height="80%", loc=1, borderpad=0)
            # Plot the orignial residuals
            axins.plot(rEpo, Res[iObs], '.r', ms=2)
            axins.grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
            axins.set_axisbelow(True)
            # Plot the fitted model
            t0 = np.linspace(np.amin(rEpo), np.amax(rEpo))
            y0 = ls.model(t0, f[np.argmax(p)])
            axins.plot(t0, y0, 'g', lw=1.5)
            # RMS before fitting
            strTmp = 'RMS={:>7.1f}'.format(RMS1)
            axins.text(0.02, 0.98, strTmp, transform=axins.transAxes, ha='left', va='top',
                       fontdict={'fontsize': 10, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'red'})
            # RMS after fitting
            y_fit = ls.model(rEpo, f[np.argmax(p)])
            RMS2 = 0.0
            Mea2 = 0.0
            Sig2 = 0.0
            for j in range(nRes):
                if not np.isnan(Res[iObs, j]):
                    RMS2 = RMS2 + (Res[iObs, j] - y_fit[j]) * \
                        (Res[iObs, j] - y_fit[j])
            if nObs > 0:
                Mea2 = np.nanmean(Res[iObs] - y_fit)
                Sig2 = np.nanstd(Res[iObs] - y_fit)
                RMS2 = np.sqrt(RMS2 / nObs)
            # RMS decrease percentage if removing fitted model
            dP = (RMS1 - RMS2) / RMS1 * 100
            strTmp = 'RMS={:>7.1f} ({:6.1f}% '.format(
                RMS2, dP) + r'$\downarrow$)'
            axins.text(0.02, 0.02, strTmp, transform=axins.transAxes, ha='left', va='bottom',
                       fontdict={'fontsize': 10, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'green'})

            axins.set_xticklabels([])
            axins.set_ylabel('[cm]', fontname='Arial', fontsize=14)
            for tl in axins.get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(12)

            axs[iRow, iCol].set_ylim(bottom=0, top=1)
            axs[iRow, iCol].set_xlim(left=0, right=25)
            axs[iRow, iCol].set_xticks([0, 1, 2, 3, 4, 5, 10, 15, 20])
            axs[iRow, iCol].set_xticklabels(
                ['0', '1', '2', '3', '4', '5', '10', '15', '20'])
            for tl in axs[iRow, iCol].get_xticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            for tl in axs[iRow, iCol].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)

            axs[iRow, iCol].text(0.02, 0.98, cLink[iLnk] + ' ' + cSer[iSer], transform=axs[iRow, iCol].transAxes,
                                 ha='left', va='top',
                                 fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
            # Label the best-fit frequency
            axs[iRow, iCol].text(f[np.argmax(p)], p[np.argmax(p)], '{:>5.2f}'.format(f[np.argmax(p)]),
                                 fontdict={'fontsize': 16, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'blue'})
            # y-label, only for the first column
            if iCol == 0:
                axs[iRow, iCol].set_ylabel(
                    'LS Power', fontname='Arial', fontsize=16)
            # x-lable, only for the last row
            if (lByCol and iRow == (nLink - 1)) or (not lByCol and iRow == (nSer - 1)):
                axs[iRow, iCol].set_xlabel(
                    'Frequency [cycle per day]', fontname='Arial', fontsize=16)

    strTmp = OutFilePrefix + OutFileSuffix + '.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResISL72(fResList, cLink0, lRng, nT, OutFilePrefix, OutFileSuffix):
    '''
    Similar to PlotResISL70, but create a figure for each link putting
    into a PDF.

    cLink0 --- Specified ISL links, set cLink0[0]='ALL-ALL' to plot all links
      lRng --- Whether for range obs. Otherwise, for clock obs
        nT --- Number of terms to use in the Fourier fit for LS modeling
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    if nT <= 0:
        nT = 1

    # Get the full link list
    cLink = []
    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                if lRng and cWords[8] == 'F':
                    continue
                if (not lRng) and cWords[11] == 'F':
                    continue
                # Do not discriminate the trans/recv end
                if cWords[3] > cWords[2]:
                    cTmp = cWords[2] + '-' + cWords[3]
                else:
                    cTmp = cWords[3] + '-' + cWords[2]
                if cLink0[0] != 'ALL-ALL' and cTmp not in cLink0:
                    continue
                if cTmp not in cLink:
                    cLink.append(cTmp)
    cLink.sort()
    nLink = len(cLink)
    # index of obs residuals
    if lRng:
        iObs = 0
    else:
        iObs = 5

    # Statistics of the top 3 peak frequencies
    xFreq = []
    for i in range(3):
        # Frequency
        xFreq.append([])
        # Number of links has this frequency
        xFreq.append([])
    # Lomb-Scarlge model parameters for each link
    LSM = []
    cLk = []

    strTmp = os.path.join(OutFilePrefix, OutFileSuffix + '.pdf')
    with PdfPages(strTmp) as pdf:
        for i in range(nLink):
            rEpo, Res = GetResISL(
                fResList, cLink[i][0:3], cLink[i][4:7], False)
            nRes = Res[iObs].size
            if nRes == 0:
                print('Link ' + cLink[i] + ' not found!')
                continue
            # Get the mean, std && rms for this link
            nObs = 0
            RMS1 = 0.0
            Mea1 = 0.0
            Sig1 = 0.0
            for j in range(nRes):
                if np.isnan(Res[iObs, j]):
                    continue
                nObs = nObs + 1
                RMS1 = RMS1 + Res[iObs, j] * Res[iObs, j]
            if nObs == 0:
                # No valid points for this link
                continue
            Mea1 = np.nanmean(Res[iObs])
            Sig1 = np.nanstd(Res[iObs])
            RMS1 = np.sqrt(RMS1 / nObs)
            # Creat a Lomb-Scarlge periodogram instance
            ls = LombScargle(
                rEpo, Res[iObs], fit_mean=True, center_data=True, nterms=nT)
            f, p = ls.autopower(minimum_frequency=0.1,
                                maximum_frequency=24, samples_per_peak=10)
            # Get the model parameters for this link
            LSM0 = []
            LSM0.append(ls.offset())
            # Sort the powers and take the top three best-fit models (Note: in ascending order)
            ind = np.argsort(p)
            for j in range(3):
                # Get the model parameters for this best-fit
                LSM0.append(f[ind[-(j + 1)]])
                LSM0.append(p[ind[-(j + 1)]])
                theta = ls.model_parameters(f[ind[-(j + 1)]])
                for k in range(theta.size):
                    LSM0.append(theta[k])
                # Whether this frequency is found
                lFound = False
                for k in range(len(xFreq[j * 2])):
                    if math.fabs(f[ind[-(j + 1)]] - xFreq[j * 2][k]) > 0.1:
                        continue
                    # Existed frequency
                    lFound = True
                    xFreq[j * 2 + 1][k] = xFreq[j * 2 + 1][k] + 1
                    break
                if lFound:
                    continue
                # New frequency
                xFreq[j * 2].append(f[ind[-(j + 1)]])
                xFreq[j * 2 + 1].append(1)
            # Accumulate to the global list
            LSM.append(LSM0)
            cLk.append(cLink[i])

            fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(8, 2.5))
            if nT == 1:
                # False Alarm Probability, only for single-term fitting
                fap = ls.false_alarm_probability(p.max())
                axs[0, 0].axhline(y=fap, color='darkgray', linestyle='dashed')
            axs[0, 0].plot(f, p, 'b', label=cLink[i])

            # Creat an inset axis for the time-series
            axins = inset_axes(axs[0, 0], width='60%',
                               height="80%", loc=1, borderpad=0)
            # Plot the orignial residuals
            axins.plot(rEpo, Res[iObs], '.r', ms=2)
            axins.grid(which='major', axis='y', color='darkgray',
                       linestyle='--', linewidth=0.4)
            axins.set_axisbelow(True)
            # Plot the fitted model
            t0 = np.linspace(np.amin(rEpo), np.amax(rEpo))
            y0 = ls.model(t0, f[np.argmax(p)])
            axins.plot(t0, y0, 'g', lw=1.5)
            # RMS before fitting
            strTmp = 'RMS={:>7.1f}'.format(RMS1)
            axins.text(0.02, 0.98, strTmp, transform=axins.transAxes, ha='left', va='top',
                       fontdict={'fontsize': 10, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'red'})
            # RMS after fitting
            y_fit = ls.model(rEpo, f[np.argmax(p)])
            RMS2 = 0.0
            Mea2 = 0.0
            Sig2 = 0.0
            for j in range(nRes):
                if not np.isnan(Res[iObs, j]):
                    RMS2 = RMS2 + (Res[iObs, j] - y_fit[j]) * \
                        (Res[iObs, j] - y_fit[j])

            Mea2 = np.nanmean(Res[iObs] - y_fit)
            Sig2 = np.nanstd(Res[iObs] - y_fit)
            RMS2 = np.sqrt(RMS2 / nObs)
            # RMS decrease percentage if removing fitted model
            dP = (RMS1 - RMS2) / RMS1 * 100
            strTmp = 'RMS={:>7.1f} ({:6.1f}% '.format(
                RMS2, dP) + r'$\downarrow$)'
            axins.text(0.02, 0.02, strTmp, transform=axins.transAxes, ha='left', va='bottom',
                       fontdict={'fontsize': 10, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'green'})

            axins.set_xticklabels([])
            axins.set_ylabel('[cm]', fontname='Arial', fontsize=14)
            for tl in axins.get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(12)

            axs[0, 0].set_ylim(bottom=0, top=1)
            axs[0, 0].set_xlim(left=0, right=25)
            axs[0, 0].set_xticks([0, 1, 2, 3, 4, 5, 10, 15, 20])
            axs[0, 0].set_xticklabels(
                ['0', '1', '2', '3', '4', '5', '10', '15', '20'])
            for tl in axs[0, 0].get_xticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            for tl in axs[0, 0].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)

            axs[0, 0].text(0.02, 0.98, cLink[i], transform=axs[0, 0].transAxes, ha='left', va='top',
                           fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
            # Label the best-fit frequency
            axs[0, 0].text(f[np.argmax(p)], p[np.argmax(p)], '{:>5.2f}'.format(f[np.argmax(p)]),
                           fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'blue'})
            axs[0, 0].set_ylabel('LS Power', fontname='Arial', fontsize=16)
            axs[0, 0].set_xlabel('Frequency [cycle per day]',
                                 fontname='Arial', fontsize=16)

            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)
    # Write the statistics and model parameters to a file
    fOut = open(OutFilePrefix + os.path.splitext(OutFileSuffix)[0], 'w')
    # First, the statistics of top three peak frequency
    fOut.write('#{: <5s} {: >19s} {: >7s}\n'.format('Peak', 'Freq', 'LinkNum'))
    for i in range(3):
        # Sort along the frequency
        ind = np.argsort(xFreq[i * 2])
        for j in range(ind.size):
            # NOTE: the frequency should be as precise as possible because of the
            #       time arguments are normally large (rMJD)
            fOut.write(' Peak{:<1d} {: >19.16f} {: >7.0f}\n'.format(i + 1,
                                                                    xFreq[i * 2][ind[j]], xFreq[i * 2 + 1][ind[j]]))
    # Then, model parameters
    # the header line
    strTmp = '#{: <7s} {: >8s}'.format('Link', 'Offset')
    for i in range(3):
        strTmp = strTmp + \
            ' {: >18s}{:<1d} Power{:<1d}   Theta0'.format('Freq', i, i)
        # For each term
        for j in range(nT):
            strTmp = strTmp + \
                '   Theta{:<1d}   Theta{:<1d}'.format(
                    2 * (j + 1) - 1, 2 * (j + 1))
    fOut.write(strTmp + '\n')
    # print for each link
    for l in range(len(cLk)):
        # Link, offset
        strTmp = ' {: <7s} {: >8.2f}'.format(cLk[l], LSM[l][0])
        # the three highest power
        for i in range(3):
            # Freq, power, and theta0
            strTmp = strTmp + ' {: >19.16f} {: >6.2f} {: >8.2f}'.format(LSM[l][i * (3 + 2 * nT) + 1],
                                                                        LSM[l][i *
                                                                               (3 + 2 * nT) + 2],
                                                                        LSM[l][i * (3 + 2 * nT) + 3])
            # For each term
            for j in range(nT):
                strTmp = strTmp + ' {: >8.2f} {: >8.2f}'.format(LSM[l][i * (3 + 2 * nT) + 4 + j * 2],
                                                                LSM[l][i * (3 + 2 * nT) + 5 + j * 2])
        fOut.write(strTmp + '\n')
    fOut.close()


def PlotResISL80(cSer, fSerList, lRng, OutFilePrefix, OutFileSuffix):
    '''
    Plot histogram for the decrease percentages of link residuals RMS
    with respect to the first solution of all other solutions

        cSer --- various solutions' name
    fSerList --- the first list is the baseline solution, and the following
                    are the various solutions to be compared with the baseline
                    solutions
        lRng --- For range or clock residuals
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # Links in the first file set
    cLink = []
    for i in range(len(fSerList[0])):
        with open(fSerList[0][i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                if lRng and cWords[8] == 'F':
                    continue
                if not lRng and cWords[11] == 'F':
                    continue
                if cWords[3] > cWords[2]:
                    cTmp = cWords[2] + '-' + cWords[3]
                else:
                    cTmp = cWords[3] + '-' + cWords[2]
                if cTmp not in cLink:
                    cLink.append(cTmp)
    nLink = len(cLink)
    cLink.sort()
    nSer = len(cSer)
    RMS = np.zeros((nLink, nSer))
    dP = np.zeros((nLink, nSer - 1))

    for i in range(nLink):
        # Get the link residual RMS for each solution
        for iSer in range(nSer):
            rEpo, Res = GetResISL(
                fSerList[iSer], cLink[i][0:3], cLink[i][4:7], False)
            if np.count_nonzero(~np.isnan(Res[0, :])) == 0:
                # Not found link
                RMS[i, iSer] = np.nan
            else:
                nObs = 0
                for j in range(Res[0].size):
                    if np.isnan(Res[0, j]):
                        continue
                    nObs = nObs + 1
                    RMS[i, iSer] = RMS[i, iSer] + Res[0, j] * Res[0, j]
                RMS[i, iSer] = np.sqrt(RMS[i, iSer] / nObs)
            if iSer > 0:
                # Decrease Percentage wrt the first solution
                if np.isnan(RMS[i, 0]) or np.isnan(RMS[i, iSer]):
                    dP[i, iSer - 1] = np.nan
                else:
                    dP[i, iSer - 1] = (RMS[i, 0] - RMS[i, iSer]
                                       ) / RMS[i, 0] * 100
                    # Report Negative decrease, i.e increase
                    if dP[i, iSer - 1] < 0.0:
                        print('{: <8s} {: >8s} {: >7.1f} {: >7.1f} {: >7.2f}'.format(cSer[iSer],
                                                                                     cLink[i], RMS[i,
                                                                                                   0], RMS[i, iSer],
                                                                                     dP[i, iSer - 1]))

    fig, axs = plt.subplots(1, nSer - 1, sharey='row',
                            squeeze=False, figsize=(6 * (nSer - 1), 4))
    fig.subplots_adjust(wspace=0.05)

    for i in range(nSer - 1):
        axs[0, i].text(0.02, 0.98, cSer[i + 1], transform=axs[0, i].transAxes, ha='left', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
        n, bins, pat = axs[0, i].hist(dP[:, i], bins=50, density=True)
        axs[0, i].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
        axs[0, i].set_axisbelow(True)
        axs[0, i].set_xlabel('Decrease percentage of RMS [%]',
                             fontname='Arial', fontsize=16)
        for tl in axs[0, i].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

    axs[0, 0].set_ylabel('Density', fontname='Arial', fontsize=16)
    axs[0, 0].yaxis.set_major_formatter('{x: >4.2f}')
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix + OutFileSuffix + '.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResISL81(fResList1, lRng1, fResList2, lRng2, cLink0, OutFilePrefix, OutFileSuffix):
    '''
    Plot the Pearson product-moment correlation coefficients for each
    specified links between two residual sets.

    lRng1 --- Range obs from first file, otherwise take the clock obs
    lRng2 --- Range obs from second file, otherwise take the clock obs

    NOTE: the res files are matched according to their names
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nFile = len(fResList1)
    # Get the basename for later matching
    fN1 = []
    fN2 = []
    rMJD = []
    for i in range(nFile):
        fN1.append(os.path.basename(fResList1[i]))
        # Epoch from the file name
        rMJD.append(GNSSTime.doy2mjd(int(fN1[i][-7:-3]), int(fN1[i][-3:])))
    for i in range(len(fResList2)):
        fN2.append(os.path.basename(fResList2[i]))

    # Global link and correlation coefficient table
    cLink = []
    R = []
    # Interate on the first res list
    for i in range(nFile):
        if fN1[i] not in fN2:
            continue
        j = fN2.index(fN1[i])
        # Read the required res for required links from the first file
        cLink1 = []
        Res1 = []
        with open(fResList1[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                if lRng1 and cWords[8] == 'F':
                    continue
                if not lRng1 and cWords[11] == 'F':
                    continue
                if (cLink0[0] != 'ALL-ALL') and \
                        (cWords[2] + '-' + cWords[3]) not in cLink0 and \
                        (cWords[3] + '-' + cWords[2]) not in cLink0:
                    continue
                if cWords[3] > cWords[2]:
                    cTmp = cWords[2] + '-' + cWords[3]
                else:
                    cTmp = cWords[3] + '-' + cWords[2]
                if cTmp not in cLink1:
                    cLink1.append(cTmp)
                    # Epoch, res
                    Res1.append([])
                    Res1.append([])
                k = cLink1.index(cTmp)
                Res1[k * 2].append(int(cWords[14]) +
                                   float(cWords[15]) / 86400.0)
                if lRng1:
                    # Range res in cm
                    Res1[k * 2 + 1].append(float(cWords[9]) * 100)
                else:
                    # Clock res in cm
                    Res1[k * 2 + 1].append(float(cWords[12]) * 100)
        # Read the required res for required links from the second file
        cLink2 = []
        Res2 = []
        with open(fResList2[j], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                if lRng2 and cWords[8] == 'F':
                    continue
                if not lRng2 and cWords[11] == 'F':
                    continue
                if (cLink0[0] != 'ALL-ALL') and \
                        (cWords[2] + '-' + cWords[3]) not in cLink0 and \
                        (cWords[3] + '-' + cWords[2]) not in cLink0:
                    continue
                if cWords[3] > cWords[2]:
                    cTmp = cWords[2] + '-' + cWords[3]
                else:
                    cTmp = cWords[3] + '-' + cWords[2]
                # Additionally, it should be the common link with the first file
                if cTmp not in cLink1:
                    continue
                if cTmp not in cLink2:
                    cLink2.append(cTmp)
                    # Epoch, res
                    Res2.append([])
                    Res2.append([])
                k = cLink2.index(cTmp)
                Res2[k * 2].append(int(cWords[14]) +
                                   float(cWords[15]) / 86400.0)
                if lRng2:
                    # Range res in cm
                    Res2[k * 2 + 1].append(float(cWords[9]) * 100)
                else:
                    # Clock res in cm
                    Res2[k * 2 + 1].append(float(cWords[12]) * 100)
        # Cal the correlation coefficients for common links
        # Iterate on the link list from the second file as it is the common link list
        for k in range(len(cLink2)):
            if cLink2[k] not in cLink:
                # New global link
                cLink.append(cLink2[k])
                R.append([])
                for l in range(nFile):
                    R[len(cLink) - 1].append(np.nan)
            iLink = cLink.index(cLink2[k])
            # Match the two time-series
            # First, sort them along time for later matching
            ind = np.argsort(Res2[k * 2])
            t2 = np.array(Res2[k * 2])[ind]
            y2 = np.array(Res2[k * 2 + 1])[ind]
            l = cLink1.index(cLink2[k])
            ind = np.argsort(Res1[l * 2])
            t1 = np.array(Res1[l * 2])[ind]
            y1 = np.array(Res1[l * 2 + 1])[ind]
            # Then, match the two time-series by epoch
            t = []
            y = [[], []]
            for m in range(t1.size):
                for n in range(t2.size):
                    if (t1[m] - t2[n]) * 86400 > 1:
                        continue
                    elif (t1[m] - t2[n]) * 86400 < -1:
                        break
                    else:
                        t.append(t1[m])
                        y[0].append(y1[m])
                        y[1].append(y2[n])
                        break
            rCor = np.corrcoef(y)
            R[iLink][i] = rCor[0, 1]
    nLink = len(cLink)

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 4))

    for i in range(nLink):
        axs[0, 0].plot(rMJD, R[i], 'o--', ms=2, lw=0.6, label=cLink[i])

    axs[0, 0].grid(which='major', axis='y', color='darkgray',
                   linestyle='--', linewidth=0.4)
    axs[0, 0].set_axisbelow(True)

    axs[0, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    axs[0, 0].set_ylabel('Correlation', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix + OutFileSuffix + '.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResISL90(cSer, fSerList, cSat0, lClk, OutFilePrefix, OutFileSuffix):
    '''
    Plot ISL range or clock resdiual RMS for each satellite
    from different series for comparison

    lClk --- Whether for clock obs, otherwise for range obs
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nSer = len(cSer)
    cSat = []
    for i in range(nSer):
        nFile = len(fSerList[i])
        for j in range(nFile):
            with open(fSerList[i][j], mode='rt') as fOb:
                for cLine in fOb:
                    if cLine[0:6] != 'ISLRES':
                        continue
                    cWords = cLine.split()
                    if (not lClk) and cWords[8] == 'F':
                        continue
                    if lClk and cWords[11] == 'F':
                        continue
                    if cSat0[0] == 'ALL':
                        if cWords[2] not in cSat:
                            cSat.append(cWords[2])
                        if cWords[3] not in cSat:
                            cSat.append(cWords[3])
                    elif cWords[2] in cSat0:
                        if cWords[2] not in cSat:
                            cSat.append(cWords[2])
                    elif cWords[3] in cSat0:
                        if cWords[3] not in cSat:
                            cSat.append(cWords[3])
    cSat.sort()
    nSat = len(cSat)
    Mea = np.zeros((nSat, nSer))
    Sig = np.zeros((nSat, nSer))
    RMS = np.zeros((nSat, nSer))

    # Header line of the report
    strTmp = 'PRN'
    for i in range(nSer):
        if not lClk:
            # For Range obs
            strTmp = strTmp + \
                ' {: >6s} {: >6s} {: >6s}'.format('MeaR', 'STDR', 'RMSR')
        else:
            # For Clock obs
            strTmp = strTmp + \
                ' {: >6s} {: >6s} {: >6s}'.format('MeaC', 'STDC', 'RMSC')
    print(strTmp)

    for k in range(nSat):
        strTmp = cSat[k]
        for i in range(nSer):
            rEpo, Res = GetResISL(fSerList[i], cSat[k], 'ALL', False)
            nRes = Res[0].size
            if nRes == 0:
                strTmp = strTmp + ' {: >6s} {: >6s} {: >6s}'.format('', '', '')
            else:
                nObs = 0
                for j in range(nRes):
                    if not lClk:
                        if not np.isnan(Res[0, j]):
                            nObs = nObs + 1
                            RMS[k, i] = RMS[k, i] + Res[0, j] * Res[0, j]
                    else:
                        if not np.isnan(Res[5, j]):
                            nObs = nObs + 1
                            RMS[k, i] = RMS[k, i] + Res[5, j] * Res[5, j]
                if not lClk:
                    if nObs > 0:
                        Mea[k, i] = np.nanmean(Res[0])
                        Sig[k, i] = np.nanstd(Res[0])
                        RMS[k, i] = np.sqrt(RMS[k, i] / nObs)
                        strTmp = strTmp + \
                            ' {:>6.2f} {:>6.2f} {:>6.2f}'.format(
                                Mea[k, i], Sig[k, i], RMS[k, i])
                    else:
                        strTmp = strTmp + \
                            ' {: >6s} {: >6s} {: >6s}'.format('', '', '')
                else:
                    if nObs > 0:
                        Mea[k, i] = np.nanmean(Res[5])
                        Sig[k, i] = np.nanstd(Res[5])
                        RMS[k, i] = np.sqrt(RMS[k, i] / nObs)
                        strTmp = strTmp + \
                            ' {:>6.2f} {:>6.2f} {:>6.2f}'.format(
                                Mea[k, i], Sig[k, i], RMS[k, i])
                    else:
                        strTmp = strTmp + \
                            ' {: >6s} {: >6s} {: >6s}'.format('', '', '')
        print(strTmp)

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(nSat * 0.6, 4))
    x = np.arange(nSat)

    axs[0, 0].set_xlim(left=-1, right=nSat)
    axs[0, 0].set_ylim(top=12)
    axs[0, 0].set_yticks([0, 2, 4, 6, 8, 10])

    # the width of the bars
    w = 1 / (nSer + 1)
    for i in range(nSer):
        axs[0, 0].bar(x + (i - nSer / 2) * w, RMS[:, i], w,
                      align='edge', label=cSer[i])

    axs[0, 0].grid(which='major', axis='y', color='darkgray',
                   linestyle='--', linewidth=0.5)
    axs[0, 0].set_axisbelow(True)
    # Only show the legend when there are more than one solution plotted.
    if nSer > 1:
        axs[0, 0].legend(ncol=nSer, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                         framealpha=0.6, prop={'family': 'Arial', 'size': 14})

    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(
        cSat, fontdict={'fontsize': 14, 'fontname': 'Arial'})
    axs[0, 0].set_xlabel('Satellite PRNs', fontname='Arial', fontsize=16)

    axs[0, 0].set_ylabel('Residual RMS [cm]', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix + OutFileSuffix + '.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResISL91(fResList, cSat0, lClk, OutFilePrefix, OutFileSuffix):
    '''
    Plot ISL range or clock residual RMS series for specified satellites

    cSat0 --- Specified satellites list, cSat0[0]='ALL' sepcifies all satellites
     lClk --- Whether for clock obs, otherwise for range obs
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    if lClk:
        iObs = 12
    else:
        iObs = 9

    cSat = []
    RMS = []
    nFile = len(fResList)
    for i in range(nFile):
        # Get the date of this file from the file name
        YYYY = int(os.path.basename(fResList[i])[-7:-3])
        DOY = int(os.path.basename(fResList[i])[-3:])
        MO, DD = GNSSTime.doy2dom(YYYY, DOY)
        rEpo = GNSSTime.doy2mjd(YYYY, DOY)
        # Satellite set in this file
        cPRN = []
        Res = []
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:42] == 'Start time && Interval                   :':
                    continue
                elif cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                if (not lClk) and cWords[8] == 'F':
                    continue
                if lClk and cWords[11] == 'F':
                    continue
                if cSat0[0] == 'ALL':
                    if cWords[2] not in cPRN:
                        cPRN.append(cWords[2])
                        Res.append([])
                    Res[cPRN.index(cWords[2])].append(
                        float(cWords[iObs]) * 100)
                    if cWords[3] not in cPRN:
                        cPRN.append(cWords[3])
                        Res.append([])
                    Res[cPRN.index(cWords[3])].append(
                        float(cWords[iObs]) * 100)
                elif cWords[2] in cSat0:
                    if cWords[2] not in cPRN:
                        cPRN.append(cWords[2])
                        Res.append([])
                    Res[cPRN.index(cWords[2])].append(
                        float(cWords[iObs]) * 100)
                elif cWords[3] in cSat0:
                    if cWords[3] not in cPRN:
                        cPRN.append(cWords[3])
                        Res.append([])
                    Res[cPRN.index(cWords[3])].append(
                        float(cWords[iObs]) * 100)
            # Cal mean, std and rms for each satellite
            for j in range(len(cPRN)):
                if cPRN[j] not in cSat:
                    cSat.append(cPRN[j])
                    for k in range(5):
                        RMS.append([])
                iSat = cSat.index(cPRN[j])
                RMS[iSat * 5].append(rEpo)
                nRes = len(Res[j])
                RMS[iSat * 5 + 1].append(nRes)
                RMS[iSat * 5 + 2].append(np.mean(Res[j]))
                RMS[iSat * 5 + 3].append(np.std(Res[j]))
                xtmp = 0.0
                for k in range(nRes):
                    xtmp = xtmp + Res[j][k] * Res[j][k]
                RMS[iSat * 5 + 4].append(np.sqrt(xtmp / nRes))
    nSat = len(cSat)
    cSat1 = cSat.copy()
    cSat1.sort()

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 5))

    axs[0, 0].set_ylabel('Residual RMS [cm]', fontname='Arial', fontsize=16)
    # axs[0,0].set_ylim(bottom=3,top=15)
    axs[0, 0].grid(b=True, which='major', axis='y', color='darkgray', linestyle='--',
                   linewidth=0.8)
    axs[0, 0].set_axisbelow(True)
    axs[0, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)

    axs[0, 0].set_prop_cycle(color=['r', 'r', 'r', 'r', 'r', 'r', 'r', 'r', 'r',
                                    'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g', 'g',
                                    'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b'],
                             marker=['.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                     '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X',
                                     '.', 'v', '^', '<', '>', '*', 'x', 'd', 'X'])

    for i in range(nSat):
        j = cSat.index(cSat1[i])
        axs[0, 0].plot(RMS[j * 5], RMS[j * 5 + 4],
                       ls='--', lw=1, label=cSat[j])

    axs[0, 0].legend(ncol=1, loc='center left', bbox_to_anchor=(1.0, 0.5), framealpha=0.6,
                     labelspacing=0.1, borderpad=0.1, prop={'family': 'Arial', 'size': 14})
    for tl in axs[0, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix + OutFileSuffix + '.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResISL92(fResList, cSat0, lClk, OutFilePrefix, OutFileSuffix):
    '''
    Plot ISL range or clock residual RMS series for specified satellites
    Similar to PlotResISL91, but plot residuals from two sessions in a broken axis

    cSat0 --- Specified satellites list, cSat0[0]='ALL' sepcifies all satellites
     lClk --- Whether for clock obs, otherwise for range obs
    '''

    if lClk:
        iObs = 12
    else:
        iObs = 9

    cSat = []
    RMS = []
    nFile = len(fResList)
    for i in range(nFile):
        # Get the date of this file from the file name
        YYYY = int(os.path.basename(fResList[i])[-7:-3])
        DOY = int(os.path.basename(fResList[i])[-3:])
        MO, DD = GNSSTime.doy2dom(YYYY, DOY)
        rEpo = datetime.datetime(YYYY, MO, DD)
        # Satellite set in this file
        cPRN = []
        Res = []
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:42] == 'Start time && Interval                   :':
                    continue
                elif cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                if (not lClk) and cWords[8] == 'F':
                    continue
                if lClk and cWords[11] == 'F':
                    continue
                if cSat0[0] == 'ALL':
                    if cWords[2] not in cPRN:
                        cPRN.append(cWords[2])
                        Res.append([])
                    # m -> cm
                    Res[cPRN.index(cWords[2])].append(
                        float(cWords[iObs]) * 100)
                    if cWords[3] not in cPRN:
                        cPRN.append(cWords[3])
                        Res.append([])
                    Res[cPRN.index(cWords[3])].append(
                        float(cWords[iObs]) * 100)
                elif cWords[2] in cSat0:
                    if cWords[2] not in cPRN:
                        cPRN.append(cWords[2])
                        Res.append([])
                    Res[cPRN.index(cWords[2])].append(
                        float(cWords[iObs]) * 100)
                elif cWords[3] in cSat0:
                    if cWords[3] not in cPRN:
                        cPRN.append(cWords[3])
                        Res.append([])
                    Res[cPRN.index(cWords[3])].append(
                        float(cWords[iObs]) * 100)
            # Cal mean, std and rms for each satellite
            for j in range(len(cPRN)):
                if cPRN[j] not in cSat:
                    cSat.append(cPRN[j])
                    for k in range(5):
                        RMS.append([])
                iSat = cSat.index(cPRN[j])
                RMS[iSat * 5].append(rEpo)
                nRes = len(Res[j])
                RMS[iSat * 5 + 1].append(nRes)
                RMS[iSat * 5 + 2].append(np.mean(Res[j]))
                RMS[iSat * 5 + 3].append(np.std(Res[j]))
                xtmp = 0.0
                for k in range(nRes):
                    xtmp = xtmp + Res[j][k] * Res[j][k]
                RMS[iSat * 5 + 4].append(np.sqrt(xtmp / nRes))
    nSat = len(cSat)
    cSat1 = cSat.copy()
    cSat1.sort()

    fig, axs = plt.subplots(1, 2, sharey='row', squeeze=False, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.05)

    # the first session: 2019001 ~ 2019007
    axs[0, 0].set_xlim(datetime.datetime(2018, 12, 31),
                       datetime.datetime(2019, 1, 8))
    axs[0, 0].spines['right'].set_visible(False)
    axs[0, 0].yaxis.tick_left()
    axs[0, 0].set_ylabel('Residual RMS [cm]', fontname='Arial', fontsize=16)
    axs[0, 0].set_xlabel('Date of 2019', fontname='Arial',
                         fontsize=16, loc='right')
    axs[0, 0].tick_params(axis='x', which='major', labelsize=14, pad=15)
    axs[0, 0].tick_params(axis='x', which='minor', labelsize=12)
    axs[0, 0].tick_params(axis='y', which='major', labelsize=14)

    # the second session: 2019335 ~ 2019365
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
        j = cSat.index(cSat1[i])
        axs[0, 0].plot(RMS[j * 5], RMS[j * 5 + 4], label=cSat[j])
        axs[0, 1].plot(RMS[j * 5], RMS[j * 5 + 4], label=cSat[j])

    d = 0.015
    axs[0, 0].plot((1 - d, 1 + d), (1 - d, 1 + d), '-k',
                   transform=axs[0, 0].transAxes, clip_on=False)
    axs[0, 0].plot((1 - d, 1 + d), (-d, +d), '-k',
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

    axs[0, 1].plot((-d, +d), (1 - d, 1 + d), '-k',
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

    strTmp = OutFilePrefix + OutFileSuffix
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResISL93(fResList, lClk, OutFilePrefix, OutFileSuffix):
    '''
    Plot ISL range or clock residual overall RMS series
    from two sessions in a broken axis

    lClk --- Whether for clock obs, otherwise for range obs
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    if lClk:
        iObs = 12
    else:
        iObs = 9

    RMS = [[], [], [], [], []]
    nFile = len(fResList)
    for i in range(nFile):
        # Get the date of this file from the file name
        YYYY = int(os.path.basename(fResList[i])[-7:-3])
        DOY = int(os.path.basename(fResList[i])[-3:])
        MO, DD = GNSSTime.doy2dom(YYYY, DOY)
        RMS[0].append(datetime.datetime(YYYY, MO, DD))
        Res = []
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:42] == 'Start time && Interval                   :':
                    continue
                elif cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                if (not lClk) and cWords[8] == 'F':
                    continue
                if lClk and cWords[11] == 'F':
                    continue
                # m -> cm
                Res.append(float(cWords[iObs]) * 100)
            # Cal mean, std and rms for each satellite
            nRes = len(Res)
            RMS[1].append(nRes)
            RMS[2].append(np.mean(Res))
            RMS[3].append(np.std(Res))
            xtmp = 0.0
            for k in range(nRes):
                xtmp = xtmp + Res[k] * Res[k]
            RMS[4].append(np.sqrt(xtmp / nRes))

    fig, axs = plt.subplots(1, 2, sharey='row', squeeze=False, figsize=(12, 5))
    fig.subplots_adjust(wspace=0.05)

    # the first session: 2019001 ~ 2019007
    axs[0, 0].set_xlim(datetime.datetime(2018, 12, 31),
                       datetime.datetime(2019, 1, 8))
    axs[0, 0].spines['right'].set_visible(False)
    axs[0, 0].yaxis.tick_left()
    axs[0, 0].set_ylabel('Residual RMS [cm]', fontname='Arial', fontsize=16)
    axs[0, 0].set_xlabel('Date of 2019', fontname='Arial',
                         fontsize=16, loc='right')
    axs[0, 0].tick_params(axis='x', which='major', labelsize=14, pad=15)
    axs[0, 0].tick_params(axis='x', which='minor', labelsize=12)
    axs[0, 0].tick_params(axis='y', which='major', labelsize=14)
    axs[0, 0].grid(b=True, which='both', axis='y', color='darkgray', linestyle='--',
                   linewidth=0.8)
    axs[0, 0].set_axisbelow(True)

    # the second session: 2019335 ~ 2019365
    axs[0, 1].set_xlim(datetime.datetime(2019, 11, 30),
                       datetime.datetime(2020, 1, 1))
    axs[0, 1].spines['left'].set_visible(False)
    axs[0, 1].tick_params(axis='x', which='major', labelsize=14, pad=15)
    axs[0, 1].tick_params(axis='x', which='minor', labelsize=12)
    axs[0, 1].tick_params(axis='y', which='both', left=False)
    axs[0, 1].grid(b=True, which='both', axis='y', color='darkgray', linestyle='--',
                   linewidth=0.8)
    axs[0, 1].set_axisbelow(True)

    # the width of the bars
    w = 1 / (1 + 1)
    x = []
    for i in range(len(RMS[0])):
        x.append(RMS[0][i] + datetime.timedelta(days=(0 - 1 / 2) * w))
    axs[0, 0].bar(x, RMS[4], w, align='edge')
    axs[0, 1].bar(x, RMS[4], w, align='edge')

    d = 0.015
    axs[0, 0].plot((1 - d, 1 + d), (1 - d, 1 + d), '-k',
                   transform=axs[0, 0].transAxes, clip_on=False)
    axs[0, 0].plot((1 - d, 1 + d), (-d, +d), '-k',
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

    axs[0, 1].plot((-d, +d), (1 - d, 1 + d), '-k',
                   transform=axs[0, 1].transAxes, clip_on=False)
    axs[0, 1].plot((-d, +d), (-d, +d), '-k',
                   transform=axs[0, 1].transAxes, clip_on=False)
    MO = mdates.MonthLocator()
    DD = mdates.DayLocator(interval=4)
    axs[0, 1].xaxis.set_major_locator(MO)
    axs[0, 1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    axs[0, 1].xaxis.set_minor_locator(DD)
    axs[0, 1].xaxis.set_minor_formatter(mdates.DateFormatter('%d'))
    for tl in axs[0, 1].get_xticklabels(which='both'):
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix + OutFileSuffix + '.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResISL94(fResList, lClk, OutFilePrefix, OutFileSuffix):
    '''
    Plot ISL range or clock residual overall RMS series
    Similar to PlotResISL93 but in non-broken axis

    lClk --- Whether for clock obs, otherwise for range obs
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    if lClk:
        iObs = 12
    else:
        iObs = 9

    RMS = [[], [], [], [], []]
    nFile = len(fResList)
    for i in range(nFile):
        # Get the date of this file from the file name
        YYYY = int(os.path.basename(fResList[i])[-7:-3])
        DOY = int(os.path.basename(fResList[i])[-3:])
        MO, DD = GNSSTime.doy2dom(YYYY, DOY)
        RMS[0].append(GNSSTime.doy2mjd(YYYY, DOY))
        Res = []
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:42] == 'Start time && Interval                   :':
                    continue
                elif cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                if (not lClk) and cWords[8] == 'F':
                    continue
                if lClk and cWords[11] == 'F':
                    continue
                # m -> cm
                Res.append(float(cWords[iObs]) * 100)
            # Cal mean, std and rms for each file
            nRes = len(Res)
            RMS[1].append(nRes)
            RMS[2].append(np.mean(Res))
            RMS[3].append(np.std(Res))
            xtmp = 0.0
            for k in range(nRes):
                xtmp = xtmp + Res[k] * Res[k]
            RMS[4].append(np.sqrt(xtmp / nRes))
    # Report the average of statistics
    print('{: >8d} {: >8.2f} {: >8.2f} {: >8.2f}'.format(len(RMS[0]),
                                                         np.mean(RMS[2]), np.mean(RMS[3]), np.mean(RMS[4])))

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 5))

    axs[0, 0].grid(b=True, which='major', axis='y', color='darkgray', linestyle='--',
                   linewidth=0.8)
    axs[0, 0].set_axisbelow(True)

    # the width of the bars
    w = 1 / (1 + 1)
    x = []
    for i in range(len(RMS[0])):
        x.append(RMS[0][i] + (0 - 1 / 2) * w)
    axs[0, 0].bar(x, RMS[4], w, align='edge')

    axs[0, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    axs[0, 0].set_ylabel('Residual RMS [cm]', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix + OutFileSuffix + '.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotNadISL1(fResList, cLink0, OutFilePrefix, OutFileSuffix):
    '''
    Plot the nadir angles extracted from the res-files for specific
    ISL link(s) in a single axis

    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    cLink = []
    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                # Whether this link is required
                lOK = True
                if cLink0[0] != 'ALL-ALL':
                    if cLink0[0][0:3] == 'EXL':
                        # Records for exlcuding satellites
                        for j in range(len(cLink0)):
                            if cLink0[j][4:7] == cWords[2] or cLink0[j][4:7] == cWords[3]:
                                lOK = False
                                break
                    elif cLink0[0][0:3] == 'INC':
                        # Records for including satellites
                        # Whether both of the two satellites of this link is found
                        # from the inclusive satellite list
                        lFound = [False, False]
                        for j in range(len(cLink0)):
                            if cLink0[j][4:7] == cWords[2]:
                                lFound[0] = True
                            elif cLink0[j][4:7] == cWords[3]:
                                lFound[1] = True
                        if (not lFound[0]) or (not lFound[1]):
                            # Either of the two satellites not found
                            lOK = False
                    elif cLink0[0][0:3] == 'LNK':
                        # One and only one end is within the specified satellite set
                        lFound = [False, False]
                        for j in range(len(cLink0)):
                            if cLink0[j][4:7] == cWords[2]:
                                lFound[0] = True
                            elif cLink0[j][4:7] == cWords[3]:
                                lFound[1] = True
                        if (not lFound[0] and not lFound[1]) or (lFound[0] and lFound[1]):
                            # Neither or both of the two satellites are found
                            lOK = False
                    else:
                        # Records for reuired links
                        if (cWords[2] + '-' + cWords[3] not in cLink0) and \
                                (cWords[3] + '-' + cWords[2] not in cLink0):
                            lOK = False
                if not lOK:
                    continue
                if cWords[3] > cWords[2]:
                    cTmp = cWords[2] + '-' + cWords[3]
                else:
                    cTmp = cWords[3] + '-' + cWords[2]
                if cTmp not in cLink:
                    cLink.append(cTmp)
    cLink.sort()
    nLink = len(cLink)

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(12, 3))
    fig.subplots_adjust(hspace=0.1)

    # Report to the terminal
    print('{: <7s} {: >5s} {: >5s} {: >5s} {: >5s}'.format(
        'Link', 'Min1', 'Max1', 'Min2', 'Max2'))
    # The overall Max and Min
    rMin = np.zeros(2)
    rMax = np.zeros(2)
    rMin[:] = 90.0
    rMax[:] = -1.0
    for i in range(nLink):
        rEpo, Res = GetResISL(fResList, cLink[i][0:3], cLink[i][4:7], False)
        rMax1 = np.amax(Res[1])
        rMin1 = np.amin(Res[1])
        rMax2 = np.amax(Res[2])
        rMin2 = np.amin(Res[2])
        print('{: <7s} {: >5.2f} {: >5.2f} {: >5.2f} {: >5.2f}'.format(cLink[i],
                                                                       rMin1, rMax1, rMin2, rMax2))
        rMin[0] = min(rMin[0], rMin1)
        rMin[1] = min(rMin[1], rMin2)
        rMax[0] = max(rMax[0], rMax1)
        rMax[1] = max(rMax[1], rMax2)

        axs[0, 0].plot(rEpo, Res[1], '.b', ms=1)
        axs[0, 0].plot(rEpo, Res[2], '.b', ms=1)
    print('{: <7s} {: >5.2f} {: >5.2f} {: >5.2f} {: >5.2f}'.format('ALL-ALL',
                                                                   rMin[0], rMax[0], rMin[1], rMax[1]))

    axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
    axs[0, 0].set_axisbelow(True)
    axs[0, 0].set_ylabel('Nadir angle [deg]', fontname='Arial', fontsize=16)
    # axs[0,0].set_ylim(bottom=10,top=70)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    axs[0, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    axs[0, 0].xaxis.set_major_formatter('{x:7.1f}')
    for tl in axs[0, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix + OutFileSuffix + '.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotNadISL2(fResList, OutFilePrefix, OutFileSuffix):
    '''
    Plot the nadir angles of all ISL links one by one, each link
    within a sub-axis
    '''

    # Find all links
    cLink = []
    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'ISLRES':
                    continue
                cWords = cLine.split()
                if cWords[3] > cWords[2]:
                    cTmp = cWords[2] + '-' + cWords[3]
                else:
                    cTmp = cWords[3] + '-' + cWords[2]
                if cTmp not in cLink:
                    cLink.append(cTmp)
    cLink.sort()
    nLink = len(cLink)

    fig, axs = plt.subplots(nLink, 1, sharex='col',
                            squeeze=False, figsize=(12, nLink * 3))
    fig.subplots_adjust(hspace=0.1)
    # Formatting tick label
    for i in range(nLink):
        rEpo, Res = GetResISL(fResList, cLink[i][0:3], cLink[i][4:7], False)
        rMax1 = np.amax(Res[1])
        rMin1 = np.amin(Res[1])
        rMax2 = np.amax(Res[2])
        rMin2 = np.amin(Res[2])
        # First satellite
        axs[i, 0].plot(rEpo, Res[1], '.r', ms=2)
        # Second satellite
        axs[i, 0].plot(rEpo, Res[2], '.b', ms=2)
        axs[i, 0].text(0.02, 0.98, cLink[i], transform=axs[i, 0].transAxes, ha='left', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
        axs[i, 0].text(0.98, 0.98, '[{:>5.2f}, {:>5.2f}]-[{:>5.2f}, {:>5.2f}]'.format(rMin1, rMax1, rMin2, rMax2),
                       transform=axs[i, 0].transAxes, ha='right', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
        axs[i, 0].set_ylabel('Nadir angle [deg]',
                             fontname='Arial', fontsize=16)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        # Report to the terminal
        strTmp = '{} {:>5.2f} {:>5.2f}'.format(
            cLink[i], rMax1 - rMin1, rMax2 - rMin2)
        print(strTmp)
    axs[i, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    axs[i, 0].xaxis.set_major_formatter('{x:7.1f}')
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix + OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResGNS1(fResList, cSta, cSat, lEpo, OutFilePrefix, OutFileSuffix):
    '''
    Plot LC && PC residuals for GNSS observations between cSta and cSat

    lEpo --- Whether plot w.r.t. epoch time, otherwise elevation
    cSta --- Specified station or 'ALL'
    cSat --- Specified satellite or 'ALL'
    '''

    ResLC, cSta0, cSat0 = GetResGNS(fResList, cSta, cSat, 1, 1000)
    ResPC, cSta0, cSat0 = GetResGNS(fResList, cSta, cSat, 2, 10)

    fig, axs = plt.subplots(2, 1, sharex='col', squeeze=False, figsize=(10, 5))
    # fig.subplots_adjust(hspace=0.1)

    if len(ResLC[0]) > 0:
        # LC residuals
        Mea = np.mean(ResLC[1])
        Sig = np.std(ResLC[1])
        if lEpo:
            axs[0, 0].plot(ResLC[0], ResLC[1], '.r', ms=2)
        else:
            axs[0, 0].plot(ResLC[2], ResLC[1], '.r', ms=2)
        axs[0, 0].text(0.02, 0.98, cSta + '-' + cSat, transform=axs[0, 0].transAxes,
                       ha='left', va='top', fontdict={'fontsize': 14, 'fontname': 'Arial'})
        axs[0, 0].text(0.98, 0.98, '{:>7.1f}+/-{:>7.1f}'.format(Mea, Sig),
                       transform=axs[0, 0].transAxes, ha='right', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial'})
        axs[0, 0].set_ylabel('LC [mm]', fontname='Arial', fontsize=16)
        for tl in axs[0, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        if lEpo:
            axs[0, 0].xaxis.set_major_formatter('{x:7.1f}')
    else:
        print('No LC observations between ' + cSta + ' and ' + cSat)

    if len(ResPC[0]) > 0:
        # PC residuals
        Mea = np.mean(ResPC[1])
        Sig = np.std(ResPC[1])
        if lEpo:
            axs[1, 0].plot(ResPC[0], ResPC[1], '.b', ms=2)
        else:
            axs[1, 0].plot(ResPC[2], ResPC[1], '.b', ms=2)
        axs[1, 0].text(0.02, 0.98, cSta + '-' + cSat, transform=axs[1, 0].transAxes,
                       ha='left', va='top', fontdict={'fontsize': 14, 'fontname': 'Arial'})
        axs[1, 0].text(0.98, 0.98, '{:>7.1f}+/-{:>7.1f}'.format(Mea, Sig),
                       transform=axs[1, 0].transAxes, ha='right', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial'})
        axs[1, 0].set_ylabel('PC [dm]', fontname='Arial', fontsize=16)
        for tl in axs[1, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        if lEpo:
            axs[1, 0].xaxis.set_major_formatter('{x:7.1f}')
            axs[1, 0].set_xlabel('Modified Julian Day',
                                 fontname='Arial', fontsize=16)
        else:
            axs[1, 0].set_xlabel(
                'Elevation [deg]', fontname='Arial', fontsize=16)
    else:
        print('No PC observations between ' + cSta + ' and ' + cSat)

    for tl in axs[1, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix + OutFileSuffix
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResGNS2(fResList, lEpo, OutFilePrefix, OutFileSuffix):
    '''
    Plot LC && PC residuals for each satellite

    lEpo --- Whether plot w.r.t. epoch time, otherwise elevation
    '''

    cSat = []
    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'RnxRES':
                    continue
                cWords = cLine.split()
                if cWords[3] not in cSat:
                    cSat.append(cWords[3])
    cSat.sort()
    nSat = len(cSat)

    fig, axs = plt.subplots(nSat * 2, 1, sharex='col',
                            squeeze=False, figsize=(8, nSat * 2 * 2))
    # fig.subplots_adjust(hspace=0.1)

    for i in range(nSat):
        ResLC, cSta0, cSat0 = GetResGNS(fResList, 'ALL ', cSat[i], 1, 1000)
        ResPC, cSta0, cSat0 = GetResGNS(fResList, 'ALL ', cSat[i], 2, 10)

        # LC residuals
        Mea = np.mean(ResLC[1])
        Sig = np.std(ResLC[1])
        if lEpo:
            axs[2 * i, 0].plot(ResLC[0], ResLC[1], '.r', ms=2)
        else:
            axs[2 * i, 0].plot(ResLC[2], ResLC[1], '.r', ms=2)
        axs[2 * i, 0].text(0.02, 0.98, cSat[i], transform=axs[2 * i, 0].transAxes,
                           ha='left', va='top', fontdict={'fontsize': 14, 'fontname': 'Arial'})
        axs[2 * i, 0].text(0.98, 0.98, '{:>7.1f}+/-{:>7.1f}'.format(Mea, Sig),
                           transform=axs[2 * i, 0].transAxes, ha='right', va='top',
                           fontdict={'fontsize': 14, 'fontname': 'Arial'})
        axs[2 * i, 0].set_ylabel('LC [mm]', fontname='Arial', fontsize=16)
        for tl in axs[2 * i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        if not lEpo:
            axs[2 * i, 0].set_xlim(left=0, right=90)
        else:
            axs[2 * i, 0].xaxis.set_major_formatter('{x:7.1f}')
        # PC residuals
        Mea = np.mean(ResPC[1])
        Sig = np.std(ResPC[1])
        if lEpo:
            axs[2 * i + 1, 0].plot(ResPC[0], ResPC[1], '.b', ms=2)
        else:
            axs[2 * i + 1, 0].plot(ResPC[2], ResPC[1], '.b', ms=2)
        axs[2 * i + 1, 0].text(0.02, 0.98, cSat[i], transform=axs[2 * i + 1, 0].transAxes,
                               ha='left', va='top', fontdict={'fontsize': 14, 'fontname': 'Arial'})
        axs[2 * i + 1, 0].text(0.98, 0.98, '{:>7.1f}+/-{:>7.1f}'.format(Mea, Sig),
                               transform=axs[2 * i + 1, 0].transAxes, ha='right', va='top',
                               fontdict={'fontsize': 14, 'fontname': 'Arial'})
        axs[2 * i + 1, 0].set_ylabel('PC [dm]', fontname='Arial', fontsize=16)
        for tl in axs[2 * i + 1, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        if not lEpo:
            axs[2 * i + 1, 0].set_xlim(left=0, right=90)
        else:
            axs[2 * i + 1, 0].xaxis.set_major_formatter('{x:7.1f}')
    if lEpo:
        axs[2 * i + 1, 0].set_xlabel('Modified Julian Day',
                                     fontname='Arial', fontsize=16)
    else:
        axs[2 * i + 1, 0].set_xlabel('Elevation [deg]',
                                     fontname='Arial', fontsize=16)
    for tl in axs[2 * i + 1, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix + OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResGNS3(fResList, cSat, lEpo, OutFilePrefix, OutFileSuffix):
    '''
    Plot LC && PC residuals for a specific satellite w.r.t. each station

    lEpo --- Whether plot w.r.t. epoch time, otherwise elevation
    '''

    cSta = []
    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'RnxRES':
                    continue
                cWords = cLine.split()
                if cWords[3] != cSat:
                    continue
                if cWords[2] not in cSta:
                    cSta.append(cWords[2])
    cSta.sort()
    nSta = len(cSta)

    fig, axs = plt.subplots(nSta * 2, 1, sharex='col',
                            squeeze=False, figsize=(8, nSta * 2 * 2))
    fig.subplots_adjust(hspace=0.1)
    # Formatting tick label

    for i in range(nSta):
        ResLC, cSta0, cSat0 = GetResGNS(fResList, cSta[i], cSat, 1, 1000)
        ResPC, cSta0, cSat0 = GetResGNS(fResList, cSta[i], cSat, 2, 10)

        # LC residuals
        if len(ResLC[1]) != 0:
            Mea = np.mean(ResLC[1])
            Sig = np.std(ResLC[1])
            if lEpo:
                axs[2 * i, 0].plot(ResLC[0], ResLC[1], '.r')
            else:
                axs[2 * i, 0].plot(ResLC[2], ResLC[1], '.r')
            axs[2 * i, 0].text(0.95, 0.95, 'Mea={:>9.4f}, STD={:>9.4f}'.format(Mea, Sig),
                               transform=axs[2 * i, 0].transAxes, ha='right', va='top')
        axs[2 * i, 0].text(0.05, 0.95, cSta[i], transform=axs[2 * i, 0].transAxes,
                           ha='left', va='top')
        axs[2 * i, 0].set_ylabel('LC [mm]')
        if not lEpo:
            axs[2 * i, 0].set_xlim(left=0, right=90)
        else:
            axs[2 * i, 0].xaxis.set_major_formatter('{x:8.2f}')

        # PC residuals
        if len(ResPC[1]) != 0:
            Mea = np.mean(ResPC[1])
            Sig = np.std(ResPC[1])
            if lEpo:
                axs[2 * i + 1, 0].plot(ResPC[0], ResPC[1], '.b')
            else:
                axs[2 * i + 1, 0].plot(ResPC[2], ResPC[1], '.b')
            axs[2 * i + 1, 0].text(0.95, 0.95, 'Mea={:>9.4f}, STD={:>9.4f}'.format(Mea, Sig),
                                   transform=axs[2 * i + 1, 0].transAxes, ha='right', va='top')
        axs[2 * i + 1, 0].text(0.05, 0.95, cSta[i], transform=axs[2 * i + 1, 0].transAxes,
                               ha='left', va='top')
        axs[2 * i + 1, 0].set_ylabel('PC [dm]')
        if not lEpo:
            axs[2 * i + 1, 0].set_xlim(left=0, right=90)
        else:
            axs[2 * i + 1, 0].xaxis.set_major_formatter('{x:8.2f}')
    if lEpo:
        axs[2 * i + 1, 0].set_xlabel('MJD')
    else:
        axs[2 * i + 1, 0].set_xlabel('Elev [deg]')

    strTmp = OutFilePrefix + OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResGNS4(fResList, lEpo, OutFilePrefix, OutFileSuffix):
    '''
    Plot LC && PC residuals for each station

    lEpo --- Whether plot w.r.t. epoch time, otherwise elevation
    '''

    cSta = []
    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'RnxRES':
                    continue
                cWords = cLine.split()
                if cWords[2] not in cSta:
                    cSta.append(cWords[2])
    cSta.sort()
    nSta = len(cSta)

    fig, axs = plt.subplots(nSta * 2, 1, sharex='col',
                            squeeze=False, figsize=(8, nSta * 2 * 2))
    fig.subplots_adjust(hspace=0.1)
    # Formatting tick label

    for i in range(nSta):
        ResLC, cSta0, cSat0 = GetResGNS(fResList, cSta[i], 'ALL', 1, 1000)
        ResPC, cSta0, cSat0 = GetResGNS(fResList, cSta[i], 'ALL', 2, 10)

        # LC residuals
        Mea = np.mean(ResLC[1])
        Sig = np.std(ResLC[1])
        if lEpo:
            axs[2 * i, 0].plot(ResLC[0], ResLC[1], '.r')
        else:
            axs[2 * i, 0].plot(ResLC[2], ResLC[1], '.r')
        axs[2 * i, 0].text(0.05, 0.95, cSta[i], transform=axs[2 * i, 0].transAxes,
                           ha='left', va='top')
        axs[2 * i, 0].text(0.95, 0.95, 'Mea={:>9.4f}, STD={:>9.4f}'.format(Mea, Sig),
                           transform=axs[2 * i, 0].transAxes, ha='right', va='top')
        axs[2 * i, 0].set_ylabel('LC [mm]')
        if not lEpo:
            axs[2 * i, 0].set_xlim(left=0, right=90)
        else:
            axs[2 * i, 0].xaxis.set_major_formatter('{x:8.2f}')

        # PC residuals
        Mea = np.mean(ResPC[1])
        Sig = np.std(ResPC[1])
        if lEpo:
            axs[2 * i + 1, 0].plot(ResPC[0], ResPC[1], '.b')
        else:
            axs[2 * i + 1, 0].plot(ResPC[2], ResPC[1], '.b')
        axs[2 * i + 1, 0].text(0.05, 0.95, cSta[i], transform=axs[2 * i + 1, 0].transAxes,
                               ha='left', va='top')
        axs[2 * i + 1, 0].text(0.95, 0.95, 'Mea={:>9.4f}, STD={:>9.4f}'.format(Mea, Sig),
                               transform=axs[2 * i + 1, 0].transAxes, ha='right', va='top')
        axs[2 * i + 1, 0].set_ylabel('PC [dm]')
        if lEpo:
            axs[2 * i + 1, 0].xaxis.set_major_formatter('{x:8.2f}')
        else:
            axs[2 * i + 1, 0].set_xlim(left=0, right=90)
    if lEpo:
        axs[2 * i + 1, 0].set_xlabel('MJD')
    else:
        axs[2 * i + 1, 0].set_xlabel('Elev [deg]')

    strTmp = OutFilePrefix + OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResGNS5(fResList, cSta, nObsTyp, lEpo, OutFilePrefix, OutFileSuffix):
    '''
    Plot LC && PC residuals for a specific station w.r.t. each satellite

    cSta --- Name of the specified station
 nObsTyp --- Number of obs type

    lEpo --- Whether plot w.r.t. epoch time, otherwise elevation
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # Get the satellite list observed by this station
    cSat = []
    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'RnxRES':
                    continue
                cWords = cLine.split()
                if cWords[2] != cSta:
                    continue
                if cWords[3] not in cSat:
                    cSat.append(cWords[3])
    cSat.sort()
    nSat = len(cSat)

    nObsTyp = 4
    cObsTyp = ['LC12', 'PC12', 'LC13', 'PC13']
    cUntObs = ['mm', 'dm', 'mm', 'dm']
    rUntObs = [1000, 10, 1000, 10]

    fig, axs = plt.subplots(nSat, nObsTyp, sharex='col',
                            squeeze=False, figsize=(nObsTyp * 4, nSat * 2))
    # fig.subplots_adjust(hspace=0.1)

    for i in range(nSat):
        axs[i, 0].text(0.02, 0.98, cSat[i], transform=axs[i, 0].transAxes, ha='left', va='top',
                       family='Arial', size=10, weight='bold')
        for j in range(nObsTyp):
            Res, cSta0, cSat0 = GetResGNS(
                fResList, cSta, cSat[i], j + 1, rUntObs[j])
            axs[i, j].text(0.50, 0.98, '#={:>5d}'.format(len(Res[1])),
                           transform=axs[i, j].transAxes, ha='center', va='top',
                           family='Arial', size=10)
            if len(Res[1]) != 0:
                Mea = np.mean(Res[1])
                Sig = np.std(Res[1])
                if lEpo:
                    axs[i, j].plot(Res[0], Res[1], '.r', ms=3)
                else:
                    axs[i, j].plot(Res[2], Res[1], '.r', ms=3)
                axs[i, j].text(0.98, 0.98, '{:>7.2f}+/-{:>7.2f}'.format(Mea, Sig),
                               transform=axs[i, j].transAxes, ha='right', va='top',
                               family='Arial', size=10)
            axs[i, j].set_ylabel(cObsTyp[j] + ' [' + cUntObs[j] + ']',
                                 fontname='Arial', fontsize=8)
            for tl in axs[i, j].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(8)
            # For the last row
            if i == (nSat - 1):
                axs[i, j].ticklabel_format(
                    axis='x', useOffset=False, useMathText=True)
                if lEpo:
                    axs[i, j].set_xlabel('MJD', fontname='Arial', fontsize=8)
                else:
                    axs[i, j].set_xlabel(
                        'Elev [deg]', fontname='Arial', fontsize=8)
                for tl in axs[i, j].get_xticklabels():
                    tl.set_fontname('Arial')
                    tl.set_fontsize(8)

    strTmp = OutFilePrefix + OutFileSuffix + '.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotResGNS6(fResList, cSat0):
    '''
    Count the number of obs for each satellite
    '''

    cSat = []
    nObs = []
    for i in range(len(fResList)):
        with open(fResList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:6] != 'RnxRES':
                    continue
                cWords = cLine.split()
                if (cSat0[0] != 'ALL') and (cWords[3] not in cSat0):
                    continue
                if cWords[3] not in cSat:
                    cSat.append(cWords[3])
                    nObs.append(0)
                iSat = cSat.index(cWords[3])
                nObsType = int(cWords[8])
                for iObsType in range(nObsType):
                    # Only read valid observations (Flag<=2)
                    Flag = int(cWords[8 + 3 * iObsType + 1])
                    if Flag < 2:
                        nObs[iSat] = nObs[iSat] + 1
    cSat1 = cSat.copy()
    cSat1.sort()
    for i in range(len(cSat)):
        j = cSat.index(cSat1[i])
        strTmp = cSat[j] + ' {:8d}'.format(nObs[j])
        print(strTmp)


def PlotTideDif(fTide1, fTide2):
    '''
    '''
    # Get the common station list
    cSta1 = []
    with open(fTide1, mode='rt') as fOb:
        for cLine in fOb:
            if len(cLine) < 5:
                continue
            cWords = cLine.split()
            if cWords[0] in cSta1:
                continue
            cSta1.append(cWords[0])
    cSta = []
    with open(fTide2, mode='rt') as fOb:
        for cLine in fOb:
            if len(cLine) < 5:
                continue
            cWords = cLine.split()
            if cWords[0] not in cSta1 or cWords[0] in cSta:
                continue
            cSta.append(cWords[0])
    cSta.sort()
    nSta = len(cSta)

    fig, axs = plt.subplots(nSta, 1, sharex='col', figsize=(8, nSta * 4))
    fig.subplots_adjust(hspace=0.1)

    for iSta in range(nSta):
        rEpo1 = []
        rEpo2 = []
        X1 = [[] for i in range(9)]
        X2 = [[] for i in range(9)]
        with open(fTide1, mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine) < 5:
                    continue
                cWords = cLine.split()
                if cWords[0] != cSta[iSta]:
                    continue
                t = int(cWords[1]) + float(cWords[2]) / 86400.0
                if len(rEpo1) > 0 and abs(rEpo1[-1] - t) * 86400.0 < 1.0:
                    X1[0][-1] = float(cWords[3])
                    X1[1][-1] = float(cWords[4])
                    X1[2][-1] = float(cWords[5])

                    X1[3][-1] = float(cWords[6])
                    X1[4][-1] = float(cWords[7])
                    X1[5][-1] = float(cWords[8])

                    X1[6][-1] = float(cWords[9])
                    X1[7][-1] = float(cWords[10])
                    X1[8][-1] = float(cWords[11])
                else:
                    rEpo1.append(t)
                    X1[0].append(float(cWords[3]))
                    X1[1].append(float(cWords[4]))
                    X1[2].append(float(cWords[5]))

                    X1[3].append(float(cWords[6]))
                    X1[4].append(float(cWords[7]))
                    X1[5].append(float(cWords[8]))

                    X1[6].append(float(cWords[9]))
                    X1[7].append(float(cWords[10]))
                    X1[8].append(float(cWords[11]))

        with open(fTide2, mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine) < 5:
                    continue
                cWords = cLine.split()
                if cWords[0] != cSta[iSta]:
                    continue
                t = int(cWords[1]) + float(cWords[2]) / 86400.0
                if len(rEpo2) > 0 and abs(rEpo2[-1] - t) * 86400.0 < 1.0:
                    X2[0][-1] = float(cWords[3]) * 1000
                    X2[1][-1] = float(cWords[4]) * 1000
                    X2[2][-1] = float(cWords[5]) * 1000

                    X2[3][-1] = float(cWords[6])
                    X2[4][-1] = float(cWords[7])
                    X2[5][-1] = float(cWords[8])

                    X2[6][-1] = float(cWords[9])
                    X2[7][-1] = float(cWords[10])
                    X2[8][-1] = float(cWords[11])
                else:
                    rEpo2.append(t)
                    X2[0].append(float(cWords[3]) * 1000)
                    X2[1].append(float(cWords[4]) * 1000)
                    X2[2].append(float(cWords[5]) * 1000)

                    X2[3].append(float(cWords[6]))
                    X2[4].append(float(cWords[7]))
                    X2[5].append(float(cWords[8]))

                    X2[6].append(float(cWords[9]))
                    X2[7].append(float(cWords[10]))
                    X2[8].append(float(cWords[11]))

        # Cal the dif
        rEpo = []
        Diff = [[] for i in range(9)]
        for j in range(len(rEpo1)):
            for k in range(len(rEpo2)):
                if abs(rEpo1[j] - rEpo2[k]) * 86400.0 < 1.0:
                    rEpo.append(rEpo1[j])
                    Diff[0].append((X1[0][j] - X2[0][k]) * 100)
                    Diff[1].append((X1[1][j] - X2[1][k]) * 100)
                    Diff[2].append((X1[2][j] - X2[2][k]) * 100)

                    Diff[3].append((X1[3][j] - X2[3][k]) * 100)
                    Diff[4].append((X1[4][j] - X2[4][k]) * 100)
                    Diff[5].append((X1[5][j] - X2[5][k]) * 100)

                    Diff[6].append((X1[6][j] - X2[6][k]) * 100)
                    Diff[7].append((X1[7][j] - X2[7][k]) * 100)
                    Diff[8].append((X1[8][j] - X2[8][k]) * 100)
                    break
        # Plot
        axs[iSta].plot(rEpo, Diff[0], '.r')
        axs[iSta].plot(rEpo, Diff[1], '.g')
        axs[iSta].plot(rEpo, Diff[2], '.b')

        axs[iSta].text(0.05, 0.95, cSta[iSta], ha='left',
                       va='top', transform=axs[iSta].transAxes)

    # plt.show()
    fig.savefig('D:/Code/PROJECT/WORK2019090/TideDiff.pdf',
                bbox_inches='tight')
    plt.close(fig)


def PlotCorISL(fCor, cLink0, rMJD, dIntv, OutFilePrefix, OutFileSuffix):
    '''
    fCor --- Lomb-Scargle fitting models for each ISL link
  cLink0 --- List of specified ISL links, set cLink0[0]='ALL-ALL' to plot all links
    rMJD --- Start and end of sampling time
   dIntv --- the interval of sampling, in days

    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # Read the model file
    cLink1, X = GetCorISL(fCor)
    # List of required ISL links
    cLink = []
    # Index list of those required ISL links in the model
    iLink = []
    for i in range(len(cLink1)):
        if cLink0[0] != 'ALL-ALL' and (cLink1[i] not in cLink0) and (
                cLink1[i][4:7] + '-' + cLink1[i][0:3] not in cLink0):
            continue
        cLink.append(cLink1[i])
        iLink.append(i)
    nLink = len(cLink)

    fig, axs = plt.subplots(nLink, 1, sharex='col',
                            squeeze=False, figsize=(12, nLink * 1.5))
    for i in range(nLink):
        t = np.arange(rMJD[0], rMJD[1], dIntv)
        y = np.zeros(t.size)
        # Note that, nterm=2
        dPhi = 2 * np.pi * X[1][iLink[i]]
        for j in range(t.size):
            y[j] = X[0][iLink[i]] + X[2][iLink[i]] + \
                X[3][iLink[i]] * np.sin(dPhi * t[j]) + \
                X[4][iLink[i]] * np.cos(dPhi * t[j]) + \
                X[5][iLink[i]] * np.sin(2 * dPhi * t[j]) + \
                X[6][iLink[i]] * np.cos(2 * dPhi * t[j])

        axs[i, 0].plot(t, y, '.-g', ms=2)

        axs[i, 0].axhline(color='darkgray', ls='dashed', lw=0.2, alpha=0.5)
        axs[i, 0].text(0.02, 0.98, cLink[i], transform=axs[i, 0].transAxes, ha='left', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
        axs[i, 0].set_ylabel('[cm]', fontname='Arial', fontsize=16)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

    axs[i, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    axs[i, 0].xaxis.set_major_formatter('{x:7.1f}')
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix + OutFileSuffix + '.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix + OutFileSuffix + '.pdf'
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
        sys.exit('Unknow environment: ' + cWhere)
    print('Run On ' + cWhere)

    cSer = []
    fSerList = []

    InFilePrefix = os.path.join(cWrkPre0, r'PRO_2019001_2020366_WORK/')
    # InFilePrefix=r'D:/Code/PROJECT/WORK2019351_ERROR/'

    OutFilePrefix = os.path.join(cDskPre0, r'PRO_2019001_2020366/I061/RES/')
    # OutFilePrefix=r'D:/Code/PROJECT/WORK2019347_ERROR/'

    fISLCor = os.path.join(
        cDskPre0, r'PRO_2019001_2020366/I610/RES/ResLink_2019335_2019365_LS_2')

    fOmCList = glob.glob(r'D:/Code/PROJECT/WORK2022054/'+'omc_2022054')
    # OutFileSuffix = 'omc_2022054'
    # PlotOmCISL1(fOmCList, False, True, OutFilePrefix, OutFileSuffix)

    fResList = glob.glob(r'D:/Code/PROJECT/WORK2022054/' + 'res_2022054')
    # OutFileSuffix = 'res_2022054_ALL_C19'
    # PlotResISL01(fResList, True, ['ALL-C19'], True, 0, OutFilePrefix, OutFileSuffix)

    # OutFileSuffix = 'res_2022054_ALL_ALL'
    # PlotResISL20(fResList, True, ['ALL-ALL'], False, 0, OutFilePrefix, OutFileSuffix)

    OutFileSuffix = 'res_2022054_N'
    # PlotResISL61(fResList, 59633.000000000, 59633.003472222,
    #              OutFilePrefix, OutFileSuffix)

    fLogList = glob.glob(InFilePrefix + 'I614/WORK2019335/logs/log_lsq_1')

    # fResList = glob.glob(InFilePrefix + 'I010/2019/RES_POD/res_20193??')
    # cSer.append('None')
    # fSerList.append(fResList)

    # fResList = glob.glob(InFilePrefix+'I021/2019/RES_POD/res_20193??')
    # cSer.append('Model 1')
    # fSerList.append(fResList)

    # fResList = glob.glob(InFilePrefix+'I022/2019/RES_POD/res_20193??')
    # cSer.append('Model 2')
    # fSerList.append(fResList)

    # fResList = glob.glob(InFilePrefix+'I023/2019/RES_POD/res_20193??')
    # cSer.append('Model 3')
    # fSerList.append(fResList)

    fResList = glob.glob(InFilePrefix+'I061/2019/RES_POD/res_20193??')
    cSer.append('Model 1')
    fSerList.append(fResList)

    fResList = glob.glob(InFilePrefix+'I062/2019/RES_POD/res_20193??')
    cSer.append('Model 2')
    fSerList.append(fResList)

    fResList = glob.glob(InFilePrefix+'I063/2019/RES_POD/res_20193??')
    cSer.append('Model 3')
    fSerList.append(fResList)

    # fResList = glob.glob(InFilePrefix+'J646/2019/RES_POD/res_20193??_fix')
    # cSer.append('8 cm')
    # fSerList.append(fResList)

    # fResList = glob.glob(InFilePrefix+'J647/2019/RES_POD/res_20193??_fix')
    # cSer.append('12 cm')
    # fSerList.append(fResList)

    # fResList = glob.glob(InFilePrefix+'J648/2019/RES_POD/res_20193??_fix')
    # cSer.append('15 cm')
    # fSerList.append(fResList)

    # fResList = glob.glob(InFilePrefix+'J651/2019/RES_POD/res_20193??_fix')
    # cSer.append('18 cm')
    # fSerList.append(fResList)

    # fResList = glob.glob(InFilePrefix+'J653/2019/RES_POD/res_20193??_fix')
    # cSer.append('20 cm')
    # fSerList.append(fResList)

    # fResList = glob.glob(InFilePrefix+'C01/2019/RES_POD/res_20193??_fix')
    # cSer.append('No ISL')
    # fSerList.append(fResList)

    # InFilePrefix=os.path.join(cWrkPre0,r'PRO_2019001_2020366_WORK/I657/2019/RES_POD/')
    # fResList=glob.glob(InFilePrefix+'res_20193??_fix')
    # cSer.append('24 cm'); fSerList.append(fResList)

    # InFilePrefix=os.path.join(cWrkPre0,r'PRO_2019001_2020366_WORK/I658/2019/RES_POD/')
    # fResList=glob.glob(InFilePrefix+'res_20193??_fix')
    # cSer.append('25 cm'); fSerList.append(fResList)

    # InFilePrefix=os.path.join(cWrkPre0,r'PRO_2019001_2020366_WORK/I640/2019/RES_POD/')
    # fResList=glob.glob(InFilePrefix+'res_20193??_fix')
    # cSer.append('No ISL'); fSerList.append(fResList)

    # OutFileSuffix='Res_YKRO'
    # PlotResGNS5(fResList,'YKRO',4,True,OutFilePrefix,OutFileSuffix)

    # PlotResGNS6(fResList,['ALL'])

    # OutFileSuffix = 'LinkMod_2019335_2019365_subset'
    # PlotCorISL(fISLCor, ['C21-C29', 'C22-C30', 'C28-C30'],
    #            [58818, 58848], 0.5, OutFilePrefix, OutFileSuffix)

    # OutFileSuffix = 'ISLCor_2019335_subset'
    # PlotResISL10(fLogList, ['C21-C29', 'C22-C30',
    #              'C28-C30'], 0, OutFilePrefix, OutFileSuffix)

    # OutFileSuffix = 'LinkRes_2019335_subset'
    # PlotResISL00(fResList, True, [
    #              'C21-C29', 'C22-C30', 'C28-C30'], 0, OutFilePrefix, OutFileSuffix)
    # OutFileSuffix='AllRes_2019335'
    # PlotResISL21(fResList,True,['ALL-ALL'],1,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='Hist_2019335'
    # PlotResISL22(fResList,['ALL-ALL'],0,OutFilePrefix,OutFileSuffix)
    # PlotResISL23(fResList,['ALL-ALL'],1,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='ISL_res_2019335_Sat.png'
    # PlotResISL40(fResList,0,False,['ALL'],OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='ResSat_2019335_2019365_1_subset'
    # PlotResISL41(fResList,False,['C23','C26','C28','C30','C32','C33'],False,2,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='ISLResSat_2019335_2'
    # PlotResISL42(fResList,False,['ALL'],True,4,OutFilePrefix,OutFileSuffix)

    # OutFileSuffix='GNS_res_2019335_fixclk.png'
    # PlotResGNS1(fResList,'WUH2','ALL',True,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='GNS_res_2019335_2.png'
    # PlotResGNS2(fResList,True,OutFilePrefix,OutFileSuffix)

    # OutFileSuffix='LinkRMS_2019335'
    # PlotResISL30(fResList,['ALL'],True,OutFilePrefix,OutFileSuffix)

    # OutFileSuffix='ISLNadir_C38_C39'
    # PlotNadISL1(fResList,['INC-C38','INC-C39'],OutFilePrefix,OutFileSuffix)
    # PlotNadISL2(fResList,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='ISLResSat_2019001.png'
    # PlotResISL40(fResList,0,False,['ALL'],OutFilePrefix,OutFileSuffix)

    OutFileSuffix = 'ISLResSatRMS_2019335_2019365'
    # PlotResISL91(fResList, ['ALL'], True, OutFilePrefix, OutFileSuffix)
    # OutFileSuffix='ISLResSatRMSAll_new.png'
    # PlotResISL92(fResList,['ALL'],False,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='ISLResRMSAll'
    # PlotResISL93(fResList,False,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='ResRMS_2019335_2019365'
    # PlotResISL94(fResList,False,OutFilePrefix,OutFileSuffix)

    # OutFileSuffix = 'ISLResSatRMS_2019335_2019365_0'
    # PlotResISL90(cSer, fSerList, ['ALL'], True, OutFilePrefix, OutFileSuffix)

    OutFileSuffix = 'ResLink_2019335_2019365_LS_2_subset'
    # cLink=['C19-C25','C21-C28','C22-C37','C24-C36','C28-C30','C34-C37']
    cLink = ['C21-C29', 'C22-C30', 'C28-C30']
    # PlotResISL70(fResList, cLink, True, 2, 1, 0.5,
    #              OutFilePrefix, OutFileSuffix)
    OutFileSuffix = 'ResLink_2019335_2019365_LS_2'
    # PlotResISL72(fResList, ['ALL-ALL'], True, 2, OutFilePrefix, OutFileSuffix)

    OutFileSuffix = 'ResLink_2019335_2019365_LS_Comp'
    # cLink=['C19-C21','C20-C22','C24-C29','C25-C27','C26-C28','C30-C35','C36-C37']
    PlotResISL71(cSer, fSerList, ['C19-C25', 'C24-C36'],
                 False, False, OutFilePrefix, OutFileSuffix)

    OutFileSuffix = 'PostSig0_Comp_APSig'
    # PlotRes0(cSer, fSerList, OutFilePrefix, OutFileSuffix)
    # PlotRes1(cSer, fSerList, OutFilePrefix, OutFileSuffix)

    # OutFileSuffix='ResLink_DePer'
    # PlotResISL80(cSer,fSerList,True,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='ResCor_C19_C21'
    # PlotResISL81(fResList1,True,fResList2,False,['C19-C21'],OutFilePrefix,OutFileSuffix)
