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
import math

# Related third party imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Local application/library specific imports
from PySRC.miscellaneous import GNSSTime


def GetOrbdif(fList, cSat0, dMax, lReport):
    '''
    Read data (in mm) from the orb-diff files

    cSat0 --- List of excluded satellites
     dMax --- If greater than zero, exclude epochs any component of which
              exceeds this value in cm (in the sense of absolute value)
  lReport --- Whether report the excluded records to the terminal
    '''

    # The whole satellite set in all files
    cSat = []
    # Number of satellites in each file
    nSat = np.zeros(len(fList), dtype=np.uint8)
    # Number of epochs in each file
    nEpo = np.zeros(len(fList), dtype=np.uint16)
    for i in range(len(fList)):
        iLine = 0
        with open(fList[i], mode='rt') as fOb:
            for cLine in fOb:
                iLine = iLine+1
                if iLine == 1:
                    cWords = cLine.split()
                    nSat[i] = int(cWords[0])
                    nEpo[i] = int(cWords[1])
                elif iLine == 2:
                    cWords = cLine.split()
                    for j in range(len(cWords)):
                        if cWords[j] in cSat or cWords[j] in cSat0:
                            continue
                        cSat.append(cWords[j])
                elif iLine > 2:
                    break
    cSat.sort()
    mSat = len(cSat)

    mEpo = np.sum(nEpo)
    rEpo = np.zeros(mEpo)
    X = np.zeros((mEpo, mSat*3))
    X[:, :] = np.nan
    xtemp = np.zeros(3)

    mEpo = -1
    for i in range(len(fList)):
        iLine = 0
        # Index of satellites in global satellites set
        iSat = np.zeros(nSat[i], dtype=np.uint8)
        iSat[:] = 255
        with open(fList[i], mode='rt') as fOb:
            for cLine in fOb:
                iLine = iLine+1
                if iLine == 2:
                    cWords = cLine.split()
                    for j in range(nSat[i]):
                        if cWords[j] in cSat:
                            iSat[j] = cSat.index(cWords[j])
                elif iLine == 1 or iLine == 3:
                    continue
                elif iLine > 3+nEpo[i]:
                    break
                else:
                    mEpo = mEpo+1
                    cWords = cLine.split()
                    rEpo[mEpo] = int(cWords[1])+float(cWords[2])/86400
                    for j in range(nSat[i]):
                        if iSat[j] == 255:
                            continue
                        xtemp[0] = float(cWords[3+j*3])
                        xtemp[1] = float(cWords[3+j*3+1])
                        xtemp[2] = float(cWords[3+j*3+2])
                        # Exclude and report satellites that have large dif values
                        if (dMax > 0) and (np.fabs(xtemp[0])/10 >= dMax or
                                           np.fabs(xtemp[1])/10 >= dMax or
                                           np.fabs(xtemp[2])/10 >= dMax):
                            if lReport:
                                # Report to the terminal
                                print(fList[i]+' '+cSat[iSat[j]]+' {:12.6f} {:>12.2f} {:>12.2f} {:>12.2f}'.format(
                                      rEpo[mEpo], xtemp[0], xtemp[1], xtemp[2]))
                            continue
                        X[mEpo, iSat[j]*3] = xtemp[0]
                        X[mEpo, iSat[j]*3+1] = xtemp[1]
                        X[mEpo, iSat[j]*3+2] = xtemp[2]

    return mEpo+1, cSat, rEpo, X


def PlotDifRMS0(fList, yMax, OutFilePrefix, OutFileSuffix):
    '''
    Plot RMS of orb diff for all satellites

    yMax --- top limit of the y-axis in cm
    '''
    cSat0 = []
    nEpo, cSat, rEpo, Dif = GetOrbdif(fList, cSat0, -1, False)
    nSat = len(cSat)
    # Output file
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    fOut = open(OutFilePrefix+os.path.splitext(OutFileSuffix)[0], 'w')

    RMS = np.zeros((nSat, 4))
    RMS[:, :] = np.nan
    fOut.write('PRN   Along   Cross  Radial     1-D\n')
    for i in range(nSat):
        for j in range(3):
            nValid = 0
            xtmp = 0
            for k in range(nEpo):
                if np.isnan(Dif[k, 3*i+j]):
                    continue
                nValid = nValid+1
                xtmp = xtmp+Dif[k, 3*i+j]*Dif[k, 3*i+j]
            if nValid > 0:
                # mm -> cm
                RMS[i, j] = np.sqrt(xtmp/nValid)/10
            else:
                RMS[i, j] = np.nan
        if np.isnan(RMS[i, 0]):
            RMS[i, 3] = np.nan
            continue
        if np.isnan(RMS[i, 1]):
            RMS[i, 3] = np.nan
            continue
        if np.isnan(RMS[i, 2]):
            RMS[i, 3] = np.nan
            continue
        # 1D RMS
        RMS[i, 3] = np.sqrt((RMS[i, 0]*RMS[i, 0] +
                            RMS[i, 1]*RMS[i, 1] +
                            RMS[i, 2]*RMS[i, 2])/3.0)
        fOut.write('{} {:7.2f} {:7.2f} {:7.2f} {:7.2f}\n'.format(cSat[i],
                   RMS[i, 0], RMS[i, 1], RMS[i, 2], RMS[i, 3]))
    fOut.write('{} {:7.2f} {:7.2f} {:7.2f} {:7.2f}\n'.format('Avg', np.nanmean(RMS[:, 0]),
               np.nanmean(RMS[:, 1]), np.nanmean(RMS[:, 2]), np.nanmean(RMS[:, 3])))
    fOut.close()

    x = np.arange(nSat)
    fig, axs = plt.subplots(1, 1, figsize=(nSat*0.6, 4))

    axs.set_xlim(left=-1, right=nSat)
    axs.set_ylim(top=yMax)
    # axs.set_yticks([5,10,15,20],minor=True)
    # axs.set_yticks([2,4,6,8,10,12,14],minor=True)
    # axs.minorticks_on()
    for tl in axs.get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    # the width of the bars
    w = 0.2
    axs.bar(x-2*w, RMS[:, 0], w, align='edge', color='r', label='Along')
    axs.bar(x - w, RMS[:, 1], w, align='edge', color='g', label='Cross')
    axs.bar(x, RMS[:, 2], w, align='edge', color='b', label='Radial')
    axs.bar(x + w, RMS[:, 3], w, align='edge', color='dimgrey', label='1D')
    axs.legend(ncol=4, loc='upper center', prop={
               'family': 'Arial', 'size': 14})
    axs.grid(which='both', axis='y', c='darkgray', ls='--', lw=0.8)
    axs.set_axisbelow(True)
    axs.set_ylabel('RMS [cm]', fontname='Arial', fontsize=16)
    axs.set_xlabel('Satellite PRNs', fontname='Arial', fontsize=16)
    axs.set_xticks(x)
    axs.set_xticklabels(cSat, fontname='Arial', fontsize=14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotDifRMS1(fList, yMax, yLab, iComp2, yMax2, yLab2, OutFilePrefix, OutFileSuffix):
    '''
    Similar to PlotDifRMS0, but capable of choosing some components to be
    plotted along the second y-axis (at the right side)

      yMax --- top limit of the first y-axis (at the left side) in cm
      yLab --- label of the first y-axis (at the left side)
    iComp2 --- list of components to be plotted along the second y-axis (at the right side)
               0, Along; 1, Cross; 2, Radial; 3, 1D
     yMax2 --- top limit of the second y-axis (at the right side) in m
     yLab2 --- label of the second y-axis (at the right side)
    '''
    cSat0 = []
    nEpo, cSat, rEpo, Dif = GetOrbdif(fList, cSat0, -1, False)
    nSat = len(cSat)
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    RMS = np.zeros((nSat, 4))
    RMS[:, :] = np.nan
    for i in range(nSat):
        for j in range(3):
            nValid = 0
            xtmp = 0
            for k in range(nEpo):
                if np.isnan(Dif[k, 3*i+j]):
                    continue
                nValid = nValid+1
                xtmp = xtmp+Dif[k, 3*i+j]*Dif[k, 3*i+j]
            if nValid > 0:
                # mm -> cm
                RMS[i, j] = np.sqrt(xtmp/nValid)/10
            else:
                RMS[i, j] = np.nan
        if np.isnan(RMS[i, 0]):
            RMS[i, 3] = np.nan
            continue
        if np.isnan(RMS[i, 1]):
            RMS[i, 3] = np.nan
            continue
        if np.isnan(RMS[i, 2]):
            RMS[i, 3] = np.nan
            continue
        # 1D RMS
        RMS[i, 3] = np.sqrt((RMS[i, 0]*RMS[i, 0] +
                            RMS[i, 1]*RMS[i, 1] +
                            RMS[i, 2]*RMS[i, 2])/3.0)

    x = np.arange(nSat)
    # Bar colors and labels
    c = ['r', 'g', 'b', 'dimgrey']
    lb = ['Along', 'Cross', 'Radial', '1D']
    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(nSat*0.6, 4))
    axs[0, 0].set_xlim(left=-1, right=nSat)
    axs[0, 0].set_ylim(top=yMax)
    # axs[0,0].set_yticks([5,10,15,20],minor=True)
    # axs[0,0].set_yticks([2,4,6,8,10,12,14],minor=True)
    # axs[0,0].minorticks_on()
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    axs[0, 0].grid(which='both', axis='y', c='darkgray', ls='--', lw=0.8)
    axs[0, 0].set_axisbelow(True)
    axs[0, 0].set_ylabel(yLab+' RMS [cm]', fontname='Arial', fontsize=16)

    # the second y-aixs, in meter
    axe = axs[0, 0].twinx()
    axe.set_ylim(top=yMax2)
    axe.set_ylabel(yLab2+' RMS [m]', fontname='Arial', fontsize=16)
    for tl in axe.get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    # the width of the bars
    w = 1/(4+1)
    # handles for each component bar plot
    h = []
    for i in range(4):
        if i not in iComp2:
            # Plot along the first y-axis
            hd = axs[0, 0].bar(x+(i-4/2)*w, RMS[:, i], w,
                               align='edge', color=c[i], label=lb[i])
        else:
            # Plot along the second y-axis, in meter
            hd = axe.bar(x+(i-4/2)*w, RMS[:, i]*1e-2,
                         w, align='edge', color=c[i], label=lb[i])
        h.append(hd)
    fig.legend(handles=h, ncol=4, loc='upper center', bbox_to_anchor=(0.5, 0.9),
               prop={'family': 'Arial', 'size': 14})

    axs[0, 0].set_xlabel('Satellite PRNs', fontname='Arial', fontsize=16)
    axs[0, 0].set_xticks(x)
    axs[0, 0].set_xticklabels(cSat, fontname='Arial', fontsize=14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotDifRMS2(cSer, fListSer, dMax, lReport, iComp, cSat0, yMax, OutFilePrefix, OutFileSuffix):
    '''
    Plot RMS of orb diff for specified (common) satellites
    from different solutions

    dMax --- If greater than zero, exclude epochs any component of which
             exceeds this value in cm (in the sense of absolute value)
 lReport --- Whether report the excluded records to the terminal

    iComp --- the component to be plotted
    # 0, along-track
    # 1, cross-track
    # 2, radial
    # 3, 1D
    # 4, all components in one figure
    cSat0 --- specified satellite list
              ['ALL'], all satellites
              ['CXX'], only one GNSS
              ['EXL'], excluded PRN list
              ['C01'], specified PRN list
    yMax --- max of y-axis for each component
    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    cComp = ['Along', 'Cross', 'Radial', '1D']

    nSer = len(cSer)
    RMS = []
    for iSer in range(nSer):
        # Along, Cross, Radial, 1D
        for i in range(4):
            RMS.append([])

    cSat = []
    for iSer in range(nSer):
        nEpo, cSatSer, rEpo, Dif = GetOrbdif(fListSer[iSer], [], dMax, lReport)
        nSatSer = len(cSatSer)
        for i in range(nSatSer):
            # Filter out the non-specified satellite list
            if cSat0[0] != 'ALL':
                if cSat0[0] == 'EXL':
                    # Excluded satellite list
                    if cSatSer[i] in cSat0[1:]:
                        continue
                elif cSat0[0][1:3] == 'XX':
                    # Specified GNSS system
                    if cSatSer[i][0:1] != cSat0[0][0:1]:
                        continue
                else:
                    # Specified PRN list
                    if cSatSer[i] not in cSat0:
                        continue

            if cSatSer[i] not in cSat:
                cSat.append(cSatSer[i])
                for j in range(nSer):
                    for k in range(4):
                        RMS[j*4+k].append(np.nan)
            iSat = cSat.index(cSatSer[i])
            # Cal the RMS for each component
            for j in range(3):
                nValid = 0
                xtmp = 0
                for k in range(nEpo):
                    if np.isnan(Dif[k, 3*i+j]):
                        continue
                    nValid = nValid+1
                    xtmp = xtmp+Dif[k, 3*i+j]*Dif[k, 3*i+j]
                if nValid > 0:
                    # mm -> cm
                    RMS[iSer*4+j][iSat] = np.sqrt(xtmp/nValid)/10
            if np.isnan(RMS[iSer*4][iSat]) or \
               np.isnan(RMS[iSer*4+1][iSat]) or \
               np.isnan(RMS[iSer*4+2][iSat]):
                RMS[iSer*4+3][iSat] = np.nan
                continue
            # Cal the 1D RMS
            RMS[iSer*4+3][iSat] = np.sqrt((RMS[iSer*4][iSat]*RMS[iSer*4][iSat] +
                                           RMS[iSer*4+1][iSat]*RMS[iSer*4+1][iSat] +
                                           RMS[iSer*4+2][iSat]*RMS[iSer*4+2][iSat])/3.0)
    nSat = len(cSat)
    cSat1 = cSat.copy()
    cSat1.sort()
    # Report to the terminal
    strTmp = '{: <3s}'.format('PRN')
    for j in range(nSer):
        strTmp = strTmp+' {: >8s} {: >8s} {: >8s} {: >8s}'.format(cSer[j]+'_A',
                                                                  cSer[j]+'_C', cSer[j]+'_R', cSer[j]+'_1D')
    print(strTmp)
    for iSat in range(nSat):
        i = cSat.index(cSat1[iSat])
        strTmp = '{: <3s}'.format(cSat[i])
        for j in range(nSer):
            strTmp = strTmp+' {: >8.2f} {: >8.2f} {: >8.2f} {: >8.2f}'.format(RMS[j*4][i],
                                                                              RMS[j*4+1][i], RMS[j*4+2][i], RMS[j*4+3][i])
        print(strTmp)
    # The overall mean
    strTmp = '{: <3s}'.format('Mea')
    for j in range(nSer):
        strTmp = strTmp+' {: >8.2f} {: >8.2f} {: >8.2f} {: >8.2f}'.format(np.nanmean(RMS[j*4]),
                                                                          np.nanmean(RMS[j*4+1]), np.nanmean(RMS[j*4+2]), np.nanmean(RMS[j*4+3]))
    print(strTmp)

    # Number of Cols for legend
    if nSer <= 5:
        nColLG = nSer
    else:
        nColLG = 5

    x = np.arange(nSat)
    if iComp != 4:
        # Only one specified component
        fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(nSat*0.6, 4))

        # the width of the bars
        w = 1/(nSer+1)
        for i in range(nSer):
            y = np.zeros(nSat)
            for j in range(nSat):
                y[j] = RMS[i*4+iComp][cSat.index(cSat1[j])]
            axs[0, 0].bar(x+(i-nSer/2)*w, y, w, align='edge', label=cSer[i])

        axs[0, 0].legend(ncol=nColLG, loc='upper center', bbox_to_anchor=(0.5, 1.0),
                         prop={'family': 'Arial', 'size': 14})
        axs[0, 0].grid(which='both', axis='y', color='darkgray', linestyle='--',
                       linewidth=0.8)
        axs[0, 0].set_axisbelow(True)
        axs[0, 0].set_ylim(top=yMax[iComp])
        # axs[0,0].set_yticks([2,4,6,8,10,12,14],minor=True)
        axs[0, 0].yaxis.set_major_formatter('{x: >3.0f}')
        axs[0, 0].set_ylabel(cComp[iComp]+' RMS [cm]',
                             fontname='Arial', fontsize=16)
        axs[0, 0].set_xlabel('Satellite PRNs', fontname='Arial', fontsize=16)
        axs[0, 0].set_xlim(left=-1, right=nSat)
        axs[0, 0].set_xticks(x)
        axs[0, 0].set_xticklabels(
            cSat1, fontdict={'fontsize': 14, 'fontname': 'Arial'})
        for tl in axs[0, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
    else:
        # All 4 components
        fig, axs = plt.subplots(
            4, 1, sharex='col', squeeze=False, figsize=(nSat*0.6, 4*4))

        # the width of the bars
        w = 1/(nSer+1)
        for k in range(4):
            for i in range(nSer):
                y = np.zeros(nSat)
                for j in range(nSat):
                    y[j] = RMS[i*4+k][cSat.index(cSat1[j])]
                axs[k, 0].bar(x+(i-nSer/2)*w, y, w,
                              align='edge', label=cSer[i])
            axs[k, 0].grid(which='major', axis='y', color='darkgray', linestyle='--',
                           linewidth=0.8)
            axs[k, 0].set_axisbelow(True)
            axs[k, 0].set_ylim(top=yMax[k])
            # axs[k,0].set_yticks([2,4,6,8,10,12,14],minor=True)
            axs[k, 0].yaxis.set_major_formatter('{x: >3.0f}')
            axs[k, 0].set_ylabel(cComp[k]+' RMS [cm]',
                                 fontname='Arial', fontsize=16)
            for tl in axs[k, 0].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
        axs[0, 0].legend(ncol=nColLG, loc='lower center', bbox_to_anchor=(0.5, 1.0),
                         prop={'family': 'Arial', 'size': 14})
        axs[k, 0].set_xlabel('Satellite PRNs', fontname='Arial', fontsize=16)
        axs[k, 0].set_xlim(left=-1, right=nSat)
        axs[k, 0].set_xticks(x)
        axs[k, 0].set_xticklabels(
            cSat1, fontdict={'fontsize': 14, 'fontname': 'Arial'})

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotDifRMS3(cSer, fListSer, iSerR, dMax, lReport, iComp, cSat0,
                yMaxL, yMaxR, ColorL, ColorR, OutFilePrefix, OutFileSuffix):
    '''
    Similar to PlotDifRMS2, but capable of choosing some series to be plotted along the second
    y-axis (at the right side). However, note that, the left y-axis use meter as the unit.


   iSerR --- Index list of series that should be plotted along the right y-axis
    dMax --- If greater than zero, exclude epochs any component of which
             exceeds this value in cm (in the sense of absolute value)
 lReport --- Whether report the excluded records to the terminal

   iComp --- the component to be plotted
    # 0, along-track
    # 1, cross-track
    # 2, radial
    # 3, 1D
    # 4, all components in one figure
    cSat0 --- specified satellite list
              ['ALL'], all satellites
              ['CXX'], only one GNSS
              ['EXL'], excluded PRN list
              ['C01'], specified PRN list
   yMaxL --- max of left y-axis for each component, in meter
   yMaxR --- max of right y-axis for each component, in centimeter
  ColorL --- Color of the left y-axis
  ColorR --- Color of the right y-axis
    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    cComp = ['Along', 'Cross', 'Radial', '1D']

    nSer = len(cSer)
    RMS = []
    for iSer in range(nSer):
        # Along, Cross, Radial, 1D
        for i in range(4):
            RMS.append([])

    cSat = []
    for iSer in range(nSer):
        nEpo, cSatSer, rEpo, Dif = GetOrbdif(fListSer[iSer], [], dMax, lReport)
        nSatSer = len(cSatSer)
        for i in range(nSatSer):
            # Filter out the non-specified satellite list
            if cSat0[0] != 'ALL':
                if cSat0[0] == 'EXL':
                    # Excluded satellite list
                    if cSatSer[i] in cSat0[1:]:
                        continue
                elif cSat0[0][1:3] == 'XX':
                    # Specified GNSS system
                    if cSatSer[i][0:1] != cSat0[0][0:1]:
                        continue
                else:
                    # Specified PRN list
                    if cSatSer[i] not in cSat0:
                        continue
            if cSatSer[i] not in cSat:
                cSat.append(cSatSer[i])
                for j in range(nSer):
                    for k in range(4):
                        RMS[j*4+k].append(np.nan)
            iSat = cSat.index(cSatSer[i])
            # Cal the RMS for each component
            for j in range(3):
                nValid = 0
                xtmp = 0
                for k in range(nEpo):
                    if np.isnan(Dif[k, 3*i+j]):
                        continue
                    nValid = nValid+1
                    xtmp = xtmp+Dif[k, 3*i+j]*Dif[k, 3*i+j]
                if nValid > 0:
                    # mm -> cm
                    RMS[iSer*4+j][iSat] = np.sqrt(xtmp/nValid)/10
            if np.isnan(RMS[iSer*4][iSat]) or \
               np.isnan(RMS[iSer*4+1][iSat]) or \
               np.isnan(RMS[iSer*4+2][iSat]):
                RMS[iSer*4+3][iSat] = np.nan
                continue
            # Cal the 1D RMS
            RMS[iSer*4+3][iSat] = np.sqrt((RMS[iSer*4][iSat]*RMS[iSer*4][iSat] +
                                           RMS[iSer*4+1][iSat]*RMS[iSer*4+1][iSat] +
                                           RMS[iSer*4+2][iSat]*RMS[iSer*4+2][iSat])/3.0)
    nSat = len(cSat)
    cSat1 = cSat.copy()
    cSat1.sort()
    # Report to the terminal
    strTmp = '{: <3s}'.format('PRN')
    for j in range(nSer):
        strTmp = strTmp+' {: >8s} {: >8s} {: >8s} {: >8s}'.format(cSer[j]+'_A',
                                                                  cSer[j]+'_C', cSer[j]+'_R', cSer[j]+'_1D')
    print(strTmp)
    for iSat in range(nSat):
        i = cSat.index(cSat1[iSat])
        strTmp = '{: <3s}'.format(cSat[i])
        for j in range(nSer):
            strTmp = strTmp+' {: >8.2f} {: >8.2f} {: >8.2f} {: >8.2f}'.format(RMS[j*4][i],
                                                                              RMS[j*4+1][i], RMS[j*4+2][i], RMS[j*4+3][i])
        print(strTmp)
    # The overall mean
    strTmp = '{: <3s}'.format('Mea')
    for j in range(nSer):
        strTmp = strTmp+' {: >8.2f} {: >8.2f} {: >8.2f} {: >8.2f}'.format(np.nanmean(RMS[j*4]),
                                                                          np.nanmean(RMS[j*4+1]), np.nanmean(RMS[j*4+2]), np.nanmean(RMS[j*4+3]))
    print(strTmp)

    # Number of Cols for legend
    if nSer <= 5:
        nColLG = nSer
    else:
        nColLG = 5

    x = np.arange(nSat)
    if iComp != 4:
        # Only one specified component
        fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(nSat*0.6, 4))
        # Whether having specified some series to be plotted along the right y-axis
        if len(iSerR) != 0:
            # the second y-aixs, in meter
            axe = axs[0, 0].twinx()

        # the width of the bars
        w = 1/(nSer+1)
        # handles for each component bar plot
        h = []
        for i in range(nSer):
            y = np.zeros(nSat)
            for j in range(nSat):
                y[j] = RMS[i*4+iComp][cSat.index(cSat1[j])]
            # Cycle the color. Note, no more than 10 series
            strTmp = 'C{:1d}'.format(i)
            if i in iSerR:
                # Plot along the right y-axis, in cm
                hd = axe.bar(x+(i-nSer/2)*w, y, w, color=strTmp,
                             align='edge', label=cSer[i])
            else:
                # Plot along the left y-axis, cm -> m
                hd = axs[0, 0].bar(x+(i-nSer/2)*w, y*1e-2, w, color=strTmp,
                                   align='edge', label=cSer[i])
            h.append(hd)
        fig.legend(handles=h, ncol=nColLG, loc='upper center', bbox_to_anchor=(0.5, 0.9),
                   prop={'family': 'Arial', 'size': 14})
        axs[0, 0].grid(which='both', axis='y', color='darkgray', linestyle='--',
                       linewidth=0.8)
        axs[0, 0].set_axisbelow(True)

        if len(iSerR) != 0:
            axe.set_ylim(top=yMaxR[iComp])
            axe.yaxis.set_major_formatter('{x: >3.0f}')
            axe.set_ylabel(cComp[iComp]+' RMS [cm]', c=ColorR,
                           fontname='Arial', fontsize=16)
            for tl in axe.get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
                tl.set_color(ColorR)

        axs[0, 0].set_ylim(top=yMaxL[iComp])
        axs[0, 0].yaxis.set_major_formatter('{x: >3.0f}')
        axs[0, 0].set_ylabel(cComp[iComp]+' RMS [m]', c=ColorL,
                             fontname='Arial', fontsize=16)
        for tl in axs[0, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
            tl.set_color(ColorL)
        axs[0, 0].set_xlabel('Satellite PRNs', fontname='Arial', fontsize=16)
        axs[0, 0].set_xlim(left=-1, right=nSat)
        axs[0, 0].set_xticks(x)
        axs[0, 0].set_xticklabels(
            cSat1, fontdict={'fontsize': 14, 'fontname': 'Arial'})
    else:
        # All 4 components
        fig, axs = plt.subplots(
            4, 1, sharex='col', squeeze=False, figsize=(nSat*0.6, 4*4))

        # the width of the bars
        w = 1/(nSer+1)
        for k in range(4):
            if k == 0:
                h = []
            # Whether having specified some series to be plotted along the right y-axis
            if len(iSerR) != 0:
                # the second y-aixs, in meter
                axe = axs[k, 0].twinx()
            for i in range(nSer):
                y = np.zeros(nSat)
                for j in range(nSat):
                    y[j] = RMS[i*4+k][cSat.index(cSat1[j])]
                # Cycle the color. Note, no more than 10 series
                strTmp = 'C{:1d}'.format(i)
                if i in iSerR:
                    # Plot along the right y-axis, in cm
                    hd = axe.bar(x+(i-nSer/2)*w, y, w, color=strTmp,
                                 align='edge', label=cSer[i])
                else:
                    # Plot along the left y-axis, cm -> m
                    hd = axs[k, 0].bar(x+(i-nSer/2)*w, y*1e-2, w, color=strTmp,
                                       align='edge', label=cSer[i])
                if k == 0:
                    h.append(hd)
            axs[k, 0].grid(which='major', axis='y', color='darkgray', linestyle='--',
                           linewidth=0.8)
            axs[k, 0].set_axisbelow(True)

            axs[k, 0].set_ylim(top=yMaxL[k])
            axs[k, 0].yaxis.set_major_formatter('{x: >3.0f}')
            axs[k, 0].set_ylabel(cComp[k]+' RMS [m]', c=ColorL,
                                 fontname='Arial', fontsize=16)
            for tl in axs[k, 0].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
                tl.set_color(ColorL)

            if len(iSerR) != 0:
                axe.set_ylim(top=yMaxR[k])
                axe.yaxis.set_major_formatter('{x: >3.0f}')
                axe.set_ylabel(cComp[k]+' RMS [cm]', c=ColorR,
                               fontname='Arial', fontsize=16)
                for tl in axe.get_yticklabels():
                    tl.set_fontname('Arial')
                    tl.set_fontsize(14)
                    tl.set_color(ColorR)
        fig.legend(handles=h, ncol=nColLG, loc='lower center', bbox_to_anchor=(0.5, 0.8),
                   prop={'family': 'Arial', 'size': 14})
        axs[k, 0].set_xlabel('Satellite PRNs', fontname='Arial', fontsize=16)
        axs[k, 0].set_xlim(left=-1, right=nSat)
        axs[k, 0].set_xticks(x)
        axs[k, 0].set_xticklabels(
            cSat1, fontdict={'fontsize': 14, 'fontname': 'Arial'})

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotDifSer0(fList, iAng, fGeoAngPath, cSat0, yMax, OutFilePrefix, OutFileSuffix):
    '''
    Plot time series of orb diff for specified satellites

    iAng --- Whether and how to plot along geometric angle
             # 0, do not plot any geometric angles
             # 1, plot beta angle at the right y-axes
             # 2, plot orb diff radial component in a second
                  "beta vs delta u" axes
    yMax --- if > 0, set the limit of the y-axis
    '''

    nEpo, cSat1, rEpo, Dif = GetOrbdif(fList, [], -1, False)
    # Pick out the intended satellites
    cSat = []
    for i in range(len(cSat1)):
        if cSat0[0] != 'ALL' and cSat1[i] not in cSat0:
            continue
        cSat.append(cSat1[i])
    nSat = len(cSat)

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # Read in the geometric angles if required
    if iAng != 0:
        Ang = []
        for j in range(nSat):
            # Epoch, beta, delta u
            Ang.append([])
            Ang.append([])
            Ang.append([])
        MJD1 = int(np.floor(np.amin(rEpo)))
        MJD2 = int(np.ceil(np.amax(rEpo)))
        for i in range(MJD1, MJD2+1):
            YYYY, DOY = GNSSTime.mjd2doy(i)
            for j in range(nSat):
                fAng = fGeoAngPath+'/GeoAng_' + \
                    '{:04d}{:03d}'.format(YYYY, DOY)+'_'+cSat[j]
                if not os.path.isfile(fAng):
                    print(fAng+' does not exist!')
                    continue
                with open(fAng, mode='rt') as fOb:
                    for cLine in fOb:
                        if cLine[0:3] != 'ANG':
                            continue
                        if len(cLine) < 5:
                            continue
                        cWords = cLine[40:].split()
                        #Epoch (rMJD)
                        Ang[3*j].append(int(cWords[0])+float(cWords[1])/86400)
                        #Beta in deg
                        Ang[3*j+1].append(float(cWords[3]))
                        # delta u, orbit angle from Noon, in deg
                        Ang[3*j+2].append(float(cWords[9]))
    if iAng == 2:
        # two-column axies
        fig, axs = plt.subplots(nSat, 2, sharex='col',
                                squeeze=False, figsize=(12, nSat*2))
    else:
        fig, axs = plt.subplots(nSat, 1, sharex='col',
                                squeeze=False, figsize=(8, nSat*2))
        # fig.subplots_adjust(hspace=0.1)

    strTmp = '{: <3s} {: >7s} {: >7s} {: >7s} {: >7s}'.format(
        'PRN', 'Along', 'Cross', 'Radial', '1D')
    print(strTmp)
    for i in range(nSat):
        iSat = cSat1.index(cSat[i])
        RMS = np.zeros(4)
        RMS[:] = np.nan
        for j in range(3):
            nValid = 0
            xtmp = 0
            for k in range(nEpo):
                if np.isnan(Dif[k, 3*iSat+j]):
                    continue
                nValid = nValid+1
                xtmp = xtmp + Dif[k, 3*iSat+j]*Dif[k, 3*iSat+j]
            if nValid > 0:
                # mm -> cm
                RMS[j] = np.sqrt(xtmp/nValid)/10
            else:
                RMS[j] = np.nan
        if np.isnan(RMS[0]):
            RMS[3] = np.nan
            continue
        if np.isnan(RMS[1]):
            RMS[3] = np.nan
            continue
        if np.isnan(RMS[2]):
            RMS[3] = np.nan
            continue
        RMS[3] = np.sqrt((RMS[0]*RMS[0]+RMS[1]*RMS[1]+RMS[2]*RMS[2])/3.0)
        # mm -> cm
        axs[i, 0].plot(rEpo, Dif[:, 3*iSat]/10, '.r', ms=1, label='Along')
        axs[i, 0].plot(rEpo, Dif[:, 3*iSat+1]/10, '.g', ms=1, label='Cross')
        axs[i, 0].plot(rEpo, Dif[:, 3*iSat+2]/10, '.b', ms=1, label='Radial')
        axs[i, 0].axhline(color='darkgray', linestyle='dashed', alpha=0.5)
        axs[i, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
        axs[i, 0].set_axisbelow(True)
        # #Set th y limit for different satellites
        # if cSat[i] in ['C01','C02','C03','C04','C05','C38','C39','C40','C59']:
        #     axs[i,0].set_ylim(bottom=-100,top=100)
        # else:
        #     axs[i,0].set_ylim(bottom=-80,top=80)
        if yMax > 0:
            axs[i, 0].set_ylim(bottom=-yMax, top=yMax)

        if iAng == 1:
            # Plot beta angles at the right y-axes
            axe = axs[i, 0].twinx()
            axe.set_ylim(bottom=-90, top=90)
            axe.set_ylabel(
                r'$\beta$ [deg]', fontname='Arial', fontsize=16, color='goldenrod')
            for tl in axe.get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
                tl.set_color('goldenrod')
            axe.plot(Ang[3*i], Ang[3*i+1], '.-',
                     c='goldenrod', ms=1, label=r'$\beta$')
            # Label out the period between +/- 4 deg
            axe.fill_between(Ang[3*i], 0, 1, where=np.abs(Ang[3*i+1]) <= 4, alpha=0.5,
                             color='silver', transform=axe.get_xaxis_transform())
        elif iAng == 2:
            # plot for the beta-delta u for radial component
            # Firstly, match the angles with the values. Remenber that the Ang is along time
            # which makes the matching a little bit time-saving.
            a = [[], [], [], [], []]
            for j in range(nEpo):
                for k in range(len(Ang[3*i])):
                    if (Ang[3*i][k]-rEpo[j])*86400 < -1.0:
                        # Not get yet
                        continue
                    elif (Ang[3*i][k]-rEpo[j])*86400 > 1.0:
                        # Over pass
                        break
                    else:
                        # delta u
                        a[0].append(Ang[3*i+2][k])
                        # beta
                        a[1].append(Ang[3*i+1][k])
                        # along-track, mm -> cm
                        a[2].append(Dif[j, 3*iSat]/10)
                        # cross-track, mm -> cm
                        a[3].append(Dif[j, 3*iSat+1]/10)
                        # radial, mm -> cm
                        a[4].append(Dif[j, 3*iSat+2]/10)
            qm = axs[i, 1].scatter(
                a[0], a[1], s=5, c=a[4], marker='.', cmap='plasma')
            # if yMax > 0:
            #     qm.set_clim(vmin=-yMax,vmax=yMax)
            axs[i, 1].set_xlim(left=-180, right=180)
            cbar = fig.colorbar(qm, ax=axs[i, 1])
            cbar.set_label('Radial [cm]', loc='center',
                           fontname='Arial', fontsize=16)
            for tl in cbar.ax.get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            axs[i, 1].set_ylabel(
                r'$\beta$ [deg]', fontname='Arial', fontsize=16)
            for tl in axs[i, 1].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)

        axs[i, 0].text(0.02, 0.98, cSat[i], transform=axs[i, 0].transAxes, ha='left', va='top',
                       family='Arial', size=16, weight='bold')
        # strTmp='RMS: {:7.2f} {:7.2f} {:7.2f} {:7.2f}'.format(RMS[0],RMS[1],RMS[2],RMS[3])
        # axs[i,0].text(0.99,0.01,strTmp,transform=axs[i,0].transAxes,ha='right',va='bottom',
        #               family='Arial',size=16)
        # Report to the terminal
        strTmp = '{: <3s} {: >7.2f} {: >7.2f} {: >7.2f} {: >7.2f}'.format(cSat[i], RMS[0],
                                                                          RMS[1], RMS[2], RMS[3])
        print(strTmp)
        axs[i, 0].yaxis.set_major_formatter('{x:2.0f}')
        axs[i, 0].set_ylabel('[cm]', fontname='Arial', fontsize=16)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        axs[i, 0].legend(ncol=3, loc='upper center', framealpha=0.6, bbox_to_anchor=(0.5, 1.0),
                         prop={'family': 'Arial', 'size': 14}, markerscale=5, numpoints=3,
                         borderaxespad=0.1, columnspacing=1.0, handlelength=1.0, handletextpad=0.4)
    axs[i, 0].set_xlabel(r'Modified Julian Day', fontname='Arial', fontsize=16)
    axs[i, 0].ticklabel_format(axis='x', useOffset=False, useMathText=True)
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    if iAng == 2:
        axs[i, 1].set_xlabel(r'$\Delta\mu$ [deg]',
                             fontname='Arial', fontsize=16)
        for tl in axs[i, 1].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotDifSer1(fList, cSat, nPer, tPer, OutFilePrefix, OutFileSuffix):
    '''
    Plot time series of orb diff for one satellite during different eclipsing seasons

        nPer --- number of time periods
        tPer --- start && end MJD of each time period, each period contains
                 one and only one eclipsing season.

    NOTE: This is for presenting orbit performance during eclipsing seasons.
          So, orbit geometry angle information files are always needed.
    '''

    nEpo, cSat1, rEpo, Dif = GetOrbdif(fList, [], -1, False)
    if cSat not in cSat1:
        sys.exit(cSat + ' not found!')
    else:
        iSat = cSat1.index(cSat)

    GeoAngPath = r'Z:/GNSS/PROJECT/OrbGeometry/ANG'
    X = [[], []]
    # Read beta angle info
    MJD1 = int(np.floor(np.amin(rEpo)))
    MJD2 = int(np.ceil(np.amax(rEpo)))
    for i in range(MJD1, MJD2+1):
        YYYY, DOY = GNSSTime.mjd2doy(i)
        fAng = GeoAngPath+'/GeoAng_'+'{:04d}{:03d}'.format(YYYY, DOY)+'_'+cSat
        if not os.path.isfile(fAng):
            print(fAng+' does not exist!')
            continue
        with open(fAng, mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:1] != ' ':
                    continue
                if len(cLine) < 5:
                    continue
                #Epoch (MJD)
                X[0].append(float(cLine[37:54]))
                #Beta in deg
                X[1].append(float(cLine[56:73]))
    Ang = np.array(X)
    ind = np.argsort(Ang[0])

    fig, axs = plt.subplots(nPer, 1, squeeze=False, figsize=(7, 4*nPer))
    # fig.subplots_adjust(hspace=0.1)

    for i in range(nPer):
        # Find out the eclipsing period, only one within each period
        rEclips = []
        lEnter1 = False
        for j in range(Ang[0].size):
            if Ang[0][ind[j]] < tPer[i][0]:
                continue
            elif Ang[0][ind[j]] > tPer[i][1]:
                break
            if np.abs(Ang[1][ind[j]]) >= 4:
                lEnter2 = False
            else:
                lEnter2 = True
            if (not lEnter1 and lEnter2) or (lEnter1 and not lEnter2):
                # Go into or out the deep eclipsing period
                print(
                    cSat+' {:>10.4f} {:>6.2f}'.format(Ang[0][ind[j]], Ang[1][ind[j]]))
                rEclips.append(Ang[0][ind[j]])
            lEnter1 = lEnter2
        # Calculate the RMS in/out eclipsing period
        RMS = np.zeros((2, 4))
        nPt = np.zeros(2, dtype=np.int32)
        for j in range(rEpo.size):
            if rEpo[j] < tPer[i][0]:
                continue
            elif rEpo[j] > tPer[i][1]:
                continue
            elif rEpo[j] >= rEclips[0] and rEpo[j] <= rEclips[1]:
                # Eclipsing period
                if np.isnan(Dif[j, 3*iSat]) or \
                   np.isnan(Dif[j, 3*iSat+1]) or \
                   np.isnan(Dif[j, 3*iSat+2]):
                    continue
                nPt[0] = nPt[0]+1
                for k in range(3):
                    RMS[0, k] = RMS[0, k]+Dif[j, 3*iSat+k]*Dif[j, 3*iSat+k]
            else:
                # Non-eclipsing period
                if np.isnan(Dif[j, 3*iSat]) or \
                   np.isnan(Dif[j, 3*iSat+1]) or \
                   np.isnan(Dif[j, 3*iSat+2]):
                    continue
                nPt[1] = nPt[1]+1
                for k in range(3):
                    RMS[1, k] = RMS[1, k]+Dif[j, 3*iSat+k]*Dif[j, 3*iSat+k]
        if nPt[0] != 0:
            for j in range(3):
                # mm -> cm
                RMS[0, j] = np.sqrt(RMS[0, j]/nPt[0])/10
            RMS[0, 3] = np.sqrt(
                (RMS[0, 0]*RMS[0, 0]+RMS[0, 1]*RMS[0, 1]+RMS[0, 2]*RMS[0, 2])/3.0)
        else:
            print('No point within eclipsing period {:2d}'.format(i))

        if nPt[1] != 0:
            for j in range(3):
                # mm -> cm
                RMS[1, j] = np.sqrt(RMS[1, j]/nPt[1])/10
            RMS[1, 3] = np.sqrt(
                (RMS[1, 0]*RMS[1, 0]+RMS[1, 1]*RMS[1, 1]+RMS[1, 2]*RMS[1, 2])/3.0)
        else:
            print('No point within non-eclipsing period {:2d}'.format(i))

        if cSat in ['C01', 'C02', 'C03', 'C04', 'C05', 'C38', 'C39', 'C40', 'C59']:
            axs[i, 0].set_ylim(bottom=-100, top=100)
            axs[i, 0].set_yticks(
                [-80, -50, -20, -10, 10, 20, 50, 80], minor=False)
        else:
            axs[i, 0].set_ylim(bottom=-80, top=80)
            axs[i, 0].set_yticks(
                [-60, -30, -20, -10, 10, 20, 30, 60], minor=False)
        axs[i, 0].text(0.02, 0.98, cSat, transform=axs[i, 0].transAxes, ha='left', va='top',
                       family='Arial', size=16)
        axs[i, 0].set_ylabel('[cm]', fontname='Arial', fontsize=16)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        axs[i, 0].set_xlim(tPer[i][0], tPer[i][1])
        # mm -> cm
        axs[i, 0].plot(rEpo, Dif[:, 3*iSat]/10, '.r', label='Along')
        axs[i, 0].plot(rEpo, Dif[:, 3*iSat+1]/10, '.g', label='Cross')
        axs[i, 0].plot(rEpo, Dif[:, 3*iSat+2]/10, '.b', label='Radial')
        axs[i, 0].axhline(color='darkgray', linestyle='dashed', alpha=0.5)
        axs[i, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)

        strTmp = 'Ecl: {:7.2f} {:7.2f} {:7.2f} {:7.2f}'.format(
            RMS[0, 0], RMS[0, 1], RMS[0, 2], RMS[0, 3])
        axs[i, 0].text(0.98, 0.02, strTmp, transform=axs[i, 0].transAxes, ha='right', va='bottom',
                       family='monospace', size=14, weight='bold', color='silver')
        strTmp = 'Nom: {:7.2f} {:7.2f} {:7.2f} {:7.2f}'.format(
            RMS[1, 0], RMS[1, 1], RMS[1, 2], RMS[1, 3])
        axs[i, 0].text(0.98, 0.12, strTmp, transform=axs[i, 0].transAxes, ha='right', va='bottom',
                       family='monospace', size=14, weight='bold', color='black')

        # Beta angle
        axe = axs[i, 0].twinx()
        axe.set_ylim(bottom=-20, top=20)
        axe.set_ylabel(r'$\beta$ [deg]',
                       fontname='Arial', fontsize=16, color='gold')
        for tl in axe.get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
            tl.set_color('gold')
        axe.plot(Ang[0][ind], Ang[1][ind], '.-',
                 ms=2, color='gold', label=r'$\beta$')
        # +/- 4 deg
        axe.fill_between(Ang[0][ind], 0, 1, where=np.abs(Ang[1][ind]) <= 4, alpha=0.5,
                         color='silver', transform=axe.get_xaxis_transform())

        axs[i, 0].legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.0), framealpha=0.6,
                         labelspacing=0.2, borderpad=0.1, handletextpad=0.2, handlelength=0.2,
                         columnspacing=0.5, prop={'family': 'Arial', 'size': 14})
        for tl in axs[i, 0].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
    axs[i, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotDifHel0(fLogList, iPlot, yUnt, yLab, OutFilePrefix, OutFileSuffix):
    '''
    Plot the Helmert parameters estimated during orb diff via `orbfit`

    iPlot --- Which component to be plotted
              # 0, only Scale
              # 1, only Translation
              # 2, only Rotation
              # 3, Scale, Translation and Rotation
     yUnt --- unit of y-axes for each component
     yLab ---
    '''
    if len(yUnt) == 0:
        yUnt = ['ppb', 'mm', 'mas']
        yLab = ['Scale [ppb]', 'Translation [mm]', 'Rotation [mas]']
    elif len(yUnt) != len(yLab):
        sys.exit('Inconsisten input for yUnt and yLab')

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    cPar = []
    xPar = []
    for i in range(len(fLogList)):
        with open(fLogList[i], mode='rt') as fOb:
            # with open(fLogList[i],mode='r',encoding='UTF-16') as fOb:
            lBeg = False
            for cLine in fOb:
                if cLine[0:18] == '+SOLUTION/ESTIMATE':
                    lBeg = True
                elif cLine[0:18] == '-SOLUTION/ESTIMATE':
                    lBeg = False
                elif lBeg:
                    cWords = cLine.split()
                    if iPlot == 0:
                        if cWords[1][0:1] != 'D':
                            continue
                    elif iPlot == 1:
                        if cWords[1][0:1] != 'T':
                            continue
                    elif iPlot == 2:
                        if cWords[1][0:1] != 'R':
                            continue
                    if cWords[1] not in cPar:
                        cPar.append(cWords[1])
                        xPar.append([])
                        xPar.append([])
                    iPar = cPar.index(cWords[1])
                    # Epoch
                    xPar[iPar*2].append(float(cWords[6]))
                    # Estimates
                    if cWords[1][0:1] == 'D' and yUnt[0] == 'ppb':
                        # 0.1 ppb -> ppb
                        xPar[iPar*2+1].append(float(cWords[4])/10)
                    elif cWords[1][0:1] == 'T' and yUnt[1] == 'm':
                        # mm -> m
                        xPar[iPar*2+1].append(float(cWords[4])/1000)
                    elif cWords[1][0:1] == 'R' and yUnt[2] == 'as':
                        # mas -> as
                        xPar[iPar*2+1].append(float(cWords[4])/1000)
                    else:
                        xPar[iPar*2+1].append(float(cWords[4]))
    nPar = len(cPar)

    if iPlot == 3:
        fig, axs = plt.subplots(3, 1, squeeze=False,
                                sharex='col', figsize=(10, 9))
        # fig.subplots_adjust(hspace=0.1)

        print('{: <15s} {: >12s} {: >12s} {: >12s}'.format(
            'ParName', 'Mean', 'STD', 'RMS'))
        for i in range(nPar):
            Mea = np.mean(xPar[2*i+1])
            Sig = np.std(xPar[2*i+1])
            RMS = 0
            mPar = len(xPar[2*i+1])
            for j in range(mPar):
                RMS = RMS + xPar[2*i+1][j]*xPar[2*i+1][j]
            RMS = np.sqrt(RMS/mPar)
            print('{: <15s} {: >12.4f} {: >12.4f} {: >12.4f}'.format(
                cPar[i], Mea, Sig, RMS))
            if cPar[i] == 'D0' or cPar[i] == 'D':
                axs[0, 0].plot(xPar[2*i], xPar[2*i+1], '.--r', ms=4)
            elif cPar[i] == 'T10' or cPar[i] == 'T1':
                axs[1, 0].plot(xPar[2*i], xPar[2*i+1],
                               '.--r', ms=4, lw=1, label='T1')
            elif cPar[i] == 'T20' or cPar[i] == 'T2':
                axs[1, 0].plot(xPar[2*i], xPar[2*i+1],
                               'o--g', ms=4, lw=1, label='T2')
            elif cPar[i] == 'T30' or cPar[i] == 'T3':
                axs[1, 0].plot(xPar[2*i], xPar[2*i+1],
                               '^--b', ms=4, lw=1, label='T3')
            elif cPar[i] == 'R10' or cPar[i] == 'R1':
                axs[2, 0].plot(xPar[2*i], xPar[2*i+1],
                               '.--r', ms=4, lw=1, label='R1')
            elif cPar[i] == 'R20' or cPar[i] == 'R2':
                axs[2, 0].plot(xPar[2*i], xPar[2*i+1],
                               'o--g', ms=4, lw=1, label='R2')
            elif cPar[i] == 'R30' or cPar[i] == 'R3':
                axs[2, 0].plot(xPar[2*i], xPar[2*i+1],
                               '^--b', ms=4, lw=1, label='R3')
        axs[0, 0].grid(which='both', axis='y', color='darkgray', linestyle='--',
                       linewidth=0.8)
        axs[1, 0].grid(which='both', axis='y', color='darkgray', linestyle='--',
                       linewidth=0.8)
        axs[2, 0].grid(which='both', axis='y', color='darkgray', linestyle='--',
                       linewidth=0.8)
        axs[2, 0].set_xlabel('Modified Julian Day',
                             fontname='Arial', fontsize=16)

        for tl in axs[2, 0].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        for i in range(3):
            # axs[i,0].ticklabel_format(axis='y',style='sci',useOffset=False,useMathText=True)
            axs[i, 0].yaxis.set_major_formatter('{x: >8.1f}')
            axs[i, 0].set_ylabel(yLab[i], fontname='Arial', fontsize=16)
            for tl in axs[i, 0].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            if i != 0:
                axs[i, 0].legend(ncol=3, loc='upper right', bbox_to_anchor=(1.0, 1.0),
                                 framealpha=0.0, prop={'family': 'Arial', 'size': 14})
    else:
        # Only one specific component
        fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(10, 3))

        print('{: <15s} {: >12s} {: >12s} {: >12s}'.format(
            'ParName', 'Mean', 'STD', 'RMS'))
        for i in range(nPar):
            Mea = np.mean(xPar[2*i+1])
            Sig = np.std(xPar[2*i+1])
            RMS = 0
            mPar = len(xPar[2*i+1])
            for j in range(mPar):
                RMS = RMS + xPar[2*i+1][j]*xPar[2*i+1][j]
            RMS = np.sqrt(RMS/mPar)
            print('{: <15s} {: >12.4f} {: >12.4f} {: >12.4f}'.format(
                cPar[i], Mea, Sig, RMS))
            if cPar[i] == 'D0' or cPar[i] == 'D':
                axs[0, 0].plot(xPar[2*i], xPar[2*i+1], '.--r', ms=4)
            elif cPar[i] == 'T10' or cPar[i] == 'T1':
                axs[0, 0].plot(xPar[2*i], xPar[2*i+1],
                               '.--r', ms=4, lw=1, label='T1')
            elif cPar[i] == 'T20' or cPar[i] == 'T2':
                axs[0, 0].plot(xPar[2*i], xPar[2*i+1],
                               'o--g', ms=4, lw=1, label='T2')
            elif cPar[i] == 'T30' or cPar[i] == 'T3':
                axs[0, 0].plot(xPar[2*i], xPar[2*i+1],
                               '^--b', ms=4, lw=1, label='T3')
            elif cPar[i] == 'R10' or cPar[i] == 'R1':
                axs[0, 0].plot(xPar[2*i], xPar[2*i+1],
                               '.--r', ms=4, lw=1, label='R1')
            elif cPar[i] == 'R20' or cPar[i] == 'R2':
                axs[0, 0].plot(xPar[2*i], xPar[2*i+1],
                               'o--g', ms=4, lw=1, label='R2')
            elif cPar[i] == 'R30' or cPar[i] == 'R3':
                axs[0, 0].plot(xPar[2*i], xPar[2*i+1],
                               '^--b', ms=4, lw=1, label='R3')
        if iPlot != 0:
            axs[0, 0].legend(ncol=3, loc='upper right', bbox_to_anchor=(1.0, 1.0),
                             framealpha=0.0, prop={'family': 'Arial', 'size': 14})
        axs[0, 0].grid(which='both', axis='y', color='darkgray', linestyle='--',
                       linewidth=0.8)
        axs[0, 0].set_xlabel('Modified Julian Day',
                             fontname='Arial', fontsize=16)
        for tl in axs[0, 0].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        axs[0, 0].set_ylabel(yLab[iPlot], fontname='Arial', fontsize=16)
        # axs[0,0].ticklabel_format(axis='y',style='sci',useOffset=False,useMathText=True)
        axs[0, 0].yaxis.set_major_formatter('{x: >8.1f}')
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


def PlotDifHel1(cSer, fSerList, cParHel, nCol, yMax, iUnit, OutFilePrefix, OutFileSuffix):
    '''
    Plot one specified Helmert parameter from several solutions.
    The input files contain estimates obtained during orb diff via `orbfit`

    cParHel --- the specified Helmert parameter, i.e.
                D0, T0, R0
       yMax --- the max limit of the y-axis
      iUnit --- the unit to be used
                # 1, use arcsec instead of milliarcsec for R0
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    if cParHel == 'D0':
        yLab = 'Scale [ppb]'
    elif cParHel == 'T0':
        yLab = 'Translation [mm]'
    elif cParHel == 'R0':
        if iUnit == 1:
            # arcsec
            yLab = 'Rotation [as]'
        else:
            # milliarcsec
            yLab = 'Rotation [mas]'
    else:
        sys.exit('Unknown Helmert parameter, '+cParHel)

    nSer = len(cSer)
    cPar = []
    xPar = []
    for iSer in range(nSer):
        for i in range(len(fSerList[iSer])):
            with open(fSerList[iSer][i], mode='rt') as fOb:
                # with open(fSerList[iSer][i],mode='r',encoding='UTF-16') as fOb:
                lBeg = False
                for cLine in fOb:
                    if cLine[0:18] == '+SOLUTION/ESTIMATE':
                        lBeg = True
                    elif cLine[0:18] == '-SOLUTION/ESTIMATE':
                        lBeg = False
                    elif lBeg:
                        cWords = cLine.split()
                        if cParHel == 'D0':
                            if cWords[1] != 'D0':
                                continue
                            else:
                                cPar0 = 'D'
                        elif cParHel == 'T0':
                            if cWords[1] == 'T10':
                                cPar0 = 'T1'
                            elif cWords[1] == 'T20':
                                cPar0 = 'T2'
                            elif cWords[1] == 'T30':
                                cPar0 = 'T3'
                            else:
                                continue
                        elif cParHel == 'R0':
                            if cWords[1] == 'R10':
                                cPar0 = 'R1'
                            elif cWords[1] == 'R20':
                                cPar0 = 'R2'
                            elif cWords[1] == 'R30':
                                cPar0 = 'R3'
                            else:
                                continue
                        if cPar0 not in cPar:
                            cPar.append(cPar0)
                            for j in range(nSer):
                                xPar.append([])
                                xPar.append([])
                        iPar = cPar.index(cPar0)
                        j = iPar*nSer*2 + iSer*2
                        # Epoch
                        xPar[j].append(float(cWords[6]))
                        # Estimates
                        if cPar0[0:1] == 'D':
                            # 0.1 ppb -> ppb
                            xPar[j+1].append(float(cWords[4])/10)
                        elif cPar0[0:1] == 'R' and iUnit == 1:
                            # mas -> as
                            xPar[j+1].append(float(cWords[4])/1000)
                        else:
                            xPar[j+1].append(float(cWords[4]))
    nPar = len(cPar)

    # Cal the number of row based on specified number of col
    nRow = math.ceil(nSer/nCol)

    fig, axs = plt.subplots(nRow, nCol, squeeze=False,
                            sharex='col', sharey='row', figsize=(6*nCol, 3*nRow))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.05)

    for iSer in range(nSer):
        iRow = math.ceil((iSer+1)/nCol)-1
        iCol = iSer-iRow*nCol
        for i in range(nPar):
            j = i*nSer*2 + iSer*2
            if cPar[i] == 'D':
                axs[iRow, iCol].plot(xPar[j], xPar[j+1], '.--r', ms=4, lw=1)
            elif cPar[i] == 'T1':
                axs[iRow, iCol].plot(xPar[j], xPar[j+1],
                                     '.--r', ms=4, lw=1, label='T1')
            elif cPar[i] == 'T2':
                axs[iRow, iCol].plot(xPar[j], xPar[j+1],
                                     'o--g', ms=4, lw=1, label='T2')
            elif cPar[i] == 'T3':
                axs[iRow, iCol].plot(xPar[j], xPar[j+1],
                                     '^--b', ms=4, lw=1, label='T3')
            elif cPar[i] == 'R1':
                axs[iRow, iCol].plot(xPar[j], xPar[j+1],
                                     '.--r', ms=4, lw=1, label='R1')
            elif cPar[i] == 'R2':
                axs[iRow, iCol].plot(xPar[j], xPar[j+1],
                                     'o--g', ms=4, lw=1, label='R2')
            elif cPar[i] == 'R3':
                axs[iRow, iCol].plot(xPar[j], xPar[j+1],
                                     '^--b', ms=4, lw=1, label='R3')
        axs[iRow, iCol].text(0.02, 0.98, cSer[iSer], transform=axs[iRow, iCol].transAxes,
                             ha='left', va='top',
                             fontdict={'fontsize': 14, 'fontname': 'Arial', 'fontweight': 'bold'})
        axs[iRow, iCol].grid(which='both', axis='y',
                             c='darkgray', ls='--', lw=0.4)
        axs[iRow, iCol].set_axisbelow(True)
        if cPar[0][0:1] == 'T' or cPar[0][0:1] == 'R':
            axs[iRow, iCol].legend(ncol=3, loc='upper right', bbox_to_anchor=(1.0, 1.0),
                                   framealpha=0.0, prop={'family': 'Arial', 'size': 14})
        # axs[iRow,iCol].ticklabel_format(axis='y',style='sci',useOffset=False,useMathText=True)
        if cParHel == 'R0' and iUnit == 1:
            axs[iRow, iCol].yaxis.set_major_formatter('{x: >3.0f}')
        else:
            axs[iRow, iCol].yaxis.set_major_formatter('{x: >8.1f}')

        axs[iRow, iCol].set_ylim(bottom=-yMax, top=yMax)
        if iCol == 0:
            axs[iRow, iCol].set_ylabel(yLab, fontname='Arial', fontsize=16)
            for tl in axs[iRow, iCol].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
        if iRow == (nRow-1):
            # Only for the last subplot
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
    fListSer = []

    # InFilePrefix='Y:/PRO_2019001_2020366_WORK/G7/WORK2019???/'
    InFilePrefix = os.path.join(cDskPre0, r'GNSS/PROJECT/OrbCompare/')
    # InFilePrefix = os.path.join(cDskPre0, r'GNSS/PROJECT/OrbOverlap/')

    OutFilePrefix = os.path.join(cDskPre0, r'PRO_2019001_2020366/I010/ORB/')
    # OutFilePrefix=r'D:/Code/PROJECT/WORK2019335_ERROR/'

    fDifList = glob.glob(
        InFilePrefix+'PRO_2019001_2020366/I010_vs_C0/dif_20193??_TRD')

    OutFileSuffix = 'OrbCmp_2019335_2019365_C0_TRD'
    PlotDifRMS0(fDifList, 25, OutFilePrefix, OutFileSuffix)
    # PlotDifRMS1(fDifList, 25, 'Radial', [
    #             0, 1, 3], 800, 'Non-Radial', OutFilePrefix, OutFileSuffix)
    # fGeoAngPath=os.path.join(cDskPre0,r'GNSS/PROJECT/OrbGeometry/ANG')
    # OutFileSuffix='DifSer0_I1'
    # PlotDifSer0(fDifList,2,fGeoAngPath,['C22','C23','C26','C30'],5,OutFilePrefix,OutFileSuffix)

    fLogList = glob.glob(InFilePrefix+'log_I010_vs_C0_TRD')
    # fLogList=glob.glob(InFilePrefix+'log_I010_TRD')
    OutFileSuffix = 'CmpHel_2019335_2019365_I010_vs_C0_TRD_R'
    # PlotDifHel0(fLogList, 2, ['', '', 'as'], ['Scale [ppb]', 'Translation [mm]', 'Rotation [as]'],
    #             OutFilePrefix, OutFileSuffix)
    # cSer.append('Grd vs ISL(None)')
    # fListSer.append(glob.glob(os.path.join(cDskPre0,r'GNSS/PROJECT/OrbCompare/log_C0_vs_I010_TRD')))
    # cSer.append('Grd vs ISL(Model 1)')
    # fListSer.append(glob.glob(os.path.join(cDskPre0,r'GNSS/PROJECT/OrbCompare/log_C0_vs_I021_TRD')))
    # cSer.append('Grd vs ISL(Model 2)')
    # fListSer.append(glob.glob(os.path.join(cDskPre0,r'GNSS/PROJECT/OrbCompare/log_C0_vs_I022_TRD')))
    # cSer.append('Grd vs ISL(Model 3)')
    # fListSer.append(glob.glob(os.path.join(cDskPre0,r'GNSS/PROJECT/OrbCompare/log_C0_vs_I023_TRD')))
    # cSer.append('ISL(None) vs ISL(Model 1)')
    # fListSer.append(glob.glob(os.path.join(cDskPre0,r'GNSS/PROJECT/OrbCompare/log_I010_vs_I021_TRD')))
    # cSer.append('ISL(None) vs ISL(Model 2)')
    # fListSer.append(glob.glob(os.path.join(cDskPre0,r'GNSS/PROJECT/OrbCompare/log_I010_vs_I022_TRD')))
    # cSer.append('ISL(None) vs ISL(Model 3)')
    # fListSer.append(glob.glob(os.path.join(cDskPre0,r'GNSS/PROJECT/OrbCompare/log_I010_vs_I023_TRD')))
    # cSer.append('ISL(Model 1) vs ISL(Model 2)')
    # fListSer.append(glob.glob(os.path.join(cDskPre0,r'GNSS/PROJECT/OrbCompare/log_I021_vs_I022_TRD')))
    # cSer.append('ISL(Model 1) vs ISL(Model 3)')
    # fListSer.append(glob.glob(os.path.join(cDskPre0,r'GNSS/PROJECT/OrbCompare/log_I021_vs_I023_TRD')))
    # cSer.append('ISL(Model 2) vs ISL(Model 3)')
    # fListSer.append(glob.glob(os.path.join(cDskPre0,r'GNSS/PROJECT/OrbCompare/log_I022_vs_I023_TRD')))
    # OutFileSuffix = 'CmpHel_2019335_2019365_ALL_TRD_R'
    # PlotDifHel1(cSer, fListSer, 'R0', 2, 30, 1, OutFilePrefix, OutFileSuffix)

    # cSer.append('None')
    # fDifList = glob.glob(InFilePrefix+'PRO_2019001_2020366/J600/dif_20193??')
    # fListSer.append(fDifList)

    # cSer.append('Model 1')
    # fDifList = glob.glob(InFilePrefix+'PRO_2019001_2020366/J601/dif_20193??')
    # fListSer.append(fDifList)

    # cSer.append('Model 2')
    # fDifList = glob.glob(InFilePrefix+'PRO_2019001_2020366/J602/dif_20193??')
    # fListSer.append(fDifList)

    # cSer.append('Model 3')
    # fDifList = glob.glob(InFilePrefix+'PRO_2019001_2020366/J603/dif_20193??')
    # fListSer.append(fDifList)

    # cSer.append('1 cm')
    # fDifList = glob.glob(InFilePrefix+'PRO_2019001_2020366/J640/dif_20193??')
    # fListSer.append(fDifList)

    # cSer.append('2 cm')
    # fDifList = glob.glob(InFilePrefix+'PRO_2019001_2020366/J641/dif_20193??')
    # fListSer.append(fDifList)

    # cSer.append('4 cm')
    # fDifList = glob.glob(InFilePrefix+'PRO_2019001_2020366/J642/dif_20193??')
    # fListSer.append(fDifList)

    # cSer.append('8 cm')
    # fDifList = glob.glob(InFilePrefix+'PRO_2019001_2020366/J646/dif_20193??')
    # fListSer.append(fDifList)

    # cSer.append('12 cm')
    # fDifList = glob.glob(InFilePrefix+'PRO_2019001_2020366/J647/dif_20193??')
    # fListSer.append(fDifList)

    # cSer.append('15 cm')
    # fDifList = glob.glob(InFilePrefix+'PRO_2019001_2020366/J648/dif_20193??')
    # fListSer.append(fDifList)

    # cSer.append('18 cm')
    # fDifList = glob.glob(InFilePrefix+'PRO_2019001_2020366/J651/dif_20193??')
    # fListSer.append(fDifList)

    # cSer.append('20 cm')
    # fDifList = glob.glob(InFilePrefix+'PRO_2019001_2020366/J653/dif_20193??')
    # fListSer.append(fDifList)

    cSer.append('No ISL')
    fDifList = glob.glob(InFilePrefix+'PRO_2019001_2020366/C01/dif_20193??')
    fListSer.append(fDifList)

    cSer.append('ISL Rng')
    fDifList = glob.glob(InFilePrefix+'PRO_2019001_2020366/I648/dif_20193??')
    fListSer.append(fDifList)

    cSer.append('ISL Clk')
    fDifList = glob.glob(InFilePrefix+'PRO_2019001_2020366/J642/dif_20193??')
    fListSer.append(fDifList)

    cSer.append('ISL Rng+Clk')
    fDifList = glob.glob(InFilePrefix+'PRO_2019001_2020366/D650/dif_20193??')
    fListSer.append(fDifList)

    # # BDS-2
    # cSat = ['C01', 'C02', 'C03', 'C04', 'C05', 'C06', 'C07', 'C08', 'C09', 'C10',
    #         'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17']
    # BDS-3
    # cSat = ['C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28',
    #         'C29', 'C30', 'C31', 'C32', 'C33', 'C34', 'C35', 'C36', 'C37', 'C38', 'C39']
    # cSat=['ALL']
    cSat = ['EXL', 'C38', 'C39']
    OutFileSuffix = 'DBDRMS0_2019335_2019365_Comp_1D'
    # PlotDifRMS2(cSer, fListSer, -200, True, 3, cSat, [
    #             50, 15, 15, 10], OutFilePrefix, OutFileSuffix)
    # PlotDifRMS3(cSer, fListSer, [1], -200, True, 4, cSat, [
    #             85, 12, 12, 50], [15, 15, 10, 25], 'C0', 'C1', OutFilePrefix, OutFileSuffix)

    # nPer=4
    # #BDS MEO-A
    # tPer=[[58585,58632],[58763,58810],[58933,58981],[59110,59158]]
    # #BDS MEO-B
    # tPer=[[58520,58560],[58700,58740],[58875,58915],[59055,59095]]
    # BDS MEO-C
    # tPer=[[58618,58660],[58798,58839],[58973,59014],[59153,59194]]
    # OutFileSuffix='DBDSer_C24.png'
    # PlotDifSer1(fDifList,'C24',nPer,tPer,OutFilePrefix,OutFileSuffix)
