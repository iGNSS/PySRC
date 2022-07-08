#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Make some plots based on the output during oi
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
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# Local application/library specific imports
from PySRC.miscellaneous import GNSSTime


def PlotVarEqu(fOrb, iPar1, iPar2, cSat0, OutFilePrefix, OutFileSuffix):
    '''
    Plot the solution of variational equations from iPar1 to iPar2
    based on the ASCII orb-file output during oi

    NOTE: Here we assume that all satellites share the same parameters set
    '''
    if iPar1 > 0 and iPar2 >= iPar1:
        nSol = iPar2-iPar1+1
    else:
        # plot for all parameters
        iPar1 = 1
        with open(fOrb, mode='rt') as fOb:
            for cLine in fOb:
                cWords = cLine.split()
                nSol = (len(cWords)-3-3)//3
                break
    # Always plot the positions as well
    nSol = nSol+1
    cSat = []
    xSol = []
    with open(fOrb, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cSat0[0] != 'ALL' and cWords[2] not in cSat0:
                continue
            if cWords[2] not in cSat:
                cSat.append(cWords[2])
                for i in range(3*nSol+1):
                    xSol.append([])
            iSat = cSat.index(cWords[2])
            xSol[iSat*(3*nSol+1)].append(int(cWords[0])+float(cWords[1])/86400)
            for j in range(nSol):
                if j == 0:
                    # Satellite positions
                    k = 2
                else:
                    k = 2 + 3 + 3*(iPar1-1) + (j-1)*3
                xSol[iSat*(3*nSol+1)+3*j+1].append(float(cWords[k+1]))
                xSol[iSat*(3*nSol+1)+3*j+2].append(float(cWords[k+2]))
                xSol[iSat*(3*nSol+1)+3*j+3].append(float(cWords[k+3]))
    nSat = len(cSat)
    fig, axs = plt.subplots(nSol, nSat, sharex='col',
                            squeeze=False, figsize=(3*nSat, nSol*2))
    formatterx = mpl.ticker.StrMethodFormatter('{x:7.1f}')
    for i in range(nSol):
        for j in range(nSat):
            axs[i, j].plot(xSol[j*(3*nSol+1)],
                           xSol[j*(3*nSol+1)+3*i+1], '.r', ms=2)
            axs[i, j].plot(xSol[j*(3*nSol+1)],
                           xSol[j*(3*nSol+1)+3*i+2], '.g', ms=2)
            axs[i, j].plot(xSol[j*(3*nSol+1)],
                           xSol[j*(3*nSol+1)+3*i+3], '.b', ms=2)
            axs[i, j].ticklabel_format(
                axis='y', style='sci', useOffset=False, useMathText=True)
            if i == 0:
                strTmp = cSat[j]+' POS'
            else:
                strTmp = cSat[j]+' PAR{:>02d}'.format(i)
            axs[i, j].text(0.01, 0.99, strTmp, transform=axs[i, j].transAxes, ha='left', va='top',
                           fontdict={'fontsize': 8, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkblue'})
            for tl in axs[i, j].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(6)
    for j in range(nSat):
        axs[i, j].xaxis.set_major_formatter(formatterx)
        axs[i, j].set_xlabel('Modified Julian Day',
                             fontname='Arial', fontsize=6)
        for tl in axs[i, j].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(6)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotAMat(fAMat, iPar1, iPar2, cSat0, OutFilePrefix, OutFileSuffix):
    '''
    Plot the design matrix of orbfit for cSat0

    NOTE: Here we assume that all satellites share the same parameters set
    '''

    if iPar1 > 0 and iPar2 >= iPar1:
        nSol = iPar2-iPar1+1
    else:
        # plot for all parameters
        iPar1 = 1
        with open(fAMat, mode='rt') as fOb:
            for cLine in fOb:
                cWords = cLine.split()
                nSol = (len(cWords)-3)//3
                break
    cSat = []
    xSol = []
    with open(fAMat, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cSat0[0] != 'ALL' and cWords[2] not in cSat0:
                continue
            if cWords[2] not in cSat:
                cSat.append(cWords[2])
                for i in range(3*nSol+1):
                    xSol.append([])
            iSat = cSat.index(cWords[2])
            xSol[iSat*(3*nSol+1)].append(int(cWords[0])+float(cWords[1])/86400)
            for j in range(nSol):
                k = 2 + 3*(iPar1-1) + j*3
                xSol[iSat*(3*nSol+1)+3*j+1].append(float(cWords[k+1]))
                xSol[iSat*(3*nSol+1)+3*j+2].append(float(cWords[k+2]))
                xSol[iSat*(3*nSol+1)+3*j+3].append(float(cWords[k+3]))
    nSat = len(cSat)
    fig, axs = plt.subplots(nSol, nSat, sharex='col',
                            squeeze=False, figsize=(3*nSat, nSol*2))
    formatterx = mpl.ticker.StrMethodFormatter('{x:7.1f}')
    for i in range(nSol):
        for j in range(nSat):
            axs[i, j].plot(xSol[j*(3*nSol+1)],
                           xSol[j*(3*nSol+1)+3*i+1], '.r', ms=2)
            axs[i, j].plot(xSol[j*(3*nSol+1)],
                           xSol[j*(3*nSol+1)+3*i+2], '.g', ms=2)
            axs[i, j].plot(xSol[j*(3*nSol+1)],
                           xSol[j*(3*nSol+1)+3*i+3], '.b', ms=2)
            axs[i, j].ticklabel_format(
                axis='y', style='sci', useOffset=False, useMathText=True)
            strTmp = cSat[j]+' PAR{:>02d}'.format(i+1)
            axs[i, j].text(0.01, 0.99, strTmp, transform=axs[i, j].transAxes, ha='left', va='top',
                           fontdict={'fontsize': 8, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkblue'})
            for tl in axs[i, j].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(6)
    for j in range(nSat):
        axs[i, j].xaxis.set_major_formatter(formatterx)
        axs[i, j].set_xlabel('Modified Julian Day',
                             fontname='Arial', fontsize=6)
        for tl in axs[i, j].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(6)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotOrbCor(fOrbCor, cSat0, OutFilePrefix, OutFileSuffix):
    '''
    Plot orbit correction because of velocity pulse based on output during orbupd
    '''

    cSat = []
    XCor = []
    with open(fOrbCor, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cSat0[0] != 'ALL' and cWords[2] not in cSat0:
                continue
            if cWords[2] not in cSat:
                cSat.append(cWords[2])
                for i in range(4):
                    XCor.append([])
            iSat = cSat.index(cWords[2])
            XCor[iSat*4].append(int(cWords[0])+float(cWords[1])/86400)
            XCor[iSat*4+1].append(float(cWords[3]))
            XCor[iSat*4+2].append(float(cWords[4]))
            XCor[iSat*4+3].append(float(cWords[5]))
    nSat = len(cSat)
    fig, axs = plt.subplots(nSat, 1, sharex='col',
                            squeeze=False, figsize=(6, nSat*2))
    formatterx = mpl.ticker.StrMethodFormatter('{x:7.1f}')
    for i in range(nSat):
        axs[i, 0].plot(XCor[i*4], XCor[i*4+1], '.r', ms=2)
        axs[i, 0].plot(XCor[i*4], XCor[i*4+2], '.g', ms=2)
        axs[i, 0].plot(XCor[i*4], XCor[i*4+3], '.b', ms=2)
        axs[i, 0].grid(which='both', axis='y', color='darkgray', linestyle='--',
                       linewidth=0.8)
        axs[i, 0].ticklabel_format(
            axis='y', style='sci', useOffset=False, useMathText=True)
        axs[i, 0].text(0.01, 0.99, cSat[i], transform=axs[i, 0].transAxes, ha='left', va='top',
                       fontdict={'fontsize': 8, 'fontname': 'Arial', 'fontweight': 'bold', 'color': 'darkblue'})
        axs[i, 0].set_ylabel('[cm]', fontname='Arial', fontsize=16)
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
    axs[i, 0].xaxis.set_major_formatter(formatterx)
    axs[i, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotOrbDiff(fOrb1, fOrb2):
    '''
    '''

    # Get the satellite list
    cSat1 = []
    with open(fOrb1, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cWords[2] in cSat1:
                continue
            cSat1.append(cWords[2])
    cSat = []
    with open(fOrb2, mode='rt') as fOb:
        for cLine in fOb:
            cWords = cLine.split()
            if cWords[2] not in cSat1:
                continue
            if cWords[2] in cSat:
                continue
            cSat.append(cWords[2])
    cSat.sort()
    nSat = len(cSat)

    fig, axs = plt.subplots(nSat, 1, sharex='col', figsize=(8, 50))
    fig.subplots_adjust(hspace=0.1)

    Orb1 = [[] for i in range(3*nSat)]
    Orb2 = [[] for i in range(3*nSat)]
    Diff = [[] for i in range(3*nSat)]
    rEpo = []
    for i in range(nSat):
        with open(fOrb1, mode='rt') as fOb:
            for cLine in fOb:
                cWords = cLine.split()
                if cWords[2] != cSat[i]:
                    continue
                if i == 0:
                    rEpo.append(int(cWords[0])+float(cWords[1])/86400.0)
                Orb1[i*3].append(float(cWords[3]))
                Orb1[i*3+1].append(float(cWords[4]))
                Orb1[i*3+2].append(float(cWords[5]))

        with open(fOrb2, mode='rt') as fOb:
            for cLine in fOb:
                cWords = cLine.split()
                if cWords[2] != cSat[i]:
                    continue
                Orb2[i*3].append(float(cWords[3]))
                Orb2[i*3+1].append(float(cWords[4]))
                Orb2[i*3+2].append(float(cWords[5]))

        # Cal the dif
        for j in range(len(rEpo)):
            Diff[i*3].append((Orb1[i*3][j]-Orb2[i*3][j])*1e5)
            Diff[i*3+1].append((Orb1[i*3+1][j]-Orb2[i*3+1][j])*1e5)
            Diff[i*3+2].append((Orb1[i*3+2][j]-Orb2[i*3+2][j])*1e5)

        axs[i].plot(rEpo, Diff[i*3], '.r')
        axs[i].plot(rEpo, Diff[i*3+1], '.g')
        axs[i].plot(rEpo, Diff[i*3+2], '.b')
        axs[i].text(0.05, 0.95, cSat[i], ha='left',
                    va='top', transform=axs[i].transAxes)

    # plt.show()
    fig.savefig('D:/Code/PROJECT/ERP_Test/OrbA.pdf', bbox_inches='tight')
    plt.close(fig)


def PlotCMat(fList, cSat0, nPar0, cPar0, OutFilePrefix):
    '''
    Plot the SRP cMat for specific satellites based on the output during
    orbit integration

    cSat0 --- Specified satellite list
    nPar0 --- Number of parameters for each specified satellite
    cPar0 --- Parameter lists. Note, this list must be sorted along the list in
              the global parameter list within the model fortran source file
    '''

    for iSat0 in range(len(cSat0)):
        X = [[]]
        for i in range(nPar0[iSat0]):
            for j in range(3):
                X.append([])
        # Read the cMat for this satellite
        for i in range(len(fList)):
            with open(fList[i], mode='r', encoding='UTF-16') as fOb:
                for cLine in fOb:
                    if cLine[0:4] != 'CMat':
                        continue
                    cWords = cLine.split()
                    if cWords[1] != cSat0[iSat0]:
                        continue
                    # Epoch
                    X[0].append(float(cWords[2])+float(cWords[3])/86400)
                    # CMat
                    for j in range(nPar0[iSat0]):
                        for k in range(3):
                            X[j*3+k+1].append(float(cWords[j*3+k+5]))
        if len(X[0]) == 0:
            sys.exit(cSat0[iSat0]+' not found')

        fig, axs = plt.subplots(
            nPar0[iSat0], 1, sharex='col', squeeze=False, figsize=(8, 3*nPar0[iSat0]))
        # fig.subplots_adjust(hspace=0.1)
        formatterx = mpl.ticker.StrMethodFormatter('{x:8.2f}')

        cLab = ['X', 'Y', 'Z']
        for j in range(nPar0[iSat0]):
            for k in range(3):
                axs[j, 0].plot(X[0], X[j*3+k+1], label=cLab[k])
            axs[j, 0].text(0.02, 0.98, cPar0[iSat0][j], color='darkgreen', weight='bold', family='Arial', size=14,
                           transform=axs[j, 0].transAxes, ha='left', va='top')
            for tl in axs[j, 0].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            axs[j, 0].legend(loc='upper right', framealpha=0.6,
                             prop={'family': 'Arial', 'size': 14})

        axs[j, 0].xaxis.set_major_formatter(formatterx)
        axs[j, 0].set_xlabel('Modified Julian Day',
                             fontname='Arial', fontsize=16)
        for tl in axs[j, 0].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)

        strTmp = OutFilePrefix+cSat0[iSat0]+'_CMat.png'
        fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
        plt.close(fig)


def PlotKep1(fList, cPRN0, OutFilePrefix, OutFileSuffix):
    '''
    Plot the orbit Kepler elements of specified satellites

    cPRN0 --- dependen on the first element in the list, i.e., cPRN0[0]
              ALL ,
              INCL, inclusive list
              EXCL, exclusive list

    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    cSat = []
    Kep = []
    for i in range(len(fList)):
        with open(fList[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:1] == '#' or len(cLine) < 20:
                    continue
                cWords = cLine.split()
                if cPRN0[0] != 'ALL':
                    if cPRN0[0] == 'INCL' and cWords[3] not in cPRN0:
                        continue
                    if cPRN0[0] == 'EXCL' and cWords[3] in cPRN0:
                        continue
                if cWords[3] not in cSat:
                    cSat.append(cWords[3])
                    for j in range(7):
                        Kep.append([])
                j = cSat.index(cWords[3])
                Kep[j*7].append(float(cWords[0])+float(cWords[1])/86400)
                # Semi-major axis, in meters
                Kep[j*7+1].append(float(cWords[4]))
                # Eccentricity
                Kep[j*7+2].append(float(cWords[5]))
                # Argument of perigee, Rad -> Deg
                Kep[j*7+3].append(np.rad2deg(float(cWords[6])))
                # RAAN,
                Kep[j*7+4].append(np.rad2deg(float(cWords[7])))
                # Inclination
                Kep[j*7+5].append(np.rad2deg(float(cWords[8])))
                # Mean anomaly
                Kep[j*7+6].append(np.rad2deg(float(cWords[9])))
    nSat = len(cSat)
    cPRN = cSat.copy()
    cPRN.sort()
    cKep = ['Semi-major axis', 'Eccentricity', 'Argument of perigee',
            'RAAN', 'Inclination', 'Mean anomaly']
    yLab = [r'$a$ [m]', r'$e$', r'$\omega$ [deg]',
            r'$\Omega$ [deg]', r'$i$ [deg]', r'$M$ [deg]']

    fig, axs = plt.subplots(6, nSat, sharex='col',
                            squeeze=False, figsize=(nSat*8, 3*6))

    # Report to the terminal
    strTmp = '{:<3s}'.format('PRN')
    for k in range(6):
        strTmp = strTmp+'{: >25s}'.format(cKep[k])
    print(strTmp)
    for i in range(nSat):
        j = cSat.index(cPRN[i])
        strTmp = '{:<3s}'.format(cPRN[i])
        for k in range(6):
            # if k==2 or k==3 or k==5:
            #     axs[k,i].set_ylim(bottom=0,top=360)
            # elif k==4:
            #     axs[k,i].set_ylim(bottom=0,top=180)
            axs[k, i].plot(Kep[7*j], Kep[7*j+k+1], '.r', ms=2)
            strTmp = strTmp + \
                '{: >15.4f} {: >9.4f}'.format(
                    np.mean(Kep[7*j+k+1]), np.std(Kep[7*j+k+1]))
            axs[k, i].grid(which='major', axis='y', color='darkgray', linestyle='--',
                           linewidth=0.8)
            axs[k, i].set_axisbelow(True)
            axs[k, i].text(0.02, 0.98, cPRN[i], color='darkgreen', weight='bold', family='Arial', size=14,
                           transform=axs[k, i].transAxes, ha='left', va='top')
            for tl in axs[k, i].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            if k == 1:
                axs[k, i].ticklabel_format(
                    axis='y', style='sci', useOffset=False, useMathText=True)
            else:
                axs[k, i].ticklabel_format(
                    axis='y', style='plain', useOffset=False, useMathText=True)
            axs[k, i].set_ylabel(yLab[k], fontsize=16)
        print(strTmp)
        axs[k, i].set_xlabel('Modified Julian Day',
                             fontname='Arial', fontsize=16)
        for tl in axs[k, i].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        axs[k, i].xaxis.set_major_formatter('{x:7.1f}')

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotKep20(fList1, fList2, cPRN0, OutFilePrefix, OutFileSuffix):
    '''
    Plot the diff of six Kepler elements from two file sets
    for specified satellites

    cPRN0 --- dependen on [0]
              ALL ,
              INCL, inclusive list
              EXCL, exclusive list
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    cSat1 = []
    Kep1 = []
    sEpo1 = []
    for i in range(len(fList1)):
        with open(fList1[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:1] == '#' or len(cLine) < 20:
                    continue
                cWords = cLine.split()
                if cPRN0[0] != 'ALL':
                    if cPRN0[0] == 'INCL' and cWords[3] not in cPRN0:
                        continue
                    if cPRN0[0] == 'EXCL' and cWords[3] in cPRN0:
                        continue
                if cWords[3] not in cSat1:
                    cSat1.append(cWords[3])
                    sEpo1.append([])
                    for j in range(7):
                        Kep1.append([])
                j = cSat1.index(cWords[3])
                # epoch string
                sEpo1[j].append(cWords[0]+' '+cWords[1])
                Kep1[j*7].append(float(cWords[0])+float(cWords[1])/86400)
                # Semi-major axis
                Kep1[j*7+1].append(float(cWords[4]))
                # Eccentricity
                Kep1[j*7+2].append(float(cWords[5]))
                # Argument of perigee, Rad -> Deg
                Kep1[j*7+3].append(np.rad2deg(float(cWords[6])))
                # RAAN,
                Kep1[j*7+4].append(np.rad2deg(float(cWords[7])))
                # Inclination
                Kep1[j*7+5].append(np.rad2deg(float(cWords[8])))
                # Mean anomaly
                Kep1[j*7+6].append(np.rad2deg(float(cWords[9])))
    cSat2 = []
    Kep2 = []
    sEpo2 = []
    for i in range(len(fList2)):
        with open(fList2[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:1] == '#' or len(cLine) < 20:
                    continue
                cWords = cLine.split()
                if cWords[3] not in cSat1:
                    continue
                if cWords[3] not in cSat2:
                    cSat2.append(cWords[3])
                    sEpo2.append([])
                    for j in range(7):
                        Kep2.append([])
                j = cSat2.index(cWords[3])
                # epoch string
                sEpo2[j].append(cWords[0]+' '+cWords[1])
                Kep2[j*7].append(float(cWords[0])+float(cWords[1])/86400)
                # Semi-major axis
                Kep2[j*7+1].append(float(cWords[4]))
                # Eccentricity
                Kep2[j*7+2].append(float(cWords[5]))
                # Argument of perigee, Rad -> Deg
                Kep2[j*7+3].append(np.rad2deg(float(cWords[6])))
                # RAAN,
                Kep2[j*7+4].append(np.rad2deg(float(cWords[7])))
                # Inclination
                Kep2[j*7+5].append(np.rad2deg(float(cWords[8])))
                # Mean anomaly
                Kep2[j*7+6].append(np.rad2deg(float(cWords[9])))
    nSat = len(cSat2)
    cPRN = cSat2.copy()
    cPRN.sort()
    cKep = ['Semi-major axis', 'Eccentricity',
            'Argument of perigee', 'RAAN', 'Inclination', 'Mean anomaly']
    yLab = [r'$a$ [cm]', r'$e$', r'$\omega$ [as]',
            r'$\Omega$ [as]', r'$i$ [as]', r'$M$ [as]']

    fig, axs = plt.subplots(6, nSat, sharex='col',
                            sharey='row', squeeze=False, figsize=(nSat*8, 3*6))
    formatterx = mpl.ticker.StrMethodFormatter('{x:7.1f}')

    for i in range(nSat):
        i1 = cSat1.index(cPRN[i])
        i2 = cSat2.index(cPRN[i])
        # Cal the diff
        dKep = [[], [], [], [], [], [], []]
        for j in range(len(sEpo1[i1])):
            if sEpo1[i1][j] not in sEpo2[i2]:
                continue
            k = sEpo2[i2].index(sEpo1[i1][j])
            # epoch
            dKep[0].append(Kep1[i1*7][j])
            # Semi-major axis, m -> cm
            dKep[1].append((Kep1[i1*7+1][j]-Kep2[i2*7+1][k])*1e2)
            # Eccentricity
            dKep[2].append(Kep1[i1*7+2][j]-Kep2[i2*7+2][k])
            # Argument of perigee, Deg -> arc sec
            xtmp = Kep1[i1*7+3][j]-Kep2[i2*7+3][k]
            if xtmp > 180:
                xtmp = xtmp-360
            elif xtmp < -180:
                xtmp = xtmp+360
            dKep[3].append(xtmp*3600)
            # RAAN
            xtmp = Kep1[i1*7+4][j]-Kep2[i2*7+4][k]
            if xtmp > 180:
                xtmp = xtmp-360
            elif xtmp < -180:
                xtmp = xtmp+360
            dKep[4].append(xtmp*3600)
            # Inclination
            dKep[5].append((Kep1[i1*7+5][j]-Kep2[i2*7+5][k])*3600)
            # Mean anomaly
            xtmp = Kep1[i1*7+6][j]-Kep2[i2*7+6][k]
            if xtmp > 180:
                xtmp = xtmp-360
            elif xtmp < -180:
                xtmp = xtmp+360
            dKep[6].append(xtmp*3600)
        for k in range(6):
            # Cal the mean, std && rms
            Mea = np.nanmean(dKep[1+k])
            Std = np.nanstd(dKep[1+k])
            nPoint = len(dKep[1+k])
            RMS = 0
            for j in range(nPoint):
                RMS = RMS + dKep[1+k][j]*dKep[1+k][j]
            RMS = np.sqrt(RMS/nPoint)
            strTmp = '{:>6.2f} {:>6.2f} {:>6.2f}'.format(Mea, Std, RMS)

            axs[k, i].plot(dKep[0], dKep[1+k], '.r', ms=2)
            axs[k, i].text(0.98, 0.98, strTmp, transform=axs[k, i].transAxes, ha='right', va='top',
                           fontdict={'fontsize': 14, 'fontname': 'Arial'})
            axs[k, i].grid(which='major', axis='y', color='darkgray', linestyle='--',
                           linewidth=0.8)
            axs[k, i].set_axisbelow(True)
            axs[k, i].text(0.02, 0.98, cPRN[i], color='darkgreen', weight='bold', family='Arial', size=14,
                           transform=axs[k, i].transAxes, ha='left', va='top')
            for tl in axs[k, i].get_yticklabels():
                tl.set_fontname('Arial')
                tl.set_fontsize(14)
            if k == 1:
                axs[k, i].ticklabel_format(
                    axis='y', style='sci', useOffset=False, useMathText=True)
            else:
                axs[k, i].ticklabel_format(
                    axis='y', style='plain', useOffset=False, useMathText=True)
            axs[k, i].set_ylabel(yLab[k], fontname='Arial', fontsize=16)

        axs[k, i].set_xlabel('Modified Julian Day',
                             fontname='Arial', fontsize=16)
        for tl in axs[k, i].get_xticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        # axs[k,i].xaxis.set_major_formatter(formatterx)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


def PlotKep21(fList1, fList2, cPRN0, cKep0, OutFilePrefix, OutFileSuffix):
    '''
    Plot the diff of one Kepler elements from two file sets
    for specified satellites

    cPRN0 --- dependen on the first element in the list, i.e., cPRN0[0]
              ALL ,
              INCL, inclusive list
              EXCL, exclusive list
    cKep0 ---
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    cSat1 = []
    Kep1 = []
    sEpo1 = []
    for i in range(len(fList1)):
        with open(fList1[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:1] == '#' or len(cLine) < 20:
                    continue
                cWords = cLine.split()
                if cPRN0[0] != 'ALL':
                    if cPRN0[0] == 'INCL' and cWords[3] not in cPRN0:
                        continue
                    if cPRN0[0] == 'EXCL' and cWords[3] in cPRN0:
                        continue
                if cWords[3] not in cSat1:
                    cSat1.append(cWords[3])
                    sEpo1.append([])
                    for j in range(7):
                        Kep1.append([])
                j = cSat1.index(cWords[3])
                # epoch string
                sEpo1[j].append(cWords[0]+' '+cWords[1])
                Kep1[j*7].append(float(cWords[0])+float(cWords[1])/86400)
                # Semi-major axis, m
                Kep1[j*7+1].append(float(cWords[4]))
                # Eccentricity
                Kep1[j*7+2].append(float(cWords[5]))
                # Argument of perigee, rad -> deg
                Kep1[j*7+3].append(np.rad2deg(float(cWords[6])))
                # RAAN, rad - > deg
                Kep1[j*7+4].append(np.rad2deg(float(cWords[7])))
                # Inclination, rad -> deg
                Kep1[j*7+5].append(np.rad2deg(float(cWords[8])))
                # Mean anomaly, rad -> deg
                Kep1[j*7+6].append(np.rad2deg(float(cWords[9])))
    cSat2 = []
    Kep2 = []
    sEpo2 = []
    for i in range(len(fList2)):
        with open(fList2[i], mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:1] == '#' or len(cLine) < 20:
                    continue
                cWords = cLine.split()
                if cWords[3] not in cSat1:
                    continue
                if cWords[3] not in cSat2:
                    cSat2.append(cWords[3])
                    sEpo2.append([])
                    for j in range(7):
                        Kep2.append([])
                j = cSat2.index(cWords[3])
                # epoch string
                sEpo2[j].append(cWords[0]+' '+cWords[1])
                Kep2[j*7].append(float(cWords[0])+float(cWords[1])/86400)
                # Semi-major axis, m
                Kep2[j*7+1].append(float(cWords[4]))
                # Eccentricity
                Kep2[j*7+2].append(float(cWords[5]))
                # Argument of perigee, rad -> deg
                Kep2[j*7+3].append(np.rad2deg(float(cWords[6])))
                # RAAN, rad -> deg
                Kep2[j*7+4].append(np.rad2deg(float(cWords[7])))
                # Inclination, rad -> deg
                Kep2[j*7+5].append(np.rad2deg(float(cWords[8])))
                # Mean anomaly, rad -> deg
                Kep2[j*7+6].append(np.rad2deg(float(cWords[9])))
    nSat = len(cSat2)
    cPRN = cSat2.copy()
    cPRN.sort()
    cKep = ['Semi-major axis', 'Eccentricity',
            'Argument of perigee', 'RAAN', 'Inclination', 'Mean anomaly']
    yLab = [r'$a$ [cm]', r'$e$', r'$\omega$ [as]',
            r'$\Omega$ [as]', r'$i$ [as]', r'$M$ [as]']
    iKep = cKep.index(cKep0)

    fig, axs = plt.subplots(nSat, 1, sharex='col',
                            squeeze=False, figsize=(8, 3*nSat))

    for i in range(nSat):
        i1 = cSat1.index(cPRN[i])
        i2 = cSat2.index(cPRN[i])
        # Cal the diff
        dKep = [[], [], [], [], [], [], []]
        for j in range(len(sEpo1[i1])):
            if sEpo1[i1][j] not in sEpo2[i2]:
                continue
            k = sEpo2[i2].index(sEpo1[i1][j])
            # epoch
            dKep[0].append(Kep1[i1*7][j])
            # Semi-major axis, m -> cm
            dKep[1].append((Kep1[i1*7+1][j]-Kep2[i2*7+1][k])*1e2)
            # Eccentricity
            dKep[2].append(Kep1[i1*7+2][j]-Kep2[i2*7+2][k])
            # Argument of perigee, deg -> arcsec
            xtmp = Kep1[i1*7+3][j]-Kep2[i2*7+3][k]
            if xtmp > 180:
                xtmp = xtmp-360
            elif xtmp < -180:
                xtmp = xtmp+360
            dKep[3].append(xtmp*3600)
            # RAAN, deg -> arcsec
            xtmp = Kep1[i1*7+4][j]-Kep2[i2*7+4][k]
            if xtmp > 180:
                xtmp = xtmp-360
            elif xtmp < -180:
                xtmp = xtmp+360
            dKep[4].append(xtmp*3600)
            # Inclination, deg -> arcsec
            dKep[5].append((Kep1[i1*7+5][j]-Kep2[i2*7+5][k])*3600)
            # Mean anomaly, deg -> arcsec
            xtmp = Kep1[i1*7+6][j]-Kep2[i2*7+6][k]
            if xtmp > 180:
                xtmp = xtmp-360
            elif xtmp < -180:
                xtmp = xtmp+360
            dKep[6].append(xtmp*3600)

        # Cal the mean, std && rms
        Mea = np.nanmean(dKep[1+iKep])
        Std = np.nanstd(dKep[1+iKep])
        nPoint = len(dKep[1+iKep])
        RMS = 0
        for j in range(nPoint):
            RMS = RMS + dKep[1+iKep][j]*dKep[1+iKep][j]
        RMS = np.sqrt(RMS/nPoint)
        strTmp = '{:>6.2f} +/- {:>6.2f} (RMS={:>6.2f})'.format(Mea, Std, RMS)

        axs[i, 0].plot(dKep[0], dKep[1+iKep], '.r', ms=2)
        axs[i, 0].text(0.98, 0.98, strTmp, transform=axs[i, 0].transAxes, ha='right', va='top',
                       fontdict={'fontsize': 14, 'fontname': 'Arial',
                                 'fontweight': 'bold'},
                       color='darkred')
        axs[i, 0].grid(which='major', axis='y', color='darkgray', linestyle='--',
                       linewidth=0.8)
        axs[i, 0].set_axisbelow(True)
        axs[i, 0].text(0.02, 0.98, cPRN[i], color='darkgreen', weight='bold', family='Arial', size=14,
                       transform=axs[i, 0].transAxes, ha='left', va='top')
        for tl in axs[i, 0].get_yticklabels():
            tl.set_fontname('Arial')
            tl.set_fontsize(14)
        if iKep == 1:
            axs[i, 0].ticklabel_format(
                axis='y', style='sci', useOffset=False, useMathText=True)
        else:
            axs[i, 0].ticklabel_format(
                axis='y', style='plain', useOffset=False, useMathText=True)
        axs[i, 0].set_ylabel(yLab[iKep], fontname='Arial', fontsize=16)

    axs[i, 0].set_xlabel('Modified Julian Day', fontname='Arial', fontsize=16)
    for tl in axs[i, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)
    axs[i, 0].xaxis.set_major_formatter('{x:7.1f}')

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
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

    InFilePrefix = os.path.join(
        cWrkPre0, r'PRO_2019001_2020366_WORK/I010/WORK20193??/')
    # InFilePrefix=r'Y:/PRO_2019001_2020366_WORK/C02/WORK20193??/'
    # InFilePrefix=r'Y:/tmp/WORK2019001_ERROR/'

    # OutFilePrefix=r'D:/Code/PROJECT/WORK2019335_ERROR/'
    OutFilePrefix = os.path.join(cDskPre0, r'PRO_2019001_2020366/I010/ORB/')
    # OutFilePrefix = os.path.join(
    #     cWrkPre0, r'PRO_2019001_2020366_WORK/I010/WORK2019335/')

    # fOrb=r'D:/Code/PROJECT/WORK2019335_ERROR/OrbA_2019335'
    # OutFileSuffix='OrbVarEqu.pdf'
    # PlotVarEqu(fOrb,0,0,['G01','G21'],OutFilePrefix,OutFileSuffix)

    # fAMat=r'D:/Code/PROJECT/WORK2019335_ERROR/AMat_2019335'
    # OutFileSuffix='AMat.pdf'
    # PlotAMat(fAMat,0,0,['G01','G21'],OutFilePrefix,OutFileSuffix)
    # fOrbCor=r'D:/Code/PROJECT/WORK2019335_ERROR/dorb_2019335'
    # fOrbCor=r'Y:/PRO_2019001_2020366_WORK/I2_G_PSV/WORK2019335/dorb_2019335'
    # OutFileSuffix='OrbCor_2019335.pdf'
    # PlotOrbCor(fOrbCor,['ALL'],OutFilePrefix,OutFileSuffix)

    # fList=glob.glob(InFilePrefix+'log_oi')

    cSat0 = ['C01', 'C02', 'C03', 'C04', 'C05']
    nPar0 = [5, 5, 5, 5, 5]
    cPar0 = [['BOXW_Y0', 'BOXW_+XR', 'BOXW_+ZR', 'BOXW_-ZR', 'BOXW_SD'],
             ['BOXW_Y0', 'BOXW_+XR', 'BOXW_+ZR', 'BOXW_-ZR', 'BOXW_SD'],
             ['BOXW_Y0', 'BOXW_+XR', 'BOXW_+ZR', 'BOXW_-ZR', 'BOXW_SD'],
             ['BOXW_Y0', 'BOXW_+XR', 'BOXW_+ZR', 'BOXW_-ZR', 'BOXW_SD'],
             ['BOXW_Y0', 'BOXW_+XR', 'BOXW_+ZR', 'BOXW_-ZR', 'BOXW_SD']]
    # PlotCMat(fList,cSat0,nPar0,cPar0,OutFilePrefix)

    fList = glob.glob(InFilePrefix+'kep_20193??')

    OutFileSuffix = 'kep_2019335_C23'
    # PlotKep1(fList, ['INCL', 'C23'], OutFilePrefix, OutFileSuffix)

    fList2 = glob.glob(os.path.join(
        cWrkPre0, r'PRO_2019001_2020366_WORK/C02/WORK20193??/kep_20193??'))
    # OutFileSuffix='DiffKep_C02'
    # PlotKep20(fList,fList2,['ALL','C29','C33','C34'],OutFilePrefix,OutFileSuffix)

    OutFileSuffix = 'DiffKep_C02_RAAN_C23'
    PlotKep21(fList, fList2, ['INCL', 'C23'],
              'RAAN', OutFilePrefix, OutFileSuffix)
