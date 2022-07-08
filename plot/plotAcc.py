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
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def GetForceAcc1(fList,cSat0):
    '''
    Get the ForceAcc records from the output file during oi, each acc term is
    presented in 3D forms, i.e. only the magnitude. All are in m/s*s
    # 0, Central body
    # 1, Third body
    # 2, Non-spheric Earth
    # 3, Solar Radiation
    # 4, Relativity
    # 5, Earth Albedo
    # 6, Antenna thrust
    # 7, PSV
    # 8, All conservative
    # 9, All dissipative
    '''

    # Number acc terms in the file
    nAcc=10

    cSat=[]; Acc=[]
    for i in range(len(fList)):
        with open(fList[i],mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:8] != 'ForceAcc':
                    continue
                cWords=cLine.split()
                if cSat0[0] != 'ALL' and cWords[1] not in cSat0:
                    continue
                if cWords[1] not in cSat:
                    cSat.append(cWords[1])
                    for j in range(nAcc+1):
                        Acc.append([])
                j=cSat.index(cWords[1])
                rMJD=float(cWords[2])+float(cWords[3])/86400
                Acc[j*(nAcc+1)].append(rMJD)
                for k in range(nAcc):
                    a=np.sqrt(float(cWords[4+k*3+1])**2 + \
                              float(cWords[4+k*3+2])**2 + \
                              float(cWords[4+k*3+3])**2)
                    Acc[j*(nAcc+1)+1+k].append(a)
    return nAcc,cSat,Acc

def GetForceAcc2(fList,cSat0):
    '''
    Get the ForceAcc records from the output file during oi, each acc term is
    presented in three components, i.e. XYZ or ACR.  All are in m/s*s
    # 0, Central body
    # 1, Third body
    # 2, Non-spheric Earth
    # 3, Solar Radiation
    # 4, Relativity
    # 5, Earth Albedo
    # 6, Antenna thrust
    # 7, PSV
    # 8, All conservative
    # 9, All dissipative
    '''

    # Number acc terms in the file:
    nAcc=10

    cSat=[]; Acc0=[]
    for i in range(len(fList)):
        with open(fList[i],mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:8] != 'ForceAcc':
                    continue
                cWords=cLine.split()
                if cSat0[0] != 'ALL' and cWords[1] not in cSat0:
                    continue
                if cWords[1] not in cSat:
                    cSat.append(cWords[1])
                    for j in range(nAcc*3+1):
                        Acc0.append([])
                j=cSat.index(cWords[1])
                rMJD=float(cWords[2])+float(cWords[3])/86400
                Acc0[j*(nAcc*3+1)].append(rMJD)
                for k in range(nAcc*3):
                    Acc0[j*(nAcc*3+1)+1+k].append(float(cWords[5+k]))
    # Sort the records along epoch
    Acc=[]
    for i in range(len(cSat)):
        # Sort the epoch list for this satellite
        ind=np.argsort(Acc0[i*(nAcc*3+1)])
        for j in range(nAcc*3+1):
            Acc.append([])
            for k in range(ind.size):
                Acc[i*(nAcc*3+1)+j].append(Acc0[i*(nAcc*3+1)+j][ind[k]])
    return nAcc,cSat,Acc

def GetSunlight(fList,cSat0):
    '''
    Get the Sunlight records from the output file during oi
    '''

    # Number of (meaningful) words within this record
    nWd=26

    cSat=[]; X=[]
    for i in range(len(fList)):
        with open(fList[i],mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:8] != 'Sunlight':
                    continue
                cWords=cLine.split()
                if cSat0[0] != 'ALL' and cWords[1] not in cSat0:
                    continue
                if cWords[1] not in cSat:
                    cSat.append(cWords[1])
                    for j in range(nWd+1):
                        X.append([])
                j=cSat.index(cWords[1])
                rMJD=float(cWords[2])+float(cWords[3])/86400
                X[j*(nWd+1)].append(rMJD)
                for k in range(nWd):
                    X[j*(nWd+1)+1+k].append(float(cWords[k+5]))
    return nWd,cSat,X


def PlotSunLight0(fList,cSat0,OutFilePrefix,OutFileSuffix):
    '''
    Plot Sun light information for a box-wing satellite. Rec index 1-5

    fList --- list of files contaning the sun light info
              for each body side and solar panle derived from
              Box-Wing model
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    nWd,cSat,X=GetSunlight(fList,cSat0)
    nSat=len(cSat); cSat1=cSat.copy(); cSat1.sort()

    fig,axs=plt.subplots(nSat,1,sharex='col',squeeze=False,figsize=(8,4*nSat))

    for i in range(nSat):
        j=cSat.index(cSat1[i])
        Ang=[[],[],[],[],[],[]]
        for k in range(len(X[j*(nWd+1)])):
            # MJD
            Ang[0].append(X[j*(nWd+1)  ][k])
            # +X
            Ang[1].append(X[j*(nWd+1)+1][k])
            # +Y
            Ang[2].append(X[j*(nWd+1)+2][k]+180)
            # +Z
            Ang[3].append(X[j*(nWd+1)+3][k]+360)
            # +SP
            Ang[4].append(X[j*(nWd+1)+4][k]+540)
            Ang[5].append(X[j*(nWd+1)+5][k])
        axs[i,0].invert_yaxis()
        axs[i,0].set_ylim(bottom=720,top=0)
        axs[i,0].set_yticks([0,90,270,450,630,720],minor=False)
        axs[i,0].set_yticklabels([r'$0^\circ$',r'$90^\circ$',r'$90^\circ$',
                                  r'$90^\circ$',r'$90^\circ$',r'$180^\circ$'])
        for tl in axs[i,0].get_yticklabels():
            tl.set_fontname('Arial'); tl.set_fontsize(14)
        axs[i,0].set_ylabel('Angle btw the Sun and Normal',fontname='Arial',fontsize=16)

        axs[i,0].grid(b=True,which='both',axis='y',color='darkgray',linestyle='--',
                      linewidth=0.8)
        axs[i,0].plot(Ang[0],Ang[1],'.',ms=6,markerfacecolor='gold',markeredgecolor='gold',label='+X')
        axs[i,0].plot(Ang[0],Ang[2],'.',ms=6,markerfacecolor='gold',markeredgecolor='gold',label='+Y')
        axs[i,0].plot(Ang[0],Ang[3],'.',ms=6,markerfacecolor='gold',markeredgecolor='gold',label='+Z')
        axs[i,0].plot(Ang[0],Ang[4],'.',ms=6,markerfacecolor='gold',markeredgecolor='gold',label='SP')

        axs[i,0].text(0.01,0.99,cSat[i],color='darkred',weight='bold',family='Arial',size=18,
                      transform=axs[i,0].transAxes,ha='left',va='top')

        axs[i,0].text(0.99,0.99,'+X',color='darkgreen',weight='bold',family='monospace',size=18,
                      transform=axs[i,0].transAxes,ha='right',va='top')
        axs[i,0].axhline(y=180,color='dimgrey',alpha=0.3)
        axs[i,0].text(0.99,0.74,'+Y',color='darkgreen',weight='bold',family='monospace',size=18,
                      transform=axs[i,0].transAxes,ha='right',va='top')
        axs[i,0].axhline(y=360,color='dimgrey',alpha=0.3)
        axs[i,0].text(0.99,0.49,'+Z',color='darkgreen',weight='bold',family='monospace',size=18,
                      transform=axs[i,0].transAxes,ha='right',va='top')
        axs[i,0].axhline(y=540,color='dimgrey',alpha=0.3)
        axs[i,0].text(0.99,0.24,'SP',color='darkgreen',weight='bold',family='monospace',size=18,
                      transform=axs[i,0].transAxes,ha='right',va='top')
    for tl in axs[i,0].get_xticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)
    axs[i,0].set_xlabel('Modified Julian Day',fontname='Arial',fontsize=16)
    axs[i,0].ticklabel_format(axis='x',useOffset=False,useMathText=True)

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    strTmp=OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotSunLight1(fList,OutFilePrefix,OutFileSuffix):
    '''
    Plot the coefficients of Box-Wing model when expressed in DYB
    '''

    cSat=[]; X=[]

    for i in range(len(fList)):
        with open(fList[i],mode='r',encoding='UTF-16') as fOb:
            for cLine in fOb:
                if cLine[0:8] != 'Sunlight':
                    continue
                cWords=cLine.split()
                if cWords[1] not in cSat:
                    #Satellite PRN
                    cSat.append(cWords[1])
                    X.append([])
                    for j in range(15):
                        X[len(cSat)-1].append([])
                iSat=cSat.index(cWords[1])
                #Epoch
                X[iSat][0].append(float(cWords[2])+float(cWords[3])/86400)
                #D const
                X[iSat][1].append(float(cWords[10]))
                #D 1-cpr sin
                X[iSat][2].append(float(cWords[12]))
                #D 1-cpr cos
                X[iSat][3].append(float(cWords[14]))
                #D 2-cpr cos
                X[iSat][5].append(float(cWords[18]))
                #D 3-cpr sin
                X[iSat][6].append(float(cWords[20]))
                #D 3-cpr cos
                X[iSat][7].append(float(cWords[22]))

                #B 1-cpr sin
                X[iSat][ 9].append(float(cWords[13]))
                #B 1-cpr cos
                X[iSat][10].append(float(cWords[15]))
                #B 2-cpr sin
                X[iSat][11].append(float(cWords[17]))
                #B 3-cpr sin
                X[iSat][13].append(float(cWords[21]))
                #B 3-cpr cos
                X[iSat][14].append(float(cWords[23]))
    cSat.sort(); nSat=len(cSat)

    fig,axs=plt.subplots(nSat,2,sharex='col',sharey='row',squeeze=False,figsize=(12,3*nSat))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.1)

    for i in range(nSat):
        axs[i,0].text(0.01,0.99,cSat[i],color='darkred',weight='bold',family='monospace',size=18,
                      transform=axs[i,0].transAxes,ha='left',va='top')
        axs[i,0].plot(X[i][0],X[i][1],ls='-',lw=0.5,marker='.',mfc='r',mec='r',c='r',ms=2,markevery=10,label='D0')
        axs[i,0].plot(X[i][0],X[i][2],ls='-',lw=0.5,marker='v',mfc='g',mec='g',c='g',ms=2,markevery=10,label='Ds1')
        axs[i,0].plot(X[i][0],X[i][3],ls='-',lw=0.5,marker='1',mfc='b',mec='b',c='b',ms=2,markevery=10,label='Dc1')
        axs[i,0].plot(X[i][0],X[i][5],ls='-',lw=0.5,marker='8',mfc='c',mec='c',c='c',ms=2,markevery=10,label='Dc2')
        axs[i,0].plot(X[i][0],X[i][6],ls='-',lw=0.5,marker='*',mfc='m',mec='m',c='m',ms=2,markevery=10,label='Ds3')
        axs[i,0].plot(X[i][0],X[i][7],ls='-',lw=0.5,marker='x',mfc='y',mec='y',c='y',ms=2,markevery=10,label='Dc3')
        axs[i,0].legend(ncol=1,labelcolor='mfc',framealpha=0.6,labelspacing=0.1,
                        borderpad=0.1,handlelength=0.5,columnspacing=0.8,loc='upper right')

        axs[i,1].plot(X[i][0],X[i][ 9],ls='-',lw=0.5,marker='v',mfc='g',mec='g',c='g',ms=2,markevery=10,label='Bs1')
        axs[i,1].plot(X[i][0],X[i][10],ls='-',lw=0.5,marker='1',mfc='b',mec='b',c='b',ms=2,markevery=10,label='Bc1')
        axs[i,1].plot(X[i][0],X[i][11],ls='-',lw=0.5,marker='s',mfc='k',mec='k',c='k',ms=2,markevery=10,label='Bs2')
        axs[i,1].plot(X[i][0],X[i][13],ls='-',lw=0.5,marker='*',mfc='m',mec='m',c='m',ms=2,markevery=10,label='Bs3')
        axs[i,1].plot(X[i][0],X[i][14],ls='-',lw=0.5,marker='x',mfc='y',mec='y',c='y',ms=2,markevery=10,label='Bc3')
        axs[i,1].legend(ncol=1,labelcolor='mfc',framealpha=0.6,labelspacing=0.1,
                        borderpad=0.1,handlelength=0.5,columnspacing=0.8,loc='upper right')

    strTmp=OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotSunLight2(fList,cSat0,OutFilePrefix,OutFileSuffix):
    '''
    Plot BW-based SRP acc and the diff between two calculation methods for
    cross-validation. Rec index 20-25
    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nWd,cSat,X=GetSunlight(fList,cSat0)
    nSat=len(cSat); cSat1=cSat.copy(); cSat1.sort()

    fig,axs=plt.subplots(nSat,1,sharex='col',squeeze=False,figsize=(12,3*nSat))

    for i in range(nSat):
        j=cSat.index(cSat1[i])
        axs[i,0].text(0.02,0.98,cSat[j],color='darkred',weight='bold',family='monospace',size=14,
                      transform=axs[i,0].transAxes,ha='left',va='top')
        axs[i,0].set_ylabel(r'Acc [$\mathregular{nm}/\mathregular{s}^\mathregular{2}$]',
                            fontname='Arial',fontsize=16)
        axs[i,0].ticklabel_format(axis='y',useOffset=False,useMathText=True)
        for tl in axs[i,0].get_yticklabels():
            tl.set_fontname('Arial'); tl.set_fontsize(14)
        axs[i,0].plot(X[j*(nWd+1)],X[j*(nWd+1)+20],'-r',lw=1,label='D')
        axs[i,0].plot(X[j*(nWd+1)],X[j*(nWd+1)+21],'-g',lw=1,label='Y')
        axs[i,0].plot(X[j*(nWd+1)],X[j*(nWd+1)+22],'-b',lw=1,label='B')
        axs[i,0].legend(loc='upper right',prop={'family':'Arial','size':14})

        axe=axs[i,0].twinx()
        axe.set_ylabel(r'Diff [$\mathregular{nm}/\mathregular{s}^\mathregular{2}$]',
                            fontname='Arial',fontsize=16)
        for tl in axe.get_yticklabels():
            tl.set_fontname('Arial'); tl.set_fontsize(14)
        axe.plot(X[j*(nWd+1)],X[j*(nWd+1)+23],'or',ls='',ms=2,label='dD')
        axe.plot(X[j*(nWd+1)],X[j*(nWd+1)+24],'og',ls='',ms=2,label='dY')
        axe.plot(X[j*(nWd+1)],X[j*(nWd+1)+25],'ob',ls='',ms=2,label='dB')
    axs[i,0].set_xlabel(r'Modified Julian Day',fontname='Arial',fontsize=16)
    axs[i,0].ticklabel_format(axis='x',useOffset=False,useMathText=True)
    for tl in axs[i,0].get_xticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    strTmp=OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)


def PlotForceAcc0(fList,cSat0,OutFilePrefix,OutFileSuffix):
    '''
    Plot all acc terms (in 3D) for specified satellites

    cSat0 --- Specified satellites
    '''

    nAcc,cSat,Acc=GetForceAcc1(fList,cSat0)
    nSat=len(cSat); cSat1=cSat.copy(); cSat1.sort()

    # Number of terms to be plotted
    nForce=9
    cForce=['2-body','3-body','Non-spheric Earth','SRP','Relativity',
            'Albedo','Thrust','Conservative','Dissipative']
    # index of those to-be-plotted terms
    iForce=[0,1,2,3,4,5,6,8,9]

    fig,axs=plt.subplots(nSat,1,sharex='col',squeeze=False,figsize=(8,nSat*10))

    for i in range(nSat):
        axs[i,0].set_ylim(bottom=1e-10,top=10)
        axs[i,0].set_yticks([1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1],minor=False)
        for tl in axs[i,0].get_yticklabels():
            tl.set_fontname('Arial'); tl.set_fontsize(14)
        axs[i,0].get_yaxis().set_major_formatter(mpl.ticker.LogFormatterSciNotation())
        axs[i,0].grid(b=True,which='both',axis='y',color='darkgray',linestyle='--',linewidth=0.8)
        axs[i,0].set_yscale('log',subs=[2,3,4,5,6,7,8,9])
        axs[i,0].set_ylabel(r'Acceleration ($m/s^2$)',fontname='Arial',fontsize=16)

        j=cSat.index(cSat1[i])
        for k in range(nForce):
            if iForce[k]<0:
                continue
            axs[i,0].plot(Acc[j*(nAcc+1)],Acc[j*(nAcc+1)+1+iForce[k]],'.-',ms=2,
                         label=cForce[k])

        axs[i,0].legend(ncol=3,loc='upper center',bbox_to_anchor=(0.5,0.8),
                        prop={'family':'Arial','size':14})
        axs[i,0].text(0.01,0.99,cSat[j],transform=axs[i,0].transAxes,ha='left',va='top',
                      family='Arial',size=16)
    axs[i,0].set_xlabel(r'Modified Julian Day',fontname='Arial',fontsize=16)
    axs[i,0].ticklabel_format(axis='x',useOffset=False,useMathText=True)
    for tl in axs[i,0].get_xticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)

    strTmp=OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotForceAcc1(fList,cSat0,iAcc,iTyp,nCol,OutFilePrefix,OutFileSuffix):
    '''
    Plot one specified acc term for specified satellites

    The three components wil be plotted either within the same axis or
    three respective axis.

    iAcc --- index (start from 0) of the specified acc term to be plotted
    # 0, Central body
    # 1, Third body
    # 2, Non-spheric Earth
    # 3, Solar Radiation
    # 4, Relativity
    # 5, Earth Albedo
    # 6, Antenna thrust
    # 7, PSV
    # 8, All conservative
    # 9, All dissipative

    iTyp --- whether three components will be plotted within one or three
             axis
             # 1, within one axes
             # 3, within three axes
    nCol --- Number of columns when iTyp==1
    '''

    if iAcc<0:
        sys.exit('Acc index should not be negative!')
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nAcc,cSat,Acc=GetForceAcc2(fList,cSat0)
    nSat=len(cSat); cSat1=cSat.copy(); cSat1.sort()

    if iTyp==1:
        # Cal the number of row based on specified number of col
        nRow=math.ceil(nSat/nCol)
        fig,axs=plt.subplots(nRow,nCol,sharex='col',sharey='row',
                             squeeze=False,figsize=(nCol*8,nRow*2.5))
        fig.subplots_adjust(hspace=0.1)
        fig.subplots_adjust(wspace=0.05)
        for i in range(nSat):
            # Cal the axis position, row-wise
            iRow=math.ceil((i+1)/nCol)-1; iCol=i-iRow*nCol
            if iAcc==5 or iAcc==6:
                # Earth Albedo or Antenna thrust
                cLab=['Along','Cross','Radial']
            elif iAcc==3:
                # SRP
                cLab=['D','Y','B']
            else:
                cLab=['X','Y','Z']

            axs[iRow,iCol].grid(which='both',axis='y',c='darkgray',ls='--',lw=0.8)

            j=cSat.index(cSat1[i])
            axs[iRow,iCol].plot(Acc[j*(nAcc*3+1)],Acc[j*(nAcc*3+1)+iAcc*3+1],'r',lw=1,label=cLab[0])
            axs[iRow,iCol].plot(Acc[j*(nAcc*3+1)],Acc[j*(nAcc*3+1)+iAcc*3+2],'g',lw=1,label=cLab[1])
            axs[iRow,iCol].plot(Acc[j*(nAcc*3+1)],Acc[j*(nAcc*3+1)+iAcc*3+3],'b',lw=1,label=cLab[2])
            # Cal the 3D acc
            X=[]
            for k in range(len(Acc[j*(nAcc*3+1)])):
                a=Acc[j*(nAcc*3+1)+iAcc*3+1][k]**2 + \
                  Acc[j*(nAcc*3+1)+iAcc*3+2][k]**2 + \
                  Acc[j*(nAcc*3+1)+iAcc*3+3][k]**2
                X.append(np.sqrt(a))
            # axs[iRow,iCol].plot(Acc[j*(nAcc*3+1)],X,'s--c',ms=2,lw=1,label='3D')

            axs[iRow,iCol].legend(ncol=4,loc='upper center',bbox_to_anchor=(0.5,1.0),
                                  prop={'family':'Arial','size':14})
            axs[iRow,iCol].text(0.01,0.99,cSat[j],transform=axs[iRow,iCol].transAxes,ha='left',va='top',
                                family='Arial',size=16,weight='bold')
            if iCol==0:
                axs[iRow,iCol].set_ylabel(r'Acc [$\mathregular{m}/\mathregular{s}^\mathregular{2}$]',
                                          fontname='Arial',fontsize=16)
                # Set the y-axis for different forces to get a better presentation
                if iAcc==5 or iAcc==6:
                    # Earth Albedo or Antenna thrust
                    axs[iRow,iCol].set_ylim(bottom=-3e-9,top=3e-9)
                    axs[iRow,iCol].set_yticks([-2.0e-9,-1.0e-9,-0.5e-9,0,0.5e-9,1.0e-9,2.0e-9],minor=True)
                elif iAcc==3:
                    # SRP
                    axs[iRow,iCol].set_ylim(bottom=-2e-7,top=2e-7)
                else:
                    axs[iRow,iCol].set_ylim(bottom=-1e-8,top=1e-8)
                for tl in axs[iRow,iCol].get_yticklabels():
                    tl.set_fontname('Arial'); tl.set_fontsize(14)
                axs[iRow,iCol].ticklabel_format(axis='y',useOffset=False,useMathText=True)
            if iRow==(nRow-1):
                axs[iRow,iCol].set_xlabel(r'Modified Julian Day',fontname='Arial',fontsize=16)
                axs[iRow,iCol].ticklabel_format(axis='x',useOffset=False,useMathText=True)
                for tl in axs[iRow,iCol].get_xticklabels():
                    tl.set_fontname('Arial'); tl.set_fontsize(14)
    else:
        fig,axs=plt.subplots(nSat,3,sharex='col',squeeze=False,figsize=(20,nSat*3))
        for i in range(nSat):
            # plotted in three axises
            if iAcc==5 or iAcc==6:
                # Earth Albedo or Antenna thrust
                # axs[i,0].set_ylim(bottom=-2e-9,top=2e-9)
                # axs[i,0].set_yticks([-1.0e-9,-0.5e-9,0,0.5e-9,1.0e-9],minor=True)
                cLab=['A','C','R']
            elif iAcc==3:
                # SRP
                # axs[i,0].set_ylim(bottom=-2e-7,top=2e-7)
                cLab=['D','Y','B']
            else:
                # axs[i,0].set_ylim(bottom=-1e-8,top=1e-8)
                cLab=['X','Y','Z']

            axs[i,0].set_ylabel(r'Acc [$\mathregular{m}/\mathregular{s}^\mathregular{2}$]',
                                fontname='Arial',fontsize=16)
            j=cSat.index(cSat1[i])
            for k in range(3):
                axs[i,k].ticklabel_format(axis='y',useOffset=False,useMathText=True)
                axs[i,k].grid(which='major',axis='y',c='darkgray',ls='--',lw=0.8)
                for tl in axs[i,k].get_yticklabels():
                    tl.set_fontname('Arial'); tl.set_fontsize(14)
                axs[i,k].plot(Acc[j*(nAcc*3+1)],Acc[j*(nAcc*3+1)+iAcc*3+1+k],lw=3)
                axs[i,k].text(0.01,0.99,cSat[j]+' '+cLab[k],transform=axs[i,k].transAxes,
                                ha='left',va='top',family='Arial',size=16,weight='bold')
        for k in range(2):
            axs[i,1+k].set_xlabel(r'Modified Julian Day',fontname='Arial',fontsize=16)
            axs[i,1+k].ticklabel_format(axis='x',useOffset=False,useMathText=True)
            for tl in axs[i,1+k].get_xticklabels():
                tl.set_fontname('Arial'); tl.set_fontsize(14)

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    strTmp=OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotForceAcc2(fList,cSat0,iAccList,OutFilePrefix,OutFileSuffix):
    '''
    Plot several specified acc terms (in 3D) for specified satellites

    iAccList --- index list (start from 0) of the specified acc terms to be plotted
    # 0, Central body
    # 1, Third body
    # 2, Non-spheric Earth
    # 3, Solar Radiation
    # 4, Relativity
    # 5, Earth Albedo
    # 6, Antenna thrust
    # 7, PSV
    # 8, All conservative
    # 9, All dissipative
    '''

    nAcc,cSat,Acc=GetForceAcc1(fList,cSat0)
    nSat=len(cSat); cSat1=cSat.copy(); cSat1.sort()

    fig,axs=plt.subplots(nSat,1,sharex='col',squeeze=False,figsize=(8,nSat*3))

    for i in range(nSat):
        # axs[i,0].set_ylim(bottom=1e-10,top=10)
        # axs[i,0].set_yticks([1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1],minor=False)
        for tl in axs[i,0].get_yticklabels():
            tl.set_fontname('Arial'); tl.set_fontsize(14)
        axs[i,0].get_yaxis().set_major_formatter(mpl.ticker.LogFormatterSciNotation())
        axs[i,0].grid(b=True,which='both',axis='y',color='darkgray',linestyle='--',linewidth=0.8)
        # axs[i,0].set_yscale('log',subs=[2,3,4,5,6,7,8,9])
        axs[i,0].set_ylabel(r'Acceleration ($m/s^2$)',fontname='Arial',fontsize=16)

        j=cSat.index(cSat1[i])
        for k in range(len(iAccList)):
            if iAccList[k]<0:
                continue
            axs[i,0].plot(Acc[j*(nAcc+1)],Acc[j*(nAcc+1)+iAccList[k]+1],'.-',ms=2)
        axs[i,0].text(0.01,0.99,cSat[j],transform=axs[i,0].transAxes,ha='left',va='top',
                      family='Arial',size=16)
    axs[i,0].set_xlabel(r'Modified Julian Day',fontname='Arial',fontsize=16)
    axs[i,0].ticklabel_format(axis='x',useOffset=False,useMathText=True)
    for tl in axs[i,0].get_xticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)

    strTmp=OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotAccDiff0(fList1,fList2,cSat0,iAcc,iTyp,OutFilePrefix,OutFileSuffix):
    '''
    Plot the diff of one specified acc term between two lists for
    specified satellites

    All three omponents wil be plotted either within the same axis or
    three respective axis.

    NOTE: Be carefull the diff calculation depends on that the files in list
          are in Chronological order.

    iAcc --- index (start from 0) of the specified acc term to be plotted
    # 0, Central body
    # 1, Third body
    # 2, Non-spheric Earth
    # 3, Solar Radiation
    # 4, Relativity
    # 5, Earth Albedo
    # 6, Antenna thrust
    # 7, PSV
    # 8, All conservative
    # 9, All dissipative

    iTyp --- whether three components will be plotted within one or three
             axis
    '''

    # Coordinate system differs fron acc to acc
    if iAcc<0:
        sys.exit('Acc index should not be negative!')
    elif iAcc==5 or iAcc==6:
        # Earth Albedo or Antenna thrust
        cLab=['A','C','R']
    elif iAcc==3:
        # SRP
        cLab=['D','Y','B']
    else:
        cLab=['X','Y','Z']

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nAcc1,cSat1,Acc1=GetForceAcc2(fList1,cSat0)
    nAcc2,cSat2,Acc2=GetForceAcc2(fList2,cSat0)
    # Get the common satellite set
    cSat=[]
    for i in range(len(cSat1)):
        if cSat1[i] not in cSat2:
            continue
        cSat.append(cSat1[i])
    nSat=len(cSat); cSat.sort()

    if iTyp==1:
        fig,axs=plt.subplots(nSat,1,sharex='col',squeeze=False,figsize=( 8,nSat*3))
    else:
        fig,axs=plt.subplots(nSat,3,sharex='col',squeeze=False,figsize=(20,nSat*3))
    # format of the y-axis tick label
    formattery=mpl.ticker.StrMethodFormatter('{x:6.2f}')

    for i in range(nSat):
        j=cSat1.index(cSat[i]); k=cSat2.index(cSat[i])
        # Cal the acc diff
        AccDiff=[[],[],[],[]]
        for m in range(len(Acc1[j*(nAcc1*3+1)])):
            for n in range(len(Acc2[k*(nAcc2*3+1)])):
                if (Acc2[k*(nAcc2*3+1)][n]-Acc1[j*(nAcc1*3+1)][m])*86400 < -1:
                    continue
                elif (Acc2[k*(nAcc2*3+1)][n]-Acc1[j*(nAcc1*3+1)][m])*86400 > 1:
                    break
                else:
                    AccDiff[0].append(Acc1[j*(nAcc1*3+1)][m])
                    # m/s**2 -> nm/s**2
                    AccDiff[1].append((Acc1[j*(nAcc1*3+1)+iAcc*3+1][m]-Acc2[k*(nAcc2*3+1)+iAcc*3+1][n])*1e9)
                    AccDiff[2].append((Acc1[j*(nAcc1*3+1)+iAcc*3+2][m]-Acc2[k*(nAcc2*3+1)+iAcc*3+2][n])*1e9)
                    AccDiff[3].append((Acc1[j*(nAcc1*3+1)+iAcc*3+3][m]-Acc2[k*(nAcc2*3+1)+iAcc*3+3][n])*1e9)
        if iTyp==1:
            # All three components are plotted within one axis
            axs[i,0].set_ylabel(r'Diff acc [$\mathregular{nm}/\mathregular{s}^\mathregular{2}$]',
                                fontname='Arial',fontsize=16)
            axs[i,0].ticklabel_format(axis='y',useOffset=False,useMathText=True)
            axs[i,0].grid(b=True,which='both',axis='y',color='darkgray',linestyle='--',linewidth=0.8)
            for tl in axs[i,0].get_yticklabels():
                tl.set_fontname('Arial'); tl.set_fontsize(14)

            axs[i,0].plot(AccDiff[0],AccDiff[1],'r',lw=3,label=cLab[0])
            axs[i,0].plot(AccDiff[0],AccDiff[2],'g',lw=3,label=cLab[1])
            axs[i,0].plot(AccDiff[0],AccDiff[3],'b',lw=3,label=cLab[2])

            axs[i,0].yaxis.set_major_formatter(formattery)
            axs[i,0].legend(ncol=3,loc='upper center',bbox_to_anchor=(0.5,1.0),
                            prop={'family':'Arial','size':14})
            axs[i,0].text(0.01,0.99,cSat[i],transform=axs[i,0].transAxes,ha='left',va='top',
                        family='Arial',size=16,weight='bold')
        else:
            # plotted in three axises
            axs[i,0].set_ylabel(r'Diff acc [$\mathregular{nm}/\mathregular{s}^\mathregular{2}$]',
                                fontname='Arial',fontsize=16)
            for k in range(3):
                axs[i,k].ticklabel_format(axis='y',useOffset=False,useMathText=True)
                axs[i,k].grid(b=True,which='major',axis='y',color='darkgray',linestyle='--',linewidth=0.8)
                for tl in axs[i,k].get_yticklabels():
                    tl.set_fontname('Arial'); tl.set_fontsize(14)
                axs[i,k].plot(AccDiff[0],AccDiff[1+k],lw=3)
                axs[i,k].yaxis.set_major_formatter(formattery)
                axs[i,k].text(0.01,0.99,cSat[i]+'-'+cLab[k],transform=axs[i,k].transAxes,
                              ha='left',va='top',family='Arial',size=16,weight='bold')
    axs[i,0].set_xlabel(r'Modified Julian Day',fontname='Arial',fontsize=16)
    axs[i,0].ticklabel_format(axis='x',useOffset=False,useMathText=True)
    for tl in axs[i,0].get_xticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)

    if iTyp!=1:
        for k in range(2):
            axs[i,1+k].set_xlabel(r'Modified Julian Day',fontname='Arial',fontsize=16)
            axs[i,1+k].ticklabel_format(axis='x',useOffset=False,useMathText=True)
            for tl in axs[i,1+k].get_xticklabels():
                tl.set_fontname('Arial'); tl.set_fontsize(14)

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    strTmp=OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotAntThrustAcc(OutFilePrefix,OutFileSuffix):
    '''
    Plot the acc of antenna thrust for BDS-3 based on a simplified model
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    # The correspoinding relation between PRN and SVN may change as time goes
    cPRN=['C19','C20','C21','C22','C23','C24','C25','C26','C27','C28',
          'C29','C30','C32','C33','C34','C35','C36','C37']
    cSVN=['C201','C202','C206','C205','C209','C210','C212','C211','C203','C204',
          'C207','C208','C213','C214','C216','C215','C218','C219']
    # Transmitt power in Watt
    rPow=[310,310,310,310,310,310,280,280,280,280,
          280,280,310,310,280,280,310,310]
    # Satellite mass in kg
    rMas=[943.0,942.0,942.0,941.0,945.0,946.0,1043.3,1041.8,1018.0,1014.4,
          1010.4,1008.6,1007.0,1007.0,1046.6,1045.0,1061.0,1061.0]
    # Mean motion of satellites, in rad/s
    rMea=[0.00013543,0.00013544,0.00013543,0.00013542,0.00013544,0.00013542,0.00013544,0.00013542,0.00013544,0.00013544,
          0.00013542,0.00013543,0.00013543,0.00013543,0.00013543,0.00013544,0.00013544,0.00013542]

    nSat=len(cPRN)
    # Accelerations induced by antenna thrust in nm/s*s
    Acc=np.zeros(nSat)
    # Radius change induced by antenna thrust in cm, based on a simplified model
    Rad=np.zeros(nSat)
    for i in range(nSat):
        Acc[i]=rPow[i]/299792458/rMas[i]*1e9
        Rad[i]=-1/(3*rMea[i]*rMea[i])*Acc[i]*1e-9*1e2

    x=np.arange(nSat)
    fig,axs=plt.subplots(1,1,squeeze=False,figsize=(nSat*0.6,4))
    axs[0,0].set_xlim(left=-1,right=nSat)

    #the width of the bars
    w=1/(2+1)

    # acceleration
    axs[0,0].bar(x-w,Acc,w,align='edge',color='darkred',label='Acc')
    axs[0,0].set_ylim(bottom=0,top=2)
    axs[0,0].set_yticks([0.5,1.0,1.5],minor=False)
    axs[0,0].set_ylabel(r'Antenna thrust acc [$\mathregular{nm}/\mathregular{s}^\mathregular{2}$]',
                        fontname='Arial',fontsize=16,color='darkred')
    for tl in axs[0,0].get_yticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14); tl.set_color('darkred')

    # Radius change
    axe=axs[0,0].twinx()
    axe.invert_yaxis()
    axe.set_ylim(bottom=0,top=-3)
    axe.set_yticks([-0.5,-1.0,-1.5,-2.0,-2.5],minor=False)
    axe.set_ylabel(r'Orbital radius change [cm]',fontname='Arial',fontsize=16,color='navy')
    for tl in axe.get_yticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14); tl.set_color('navy')
    axe.bar(x  ,Rad,w,align='edge',color='navy',label='Rad')

    axs[0,0].set_xlabel('Satellite PRNs',fontname='Arial',fontsize=16)
    axs[0,0].set_xticks(x)
    axs[0,0].set_xticklabels(cPRN,fontdict={'fontsize':14,'fontname':'Arial'})

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    strTmp=OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)


def PlotSRP1(fList,OutFilePrefix,OutFileSuffix):
    '''
    Plot SRP acc
    '''
    cSat=[]; Acc=[]

    for i in range(len(fList)):
        with open(fList[i],mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:3] != 'SRP':
                    continue
                cWords=cLine.split()
                if cWords[1]+'-'+cWords[2] not in cSat:
                    #Satellite PRN-SVN
                    cSat.append(cWords[1]+'-'+cWords[2])
                    Acc.append([])
                    for j in range(10):
                        Acc[len(cSat)-1].append([])
                iSat=cSat.index(cWords[1]+'-'+cWords[2])
                #Epoch
                Acc[iSat][0].append(float(cWords[5]))
                #Acc in CRS
                Acc[iSat][1].append(float(cWords[6]))
                Acc[iSat][2].append(float(cWords[7]))
                Acc[iSat][3].append(float(cWords[8]))
                #Acc in ACR
                Acc[iSat][4].append(float(cWords[9]))
                Acc[iSat][5].append(float(cWords[10]))
                Acc[iSat][6].append(float(cWords[11]))
                #Acc in DYB
                Acc[iSat][7].append(float(cWords[12]))
                Acc[iSat][8].append(float(cWords[13]))
                Acc[iSat][9].append(float(cWords[14]))
    cSat.sort(); nSat=len(cSat)

    fig,axs=plt.subplots(nSat,3,sharex='col',sharey='row',squeeze=False,figsize=(12,3*nSat))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.1)

    for i in range(nSat):
        axs[i,0].set_ylabel(r'[$nm/s^2$]')
        for j in range(3):
            if j==0:
                cLabel=['X','Y','Z']
            elif j==1:
                cLabel=['A','C','R']
            else:
                cLabel=['D','Y','B']
            axs[i,j].text(0.01,0.99,cSat[i],color='darkred',weight='bold',family='monospace',size=18,
                          transform=axs[i,j].transAxes,ha='left',va='top')
            for k in range(3):
                axs[i,j].plot(Acc[i][0],Acc[i][1+j*3+k],ls='',marker='.',ms=4,label=cLabel[k])
            axs[i,j].legend(loc='upper right')

    strTmp=OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)


def PlotEcl(fList,cSat0,OutFilePrefix,OutFileSuffix):
    '''
    Plot the eclipse info for specified satellites based on ecl-file output
    during oi
    '''

    cSat=[]; Ecl=[]
    for i in range(len(fList)):
        with open(fList[i],mode='rt') as fOb:
            for cLine in fOb:
                cWords=cLine.split()
                if cSat0[0] != 'ALL' and cWords[0] not in cSat0:
                    continue
                if cWords[0] not in cSat:
                    cSat.append(cWords[0])
                    for j in range(6):
                        Ecl.append([])
                j=cSat.index(cWords[0])
                rMJD=float(cWords[3])+float(cWords[4])/86400
                Ecl[j*6  ].append(rMJD)
                # integration direction, -1/+1
                Ecl[j*6+1].append(int(cWords[6]))
                # Earth eclipsing type, 0~3 -> 3~6
                Ecl[j*6+2].append(int(cWords[7])+3)
                # Moon eclipsing type, 0~3 -> 8~11
                Ecl[j*6+3].append(int(cWords[8])+8)
                # Whether 3-body eclipsing occurring, 0~1 -> 13~14
                if cWords[9]=='F':
                    Ecl[j*6+4].append(0+13)
                else:
                    Ecl[j*6+4].append(1+13)
                # eclipsing factor, 0~1 -> 16~17
                Ecl[j*6+5].append(float(cWords[10])+16)
    nSat=len(cSat); cSat1=cSat.copy(); cSat1.sort()

    fig,axs=plt.subplots(nSat,1,sharex='col',squeeze=False,figsize=(8,nSat*4))
    # formatterx=mpl.ticker.StrMethodFormatter('{x:7.1f}')

    for i in range(nSat):
        axs[i,0].set_ylim(bottom=-2,top=18)
        axs[i,0].set_yticks([-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],minor=False)
        axs[i,0].set_yticklabels(['-1','','1','', '0','1','2','3','','0','1','2','3','',
                                  '0','1','','0','1'])
        for tl in axs[i,0].get_yticklabels():
            tl.set_fontname('Arial'); tl.set_fontsize(14)
        # integration direction, -1/+1
        axs[i,0].axhline(y=-1,color='darkgray',ls='-',lw=0.8,alpha=0.5)
        axs[i,0].axhline(y= 1,color='darkgray',ls='-',lw=0.8,alpha=0.5)
        # Earth eclipsing type, 0~3 -> 3~6
        axs[i,0].axhline(y= 3,color='darkgray',ls='-',lw=0.8,alpha=0.5)
        axs[i,0].axhline(y= 4,color='darkgray',ls='--',lw=0.8,alpha=0.5)
        axs[i,0].axhline(y= 5,color='darkgray',ls='--',lw=0.8,alpha=0.5)
        axs[i,0].axhline(y= 6,color='darkgray',ls='-',lw=0.8,alpha=0.5)
        # Moon eclipsing type, 0~3 -> 8~11
        axs[i,0].axhline(y= 8,color='darkgray',ls='-',lw=0.8,alpha=0.5)
        axs[i,0].axhline(y= 9,color='darkgray',ls='--',lw=0.8,alpha=0.5)
        axs[i,0].axhline(y=10,color='darkgray',ls='--',lw=0.8,alpha=0.5)
        axs[i,0].axhline(y=11,color='darkgray',ls='-',lw=0.8,alpha=0.5)
        # Whether 3-body eclipsing occurring, 0~1 -> 13~14
        axs[i,0].axhline(y=13,color='darkgray',ls='-',lw=0.8,alpha=0.5)
        axs[i,0].axhline(y=14,color='darkgray',ls='-',lw=0.8,alpha=0.5)
        # eclipsing factor, 0~1 -> 16~17
        axs[i,0].axhline(y=16,color='darkgray',ls='-',lw=0.8,alpha=0.5)

        axs[i,0].set_ylabel(cSat[i],fontname='Arial',fontsize=16)

        j=cSat.index(cSat1[i])
        # integration direction
        axs[i,0].plot(Ecl[j*6],Ecl[j*6+1],'r.',ms=1,label='iDir')
        # Earth eclipsing type
        axs[i,0].plot(Ecl[j*6],Ecl[j*6+2],'gv',ms=1,label='EEcl')
        # Moon eclipsing type
        axs[i,0].plot(Ecl[j*6],Ecl[j*6+3],'mv',ms=1,label='MEcl')
        # Whether 3-body eclipsing occurring
        axs[i,0].plot(Ecl[j*6],Ecl[j*6+4],'bo',ms=1,label='3Ecl')
        # eclipsing factor
        axs[i,0].plot(Ecl[j*6],Ecl[j*6+5],'c.',ms=1,label='EclF')

        axs[i,0].legend(ncol=1,loc='upper left',bbox_to_anchor=(1.0,1.0),
                        prop={'family':'Arial','size':14})
        # axs[i,0].text(0.01,0.99,cSat[j],transform=axs[i,0].transAxes,ha='left',va='top',
        #               family='Arial',size=16)
    axs[i,0].set_xlabel(r'Modified Julian Day',fontname='Arial',fontsize=16)
    axs[i,0].ticklabel_format(axis='x',useOffset=False,useMathText=True)
    for tl in axs[i,0].get_xticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)

    strTmp=OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    import argparse

    cWhere='Local'
    # cWhere='Cluster'
    if cWhere=='Local':
        # Local Windows
        cWrkPre0=r'Y:/'

        cDskPre0=r'Z:/'
    else:
        # GFZ section cluster
        cWrkPre0=r'/wrk/hanbing/'

        cDskPre0=r'/dsk/hanbing/'
    print('Run On '+cWhere)

    InFilePrefix=os.path.join(cWrkPre0,r'PRO_2019001_2020366_WORK/I1_d/WORK20193??/')
    # InFilePrefix=r'D:/Code/PROJECT/WORK2019335_ERROR/'
    fList=glob.glob(InFilePrefix+'oi_Debug_20193??')
    # OutFilePrefix=r'Y:\PRO_2019001_2020366_WORK\I2_G_PWL\WORK2019335/'
    # OutFilePrefix=r'D:/Code/PROJECT/WORK2019335_ERROR/'
    OutFilePrefix=os.path.join(cDskPre0,r'PRO_2019001_2020366/I1_b/ORB/')

    # OutFileSuffix='Sunlight'
    # PlotSunLight0(fList,['ALL'],OutFilePrefix,OutFileSuffix)
    # PlotForceAcc0(fList,['ALL'],OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='AccAlb_1'
    # PlotForceAcc1(fList,['C22','C23','C26','C30'],5,1,2,OutFilePrefix,OutFileSuffix)

    # InFilePrefix2=r'Y:\PRO_2019001_2020366_WORK\I1_c\WORK2019335/'
    # fList2=glob.glob(InFilePrefix2+'oi_Debug_2019???')

    # OutFileSuffix='AccSRP_2019335_diff'
    # PlotAccDiff0(fList,fList2,['ALL'],3,3,OutFilePrefix,OutFileSuffix)

    OutFileSuffix='AccAnt_Analysis'
    PlotAntThrustAcc(OutFilePrefix,OutFileSuffix)

    # OutFileSuffix='AccPSV2.pdf'
    # PlotForceAcc2(fList,['ALL'],[7],OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='BWC.pdf'
    # PlotSunLight1(fList,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='AccSRPDiff'
    # PlotSunLight2(fList,['ALL'],OutFilePrefix,OutFileSuffix)


    # InFilePrefix=r'D:/Code/PROJECT/WORK_ANG/'
    # fList=glob.glob(InFilePrefix+'GeoAng_*')
    # OutFilePrefix=r'D:/Code/PROJECT/WORK_ANG/'

    # OutFileSuffix='AccSRP.png'
    # PlotSRP1(fList,OutFilePrefix,OutFileSuffix)

    # InFilePrefix=r'D:/Code/PROJECT/WORK2019335_ERROR/'
    # fList=glob.glob(InFilePrefix+'ecl_2019335_???')
    # OutFilePrefix=r'D:/Code/PROJECT/WORK2019335_ERROR/'
    # OutFileSuffix='EclInfo.pdf'
    # PlotEcl(fList,['ALL'],OutFilePrefix,OutFileSuffix)