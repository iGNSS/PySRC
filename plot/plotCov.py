#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
'''
__author__ = 'hanbing'

# Standard library imports
import os,math
import sys
import os.path
import glob

# Related third party imports
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


def PlotStateCov(fList,cSat0,OutFilePrefix,OutFileSuffix):
    '''
    Plot formal variance of orbit state vector for specified satellites
    '''

    cSat=[]
    nFile=len(fList)
    for i in range(nFile):
        with open(fList[i],mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine)<5:
                    continue
                cWords=cLine.split()
                if cSat0[0]!='ALL' and cWords[3] not in cSat0:
                    continue
                if cWords[3] not in cSat:
                    cSat.append(cWords[3])
    cSat.sort()
    nSat=len(cSat)

    Cov=[]
    for i in range(nSat):
        Cov.append([])
        for j in range(4):
            Cov[i].append([])
    for i in range(nFile):
        with open(fList[i],mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine)<5:
                    continue
                cWords=cLine.split()
                if cWords[3] not in cSat:
                    continue
                j=cSat.index(cWords[3])
                rEpo=float(cWords[0])+float(cWords[1])/86400
                Cov[j][0].append(rEpo)
                #Along-track
                Cov[j][1].append(float(cWords[4]))
                #Cross-track
                Cov[j][2].append(float(cWords[6]))
                #Radial
                Cov[j][3].append(float(cWords[9]))

    fig,axs=plt.subplots(nSat,3,sharex='col',squeeze=False,figsize=(3*5,nSat*3))
    fig.subplots_adjust(hspace=0.1)
    # Formatting tick label
    formatterx=mpl.ticker.StrMethodFormatter('{x:8.2f}')
    for i in range(nSat):
        axs[i,0].plot(Cov[i][0],Cov[i][1],label='Along')
        axs[i,0].legend(loc='upper center')
        axs[i,0].text(0.05,0.95,cSat[i],transform=axs[i,0].transAxes,ha='left',va='top')
        axs[i,1].plot(Cov[i][0],Cov[i][2],label='Cross')
        axs[i,1].legend(loc='upper center')
        axs[i,2].plot(Cov[i][0],Cov[i][3],label='Radial')
        axs[i,2].legend(loc='upper center')
    axs[i,0].set_xlabel('Modified Julian Day')
    axs[i,1].set_xlabel('Modified Julian Day')
    axs[i,2].set_xlabel('Modified Julian Day')
    # axs[i,0].xaxis.set_major_formatter(formatterx)

    strTmp=OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotStateCovDif(fList1,fList2,cSat0,OutFilePrefix,OutFileSuffix):
    '''
    Plot the decrease percentage of orbit state vector formal variance in
    fList2 with respect to that in fList1 for specified satellites
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # Satellite list from the first file set
    cSat1=[]; nFile=len(fList1)
    for i in range(nFile):
        with open(fList1[i],mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine)<5:
                    continue
                cWords=cLine.split()
                if cSat0[0]!='ALL' and cWords[3] not in cSat0:
                    continue
                if cWords[3] not in cSat1:
                    cSat1.append(cWords[3])
    # Common satellite list from both file sets
    cSat=[]; nFile=len(fList2)
    for i in range(nFile):
        with open(fList2[i],mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine)<5:
                    continue
                cWords=cLine.split()
                if (cWords[3] in cSat1) and (cWords[3] not in cSat):
                    cSat.append(cWords[3])
    cSat.sort(); nSat=len(cSat)

    Cov1=[]
    for i in range(nSat):
        Cov1.append([])
        for j in range(4):
            Cov1[i].append([])
    nFile=len(fList1)
    for i in range(nFile):
        with open(fList1[i],mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine)<5:
                    continue
                cWords=cLine.split()
                if cWords[3] not in cSat:
                    continue
                j=cSat.index(cWords[3])
                rEpo=float(cWords[0])+float(cWords[1])/86400
                Cov1[j][0].append(rEpo)
                #Along-track, formal standard error
                Cov1[j][1].append(math.sqrt(float(cWords[4])))
                #Cross-track, formal standard error
                Cov1[j][2].append(math.sqrt(float(cWords[6])))
                #Radial, formal standard error
                Cov1[j][3].append(math.sqrt(float(cWords[9])))

    Cov2=[]
    for i in range(nSat):
        Cov2.append([])
        for j in range(4):
            Cov2[i].append([])
    nFile=len(fList2)
    for i in range(nFile):
        with open(fList2[i],mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine)<5:
                    continue
                cWords=cLine.split()
                if cWords[3] not in cSat:
                    continue
                j=cSat.index(cWords[3])
                rEpo=float(cWords[0])+float(cWords[1])/86400
                Cov2[j][0].append(rEpo)
                #Along-track, formal standard error
                Cov2[j][1].append(math.sqrt(float(cWords[4])))
                #Cross-track, formal standard error
                Cov2[j][2].append(math.sqrt(float(cWords[6])))
                #Radial, formal standard error
                Cov2[j][3].append(math.sqrt(float(cWords[9])))

    fig,axs=plt.subplots(nSat,1,sharex='col',squeeze=False,figsize=(8,nSat*3))
    # fig.subplots_adjust(hspace=0.1)
    # Formatting tick label
    formatterx=mpl.ticker.StrMethodFormatter('{x:7.1f}')

    for i in range(nSat):
        nEpo1=len(Cov1[i][0]); nEpo2=len(Cov2[i][0])
        Dif=[[],[],[],[]]
        for j in range(nEpo1):
            for k in range(nEpo2):
                if abs(Cov1[i][0][j]-Cov2[i][0][k])*86400<1.0:
                    Dif[0].append(Cov1[i][0][j])
                    # Formal standard error decrease percentage
                    xtmp=(Cov1[i][1][j]-Cov2[i][1][k])/Cov1[i][1][j]*1e2
                    Dif[1].append(xtmp)
                    xtmp=(Cov1[i][2][j]-Cov2[i][2][k])/Cov1[i][2][j]*1e2
                    Dif[2].append(xtmp)
                    xtmp=(Cov1[i][3][j]-Cov2[i][3][k])/Cov1[i][3][j]*1e2
                    Dif[3].append(xtmp)
                    break
        axs[i,0].set_ylabel('Formal error decrease [%]',fontname='Arial',fontsize=16)
        for tl in axs[i,0].get_yticklabels():
            tl.set_fontname('Arial'); tl.set_fontsize(14)
        axs[i,0].plot(Dif[0],Dif[1],color='r',label='Along')
        axs[i,0].plot(Dif[0],Dif[2],color='g',label='Cross')
        axs[i,0].plot(Dif[0],Dif[3],color='b',label='Radial')
        # axs[i,0].axhline(color='darkgray',linestyle='dashed',alpha=0.5)
        axs[i,0].text(0.02,0.98,cSat[i],transform=axs[i,0].transAxes,ha='left',va='top',
                      color='darkred',weight='bold',family='Arial',size=16)
        axs[i,0].legend(ncol=3,loc='upper center',framealpha=0.6,bbox_to_anchor=(0.5,1.0),
                        prop={'family':'Arial','size':14})
    axs[i,0].xaxis.set_major_formatter(formatterx)
    axs[i,0].set_xlabel('Modified Julian Day',fontname='Arial',fontsize=16)
    for tl in axs[i,0].get_xticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    strTmp=OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    import argparse

    # fList=[]
    # fList.append(r'D:/Code/PROJECT/WORK2019001/StateCov')
    # cSat=['ALL']
    # OutFilePrefix='D:/Code/PROJECT/WORK2019001/'
    # OutFileSuffix='StateCov.pdf'
    # PlotStateCov(fList,cSat,OutFilePrefix,OutFileSuffix)

    InFilePrefix=r'Y:/PRO_2019001_2020366_WORK/I1/WORK2019???/'
    fList1=glob.glob(InFilePrefix+'StateCov_2019???')
    # cSat=['ALL']
    # OutFilePrefix=r'Y:/PRO_2019001_2019007/Sum/'
    # OutFileSuffix='StateCov_S1.png'
    # PlotStateCov(fList1,cSat,OutFilePrefix,OutFileSuffix)

    InFilePrefix=r'Y:/PRO_2019001_2020366_WORK/I6/WORK2019???/'
    fList2=glob.glob(InFilePrefix+'StateCov_2019???')
    cSat=['ALL']
    # cSat=['C28']
    OutFilePrefix=r'Z:/PRO_2019001_2020366/I6/ORB/'
    OutFileSuffix='StateCov_I1'
    PlotStateCovDif(fList1,fList2,cSat,OutFilePrefix,OutFileSuffix)
    # PlotStateCov(fList2,cSat,OutFilePrefix,OutFileSuffix)

    # InFilePrefix=r'Y:/PRO_2019001_2019007_WORK/S4/WORK201900?/'
    # fList1=glob.glob(InFilePrefix+'StateCov_201900?')
    # InFilePrefix=r'Y:/PRO_2019001_2019007_WORK/S5/WORK201900?/'
    # fList2=glob.glob(InFilePrefix+'StateCov_201900?')
    # cSat=['C19','C24','C28','C34']
    # OutFilePrefix=r'Y:/PRO_2019001_2019007/Sum/'
    # OutFileSuffix='StateCov_S5_S4_C19C24C28C34.png'
    # PlotStateCovDif(fList1,fList2,cSat,OutFilePrefix,OutFileSuffix)
