#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Make some plots related to EOP
'''
__author__ = 'hanbing'

# Standard library imports
import subprocess
import os
import sys
import os.path
import glob
import datetime,math

# Related third party imports
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

from astropy.stats import sigma_clipped_stats

# Local application/library specific imports
from PySRC.miscellaneous import GNSSTime


def GetIERSC04(fC04,rMJD1,rMJD2):
    '''
    Read the IERS C04 series for period [rMJD1, rMJD2]

    UT1-TAI
    '''

    EOP=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    with open(fC04,mode='rt') as fOb:
        nLine=0
        for cLine in fOb:
            nLine=nLine+1
            if nLine < 15:
                continue
            cWords=cLine.split()
            rMJD=float(cWords[3])
            if rMJD < rMJD1:
                continue
            elif rMJD > rMJD2:
                break
            EOP[0].append(rMJD)
            # xpole, ypole, arcsec
            EOP[ 1].append(float(cWords[ 4])); EOP[ 2].append(float(cWords[ 5]))
            # UT1-UTC -> UT1-TAI, LOD, sec
            EOP[ 3].append(float(cWords[ 6])-GNSSTime.UTC2TAI(rMJD)); EOP[ 4].append(float(cWords[ 7]))
            # dX, dY, arcsec
            EOP[ 5].append(float(cWords[ 8])); EOP[ 6].append(float(cWords[ 9]))
            # xpole, ypole error, arcsec
            EOP[ 7].append(float(cWords[10])); EOP[ 8].append(float(cWords[11]))
            # UT1-UTC, LOD error, sec
            EOP[ 9].append(float(cWords[12])); EOP[10].append(float(cWords[13]))
            # dX, dY error, aresec
            EOP[11].append(float(cWords[14])); EOP[12].append(float(cWords[15]))
            # For additionally calculated terms
            EOP[13].append(np.nan); EOP[14].append(np.nan)
            EOP[15].append(np.nan); EOP[16].append(np.nan)
            EOP[17].append(np.nan); EOP[18].append(np.nan)
            nRec=len(EOP[0])
            if nRec==1:
                continue
            # Start to calculate from the second epoch
            dt=EOP[0][nRec-1]-EOP[0][nRec-2]
            # dX rate
            EOP[13][nRec-2]=(EOP[5][nRec-1]-EOP[5][nRec-2])/dt
            # dX rate sigma
            EOP[14][nRec-2]=math.sqrt(EOP[11][nRec-1]**2+EOP[11][nRec-2]**2)/dt
            # dY rate
            EOP[15][nRec-2]=(EOP[6][nRec-1]-EOP[6][nRec-2])/dt
            # dY rate sigma
            EOP[16][nRec-2]=math.sqrt(EOP[12][nRec-1]**2+EOP[12][nRec-2]**2)/dt
            # diff btw calculated LOD from UT1 and the reported LOD
            EOP[17][nRec-2]=(EOP[3][nRec-1]-EOP[3][nRec-2])/dt + EOP[4][nRec-2]

    return np.array(EOP)

def GetFinals2000(fFinal,rMJD1,rMJD2):
    '''
    Read the IERS finals2000A series for period [rMJD1, rMJD2]

    UT1-UTC
    '''

    EOP=[[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
    with open(fFinal,mode='rt') as fOb:
        for cLine in fOb:
            if len(cLine) < 20:
                continue
            rMJD=float(cLine[7:15])
            if rMJD < rMJD1:
                continue
            elif rMJD > rMJD2:
                break
            EOP[0].append(rMJD)
            # xpole, ypole, arcsec
            EOP[ 1].append(float(cLine[ 18: 27])); EOP[ 2].append(float(cLine[ 37: 46]))
            # xpole, ypole error, arcsec
            EOP[ 7].append(float(cLine[ 27: 36])); EOP[ 8].append(float(cLine[ 46: 55]))
            # UT1-UTC, LOD, sec
            EOP[ 3].append(float(cLine[ 58: 68])); EOP[ 4].append(float(cLine[ 79: 86])*1.e-3)
            # UT1-UTC, LOD error, sec
            EOP[ 9].append(float(cLine[ 68: 78])); EOP[10].append(float(cLine[ 86: 93])*1.e-3)
            # dX, dY, arcsec
            EOP[ 5].append(float(cLine[ 97:106])*1.e-3); EOP[ 6].append(float(cLine[116:125])*1.e-3)
            EOP[11].append(float(cLine[106:115])*1.e-3); EOP[12].append(float(cLine[125:134])*1.e-3)
            # For additionally calculated terms
            EOP[13].append(np.nan); EOP[14].append(np.nan)
            EOP[15].append(np.nan); EOP[16].append(np.nan)
            EOP[17].append(np.nan); EOP[18].append(np.nan)
            nRec=len(EOP[0])
            if nRec==1:
                continue
            dt=EOP[0][nRec-1]-EOP[0][nRec-2]
            # dX rate
            EOP[13][nRec-2]=(EOP[5][nRec-1]-EOP[5][nRec-2])/dt
            # dX rate sigma
            EOP[14][nRec-2]=math.sqrt(EOP[11][nRec-1]**2+EOP[11][nRec-2]**2)/dt
            # dY rate
            EOP[15][nRec-2]=(EOP[6][nRec-1]-EOP[6][nRec-2])/dt
            # dY rate sigma
            EOP[16][nRec-2]=math.sqrt(EOP[12][nRec-1]**2+EOP[12][nRec-2]**2)/dt
            # diff btw calculated LOD from UT1 and the reported LOD
            EOP[17][nRec-2]=(EOP[3][nRec-1]-EOP[3][nRec-2])/dt + EOP[4][nRec-2]
    return np.array(EOP)

def GetPHBERP(fList,rMJD1,rMJD2):
    '''
    Read ERP file (in IGS format) from my solutions

    EOP --- rMJD, xp, xp_sig, yp, yp_sig, ut1, ut1_sig
            dxp, dxp_sig, dyp, dyp_sig, LOD, LOD_sig
    '''

    EOP=[[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for i in range(len(fList)):
        with open(fList[i],mode='rt') as fOb:
            nLine=0
            for cLine in fOb:
                nLine=nLine+1
                if (nLine < 6) or (len(cLine) < 5):
                    continue
                # From 6-th line on
                cWords=cLine.split()
                rMJD=float(cWords[ 0])
                if (rMJD-rMJD1) < -0.1:
                    continue
                if (rMJD-rMJD2) >  0.1:
                    continue
                EOP[ 0].append(rMJD)
                # xpole, 1e-6 arcsec
                EOP[ 1].append(float(cWords[ 1]))
                # xpole error, 1e-6 arcsec
                EOP[ 2].append(float(cWords[ 5]))
                # ypole, 1e-6 arcsec
                EOP[ 3].append(float(cWords[ 2]))
                # ypole error, 1e-6 arcsec
                EOP[ 4].append(float(cWords[ 6]))
                # UT1-TAI, 1e-7 sec
                EOP[ 5].append(float(cWords[ 3]))
                # UT1-TAI error, 1e-7 sec
                EOP[ 6].append(float(cWords[ 7]))
                # xpole rate, 1e-6 arcsec/d
                EOP[ 7].append(float(cWords[12]))
                # xpole rate error, 1e-6 arcsec/d
                EOP[ 8].append(float(cWords[14]))
                # ypole rate, 1e-6 arcsec/d
                EOP[ 9].append(float(cWords[13]))
                # ypole rate error, 1e-6 arcsec/d
                EOP[10].append(float(cWords[15]))
                # LOD, 1e-7 sec/d
                EOP[11].append(float(cWords[ 4]))
                # LOD error, 1e-7 sec/d
                EOP[12].append(float(cWords[ 8]))
    nRec=len(EOP[0]); X=EOP.copy()
    # sort the EOP series
    ind=np.argsort(X[0])
    EOP=np.zeros((13,nRec))
    for i in range(nRec):
        for j in range(13):
            EOP[j,i]=X[j][ind[i]]
    return EOP

def GetIGSERP(fList,rMJD1,rMJD2):
    '''
    Read ERP file (in IGS format) from IGS products

    EOP --- rMJD, xp, xp_sig, yp, yp_sig, ut1, ut1_sig
            dxp, dxp_sig, dyp, dyp_sig, LOD, LOD_sig
    '''

    EOP=[[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for i in range(len(fList)):
        with open(fList[i],mode='rt') as fOb:
            nLine=0
            for cLine in fOb:
                nLine=nLine+1
                if (nLine < 5) or (len(cLine) < 5):
                    continue
                # From 5-th line on
                cWords=cLine.split()
                rMJD=float(cWords[ 0])
                if (rMJD-rMJD1) < -0.1:
                    continue
                if (rMJD-rMJD2) >  0.1:
                    continue
                EOP[ 0].append(rMJD)
                # xpole, 1e-6 arcsec
                EOP[ 1].append(float(cWords[ 1]))
                # xpole error, 1e-6 arcsec
                EOP[ 2].append(float(cWords[ 5]))
                # ypole, 1e-6 arcsec
                EOP[ 3].append(float(cWords[ 2]))
                # ypole error, 1e-6 arcsec
                EOP[ 4].append(float(cWords[ 6]))
                # UT1-UTC -> UT1-TAI, 1e-7 sec
                EOP[ 5].append(float(cWords[ 3])-GNSSTime.UTC2TAI(rMJD)*1e7)
                # UT1-UTC/TAI error, 1e-7 sec
                EOP[ 6].append(float(cWords[ 7]))
                # xpole rate, 1e-6 arcsec/d
                EOP[ 7].append(float(cWords[12]))
                # xpole rate error, 1e-6 arcsec/d
                EOP[ 8].append(float(cWords[14]))
                # ypole rate, 1e-6 arcsec/d
                EOP[ 9].append(float(cWords[13]))
                # ypole rate error, 1e-6 arcsec/d
                EOP[10].append(float(cWords[15]))
                # LOD, 1e-7 sec/d
                EOP[11].append(float(cWords[ 4]))
                # LOD error, 1e-7 sec/d
                EOP[12].append(float(cWords[ 8]))
    nRec=len(EOP[0]); X=EOP.copy()
    # sort the EOP series
    ind=np.argsort(X[0])
    EOP=np.zeros((13,nRec))
    for i in range(nRec):
        for j in range(13):
            EOP[j,i]=X[j][ind[i]]
    return EOP

def GetGFZERP(fList,rMJD1,rMJD2):
    '''
    Read ERP file (in IGS format) from GFZ products

    EOP --- rMJD, xp, xp_sig, yp, yp_sig, ut1, ut1_sig
            dxp, dxp_sig, dyp, dyp_sig, LOD, LOD_sig
    '''

    EOP=[[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for i in range(len(fList)):
        with open(fList[i],mode='rt') as fOb:
            nLine=0
            for cLine in fOb:
                nLine=nLine+1
                if (nLine < 6) or (len(cLine) < 5):
                    continue
                # From 6-th line on
                cWords=cLine.split()
                rMJD=float(cWords[ 0])
                if (rMJD-rMJD1) < -0.1:
                    continue
                if (rMJD-rMJD2) >  0.1:
                    continue
                EOP[ 0].append(rMJD)
                # xpole, 1e-6 arcsec
                EOP[ 1].append(float(cWords[ 1]))
                # xpole error, 1e-6 arcsec
                EOP[ 2].append(float(cWords[ 5]))
                # ypole, 1e-6 arcsec
                EOP[ 3].append(float(cWords[ 2]))
                # ypole error, 1e-6 arcsec
                EOP[ 4].append(float(cWords[ 6]))
                # UT1-TAI, 1e-7 sec
                EOP[ 5].append(float(cWords[ 3]))
                # UT1-TAI error, 1e-7 sec
                EOP[ 6].append(float(cWords[ 7]))
                # xpole rate, 1e-6 arcsec/d
                EOP[ 7].append(float(cWords[12]))
                # xpole rate error, 1e-6 arcsec/d
                EOP[ 8].append(float(cWords[14]))
                # ypole rate, 1e-6 arcsec/d
                EOP[ 9].append(float(cWords[13]))
                # ypole rate error, 1e-6 arcsec/d
                EOP[10].append(float(cWords[15]))
                # LOD, 1e-7 sec/d
                EOP[11].append(float(cWords[ 4]))
                # LOD error, 1e-7 sec/d
                EOP[12].append(float(cWords[ 8]))
    nRec=len(EOP[0]); X=EOP.copy()
    # sort the EOP series
    ind=np.argsort(X[0])
    EOP=np.zeros((13,nRec))
    for i in range(nRec):
        for j in range(13):
            EOP[j,i]=X[j][ind[i]]
    return EOP

def GetCODERP(fList,rMJD1,rMJD2):
    '''
    Read ERP file (in IGS format) from CODE cod/cof products

    EOP --- rMJD, xp, xp_sig, yp, yp_sig, ut1, ut1_sig
            dxp, dxp_sig, dyp, dyp_sig, LOD, LOD_sig
    '''

    EOP=[[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for i in range(len(fList)):
        with open(fList[i],mode='rt') as fOb:
            nLine=0
            for cLine in fOb:
                nLine=nLine+1
                if (nLine < 7) or (len(cLine) < 5):
                    continue
                # From 7-th line on
                cWords=cLine.split()
                rMJD=float(cWords[ 0])
                if (rMJD-rMJD1) < -0.1:
                    continue
                if (rMJD-rMJD2) >  0.1:
                    continue
                EOP[ 0].append(rMJD)
                # xpole, 1e-6 arcsec
                EOP[ 1].append(float(cWords[ 1]))
                # xpole error, 1e-6 arcsec
                EOP[ 2].append(float(cWords[ 5]))
                # ypole, 1e-6 arcsec
                EOP[ 3].append(float(cWords[ 2]))
                # ypole error, 1e-6 arcsec
                EOP[ 4].append(float(cWords[ 6]))
                # UT1-UTC -> UT1-TAI, 1e-7 sec
                EOP[ 5].append(float(cWords[ 3])-GNSSTime.UTC2TAI(rMJD)*1e7)
                # UT1-TAI error, 1e-7 sec
                EOP[ 6].append(float(cWords[ 7]))
                # xpole rate, 1e-6 arcsec/d
                EOP[ 7].append(float(cWords[12]))
                # xpole rate error, 1e-6 arcsec/d
                EOP[ 8].append(float(cWords[14]))
                # ypole rate, 1e-6 arcsec/d
                EOP[ 9].append(float(cWords[13]))
                # ypole rate error, 1e-6 arcsec/d
                EOP[10].append(float(cWords[15]))
                # LOD, 1e-7 sec/d
                EOP[11].append(float(cWords[ 4]))
                # LOD error, 1e-7 sec/d
                EOP[12].append(float(cWords[ 8]))
    nRec=len(EOP[0]); X=EOP.copy()
    # sort the EOP series
    ind=np.argsort(X[0])
    EOP=np.zeros((13,nRec))
    for i in range(nRec):
        for j in range(13):
            EOP[j,i]=X[j][ind[i]]
    return EOP

def GetJPLERP(fList,rMJD1,rMJD2):
    '''
    Read ERP file (in IGS format) from JPL products

    EOP --- rMJD, xp, xp_sig, yp, yp_sig, ut1, ut1_sig
            dxp, dxp_sig, dyp, dyp_sig, LOD, LOD_sig
    '''

    EOP=[[],[],[],[],[],[],[],[],[],[],[],[],[]]
    for i in range(len(fList)):
        with open(fList[i],mode='rt') as fOb:
            nLine=0
            for cLine in fOb:
                nLine=nLine+1
                if (nLine < 4) or (len(cLine) < 5):
                    continue
                # From 4-th line on
                cWords=cLine.split()
                rMJD=float(cWords[ 0])
                if (rMJD-rMJD1) < -0.1:
                    continue
                if (rMJD-rMJD2) >  0.1:
                    continue
                EOP[ 0].append(rMJD)
                # xpole, 1e-6 arcsec
                EOP[ 1].append(float(cWords[ 1]))
                # xpole error, 1e-6 arcsec
                EOP[ 2].append(float(cWords[ 5]))
                # ypole, 1e-6 arcsec
                EOP[ 3].append(float(cWords[ 2]))
                # ypole error, 1e-6 arcsec
                EOP[ 4].append(float(cWords[ 6]))
                # UT1-UTC -> UT1-TAI, 1e-7 sec
                EOP[ 5].append(float(cWords[ 3])-GNSSTime.UTC2TAI(rMJD)*1e7)
                # UT1-TAI error, 1e-7 sec
                EOP[ 6].append(float(cWords[ 7]))
                # xpole rate, 1e-6 arcsec/d
                EOP[ 7].append(float(cWords[12]))
                # xpole rate error, 1e-6 arcsec/d
                EOP[ 8].append(float(cWords[14]))
                # ypole rate, 1e-6 arcsec/d
                EOP[ 9].append(float(cWords[13]))
                # ypole rate error, 1e-6 arcsec/d
                EOP[10].append(float(cWords[15]))
                # LOD, 1e-7 sec/d
                EOP[11].append(float(cWords[ 4]))
                # LOD error, 1e-7 sec/d
                EOP[12].append(float(cWords[ 8]))
    nRec=len(EOP[0]); X=EOP.copy()
    # sort the EOP series
    ind=np.argsort(X[0])
    EOP=np.zeros((13,nRec))
    for i in range(nRec):
        for j in range(13):
            EOP[j,i]=X[j][ind[i]]
    return EOP

def PlotEOP1(fEOP,cType,rMJD1,rMJD2,OutFilePrefix,OutFileSuffix):
    '''
    Plot EOP series from finals2000A or IERS C04
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    if cType=='FIN':
        # finals2000A
        EOP=GetFinals2000(fEOP,rMJD1,rMJD2)
    elif cType=='C04':
        EOP=GetIERSC04(fEOP,rMJD1,rMJD2)

    fig,axs=plt.subplots(9,1,sharex='col',sharey='row',squeeze=False,figsize=(14,6*4))

    cEOP=['XPOLE','YPOLE','UT1-UTC','LOD','dX','dY','dX rate','dY rate','dUT1+LOD']
    yLab=[r'$x_p$ [as]',r'$y_p$ [as]',r'UT1-UTC [s]',r'LOD [s]',
          r'dX [as]',r'dY [as]',r'$\.{dX}$ [as/d]',r'$\.{dY}$ [as/d]',
          r'UT1-UTC (+LOD) [s]']
    iX=[1,2,3,4,5,6,13,15,17]; iE=[7,8,9,10,11,12,14,16,18]

    for i in range(9):
        # for LOD in us, dX/dY in uas
        if i>=3:
            EOP[iX[i]]=EOP[iX[i]]*1e6
            EOP[iE[i]]=EOP[iE[i]]*1e6

        if i>=6:
            # dX/dY rate, no sigma
            axs[i,0].plot(EOP[0],EOP[iX[i]],'o--',ms=3,lw=1,label=cEOP[i])
        else:
            axs[i,0].errorbar(EOP[0],EOP[iX[i]],yerr=EOP[iE[i]],fmt='o--',ms=3,lw=1,
                            capsize=3,label=cEOP[i])
        axs[i,0].grid(b=True,which='both',axis='y',color='darkgray',linestyle='--',
                      linewidth=0.8)
        axs[i,0].set_ylabel(yLab[i],fontname='Arial',fontsize=16)
        for tl in axs[i,0].get_yticklabels():
            tl.set_fontname('Arial'); tl.set_fontsize(14)
    axs[i,0].set_xlabel('Modified Julian Day',fontname='Arial',fontsize=16)
    for tl in axs[i,0].get_xticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    strTmp=OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotEOP2(AC,fERPList,OutFilePrefix,OutFileSuffix):
    '''
    Plot the ERP sig recorded in the IGS ERP format files

          AC ---
    fERPList --- ERP files
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    if AC=='phb':
        EOP=GetPHBERP(fERPList,0,99999)
    elif AC=='gfz':
        EOP=GetGFZERP(fERPList,0,99999)
    elif AC=='cod' or AC=='cof':
        EOP=GetCODERP(fERPList,0,99999)
    elif AC=='jpl':
        EOP=GetJPLERP(fERPList,0,99999)
    elif AC=='igs':
        EOP=GetIGSERP(fERPList,0,99999)
    else:
        sys.exit('Unknown AC, '+AC)

    fig,axs=plt.subplots(3,2,sharex='col',squeeze=False,figsize=(14,8))
    # fig.subplots_adjust(hspace=0.1)

    yLab=[r'$x_p$ [$\mu$as]',r'$\.x_p$ [$\mu$as/d]',
          r'$y_p$ [$\mu$as]',r'$\.y_p$ [$\mu$as/d]',
          r'UT1-UTC [$\mu$s]',r'LOD [$\mu$s/d]']
    iEOP=[2,8,4,10,6,12]

    for i in range(3):
        for j in range(2):
            k=2*i+j
            #Cal the robusted mean, median and std
            Mea,Med,Sig=sigma_clipped_stats(np.array(EOP[iEOP[k]]),sigma=3,maxiters=5)

            axs[i,j].plot(EOP[0],EOP[iEOP[k]],'o--r',ms=3,lw=1)
            # axs[i,j].axhline(color='darkgray',linestyle='dashed',alpha=0.5)
            axs[i,j].grid(b=True,which='both',axis='y',color='darkgray',linestyle='--',
                          linewidth=0.8)
            axs[i,j].set_ylabel(yLab[k],fontname='Arial',fontsize=16)
            for tl in axs[i,j].get_yticklabels():
                tl.set_fontname('Arial'); tl.set_fontsize(14)
            axs[i,j].text(0.5,0.98,'{:>8.3f} ({:>8.3f}) +/- {:>6.3f}'.format(Mea,Med,Sig),
                          transform=axs[i,j].transAxes,ha='center',va='top',
                          fontdict={'fontsize':14,'fontname':'Arial','fontweight':'bold'})
            if i==2:
                axs[i,j].set_xlabel('Modified Julian Day',fontname='Arial',fontsize=16)
                for tl in axs[i,j].get_xticklabels():
                    tl.set_fontname('Arial'); tl.set_fontsize(14)

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    strTmp=OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotEOPDiff1(fC04,fFinal,rMJD1,rMJD2,OutFilePrefix,OutFileSuffix):
    '''
    Plot the EOP diff between IERS C04 and finnal2000A
    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    EOP1=GetIERSC04(fC04,rMJD1,rMJD2)
    EOP2=GetFinals2000(fFinal,rMJD1,rMJD2)

    X=[[],[],[],[],[],[],[]]
    for i in range(EOP1[0].size):
        for j in range(EOP2[0].size):
            if EOP2[0,j] < EOP1[0,i]-0.1:
                continue
            elif EOP2[0,j] > EOP1[0,i]+0.1:
                break
            else:
                X[0].append(EOP1[0,i])
                # diff of xpole, ypole in arcsec
                X[1].append(EOP2[1,j]-EOP1[1,i]); X[2].append(EOP2[2,j]-EOP1[2,i])
                # diff of UT1-UTC, LOD in sec
                X[3].append(EOP2[3,j]-EOP1[3,i]-GNSSTime.UTC2TAI(EOP1[0,i]))
                X[4].append(EOP2[4,j]-EOP1[4,i])
                # diff of dX, dY in arcsec
                X[5].append(EOP2[5,j]-EOP1[5,i]); X[6].append(EOP2[6,j]-EOP1[6,i])
    Diff=np.array(X)

    fig,axs=plt.subplots(3,2,sharex='col',sharey='row',squeeze=False,figsize=(14,8))
    # fig.subplots_adjust(hspace=0.1)
    yLab=[r'$x_p$ [$\mu$as]',r'$y_p$ [$\mu$as]',r'UT1-UTC [$\mu$s]',
          r'LOD [$\mu$s]',r'dX [$\mu$as]',r'dY [$\mu$as]']

    for i in range(3):
        for j in range(2):
            k=i*2+j+1
            #Cal the robusted mean, median and std
            Mea,Med,Sig=sigma_clipped_stats(np.array(Diff[k]*1e6),sigma=3,maxiters=5)

            axs[i,j].plot(Diff[0],Diff[k]*1.0e6,'.r',ms=2)
            axs[i,j].axhline(color='darkgray',linestyle='dashed',alpha=0.5)
            axs[i,j].set_ylabel(yLab[k-1],fontname='Arial',fontsize=16)
            for tl in axs[i,j].get_yticklabels():
                tl.set_fontname('Arial'); tl.set_fontsize(14)
            axs[i,j].text(0.5,0.98,'{:>8.3f} ({:>8.3f}) +/- {:>6.3f}'.format(Mea,Med,Sig),
                          transform=axs[i,j].transAxes,ha='center',va='top',
                          fontdict={'fontsize':14,'fontname':'Arial','fontweight':'bold'})
            if i==2:
                axs[i,j].set_xlabel('Modified Julian Day',fontname='Arial',fontsize=16)
                for tl in axs[i,j].get_xticklabels():
                    tl.set_fontname('Arial'); tl.set_fontsize(14)

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    strTmp=OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotEOPDiff2(AC1,fERPList1,AC2,fERPList2,rMJD1,rMJD2,OutFilePrefix,OutFileSuffix):
    '''
    Plot the ERP diff between two AC solutions

      AC1/AC2 --- Code ID of the ac, such as
                  # phb, my own solution
                  # gfz
                  # jpl
    fERPList1 --- ERP file from AC1 solutions
    fERPList2 --- ERP file from AC2 products
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    if AC1=='phb':
        EOP1=GetPHBERP(fERPList1,rMJD1,rMJD2)
    elif AC1=='gfz':
        EOP1=GetGFZERP(fERPList1,rMJD1,rMJD2)
    elif AC1=='cod' or AC1=='cof':
        EOP1=GetCODERP(fERPList1,rMJD1,rMJD2)
    elif AC1=='jpl':
        EOP1=GetJPLERP(fERPList1,rMJD1,rMJD2)
    elif AC1=='igs':
        EOP1=GetIGSERP(fERPList1,rMJD1,rMJD2)
    else:
        sys.exit('Unknown AC, '+AC1)

    if AC2=='phb':
        EOP2=GetPHBERP(fERPList2,rMJD1,rMJD2)
    elif AC2=='gfz':
        EOP2=GetGFZERP(fERPList2,rMJD1,rMJD2)
    elif AC2=='cod' or AC2=='cof':
        EOP2=GetCODERP(fERPList2,rMJD1,rMJD2)
    elif AC2=='jpl':
        EOP2=GetJPLERP(fERPList2,rMJD1,rMJD2)
    elif AC2=='igs':
        EOP2=GetIGSERP(fERPList2,rMJD1,rMJD2)
    else:
        sys.exit('Unknown AC, '+AC2)

    X=[[],[],[],[],[],[],[]]
    for i in range(EOP1[0].size):
        for j in range(EOP2[0].size):
            if EOP2[0,j] < EOP1[0,i]-0.1:
                continue
            elif EOP2[0,j] > EOP1[0,i]+0.1:
                break
            else:
                dEOP=np.zeros(7)
                dEOP[0]=EOP1[0,i]
                YYYY,DOY=GNSSTime.mjd2doy(int(dEOP[0]))
                # diff of xpole, xpole rate in 1e-6 arcsec
                dEOP[1]=EOP2[1,j]-EOP1[1,i]; dEOP[2]=EOP2[7,j]-EOP1[7,i]
                # diff of ypole, ypole rate in 1e-6 arcsec
                dEOP[3]=EOP2[3,j]-EOP1[3,i]; dEOP[4]=EOP2[9,j]-EOP1[9,i]
                # diff of UT1-UTC, LOD in 1e-6 sec
                dEOP[5]=(EOP2[5,j]-EOP1[5,i])/10; dEOP[6]=(EOP2[11,j]-EOP1[11,i])/10
                # # Report to the terminal
                # if (abs(dEOP[1]) > 200) or (abs(dEOP[3]) > 200) or (abs(dEOP[5]) > 100):
                #     strTmp='{:>04d} {:>03d} {:>10.2f} {:>10.2f} {:>10.2f}'.format(YYYY,DOY,
                #             dEOP[1],dEOP[3],dEOP[5])
                #     print(strTmp)
                #     continue
                for k in range(7):
                    X[k].append(dEOP[k])
    # No diff available
    if len(X[0])==0:
        return
    # Report to the terminal
    cEOP=['xpole','dxpole','ypole','dypole','UT1','LOD']
    print('{: <15s} {: >11s} {: >11s} {: >11s}'.format('EOP_'+AC2+'-'+AC1,
          'Mean','STD','RMS'))
    for i in range(6):
        # Mean && STD
        Mea=np.mean(X[i+1]); Sig=np.std(X[i+1])
        # RMS
        RMS=0; nEpo=len(X[0])
        for j in range(nEpo):
            RMS=RMS + X[i+1][j]*X[i+1][j]
        RMS=np.sqrt(RMS/nEpo)
        print('{: <15s} {: >11.3f} {: >11.3f} {: >11.3f}'.format(cEOP[i],Mea,Sig,RMS))

    Diff=np.array(X)
    fig,axs=plt.subplots(3,2,sharex='col',squeeze=False,figsize=(14,8))
    # fig.subplots_adjust(hspace=0.1)

    yLab=[r'$x_p$ [$\mu$as]',r'$\.x_p$ [$\mu$as/d]',
          r'$y_p$ [$\mu$as]',r'$\.y_p$ [$\mu$as/d]',
          r'UT1-UTC [$\mu$s]',r'LOD [$\mu$s/d]']

    for i in range(3):
        for j in range(2):
            k=1+2*i+j
            #Cal the robusted mean, median and std
            Mea,Med,Sig=sigma_clipped_stats(np.array(Diff[k]),sigma=3,maxiters=5)

            axs[i,j].plot(Diff[0],Diff[k],'o--r',ms=3,lw=1)
            axs[i,j].axhline(color='darkgray',linestyle='dashed',alpha=0.5)
            axs[i,j].grid(b=True,which='both',axis='y',color='darkgray',linestyle='--',
                          linewidth=0.8)
            axs[i,j].set_ylabel(yLab[k-1],fontname='Arial',fontsize=16)
            for tl in axs[i,j].get_yticklabels():
                tl.set_fontname('Arial'); tl.set_fontsize(14)
            axs[i,j].text(0.5,0.98,'{:>8.3f} ({:>8.3f}) +/- {:>6.3f}'.format(Mea,Med,Sig),
                          transform=axs[i,j].transAxes,ha='center',va='top',
                          fontdict={'fontsize':14,'fontname':'Arial','fontweight':'bold'})
            if i==2:
                axs[i,j].set_xlabel('Modified Julian Day',fontname='Arial',fontsize=16)
                for tl in axs[i,j].get_xticklabels():
                    tl.set_fontname('Arial'); tl.set_fontsize(14)

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    strTmp=OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotEOPFit1(fEOPFit,OutFilePrefix,OutFileSuffix):
    '''
    Plot the eopfit file, i.e. the fitted model VS the observations
    '''

    cEOP=['XPOLE','YPOLE','UT1','DPSI','DEPSI']
    fig,axs=plt.subplots(5,1,sharex='col',squeeze=False,figsize=(12,4*5))
    formatterx=mpl.ticker.StrMethodFormatter('{x:7.1f}')

    for k in range(5):
        with open(fEOPFit,mode='rt') as fOb:
            # Read the fitted parameters
            lBeg=False
            for cLine in fOb:
                if '+EST '+cEOP[k] in cLine:
                    lBeg=True
                    cWords=cLine.split()
                    nDeg=int(cWords[3]); nPW=int(cWords[4])
                    PW=np.zeros((nPW,2+nDeg+1))
                    iPW=-1
                elif '-EST '+cEOP[k] in cLine:
                    break
                elif lBeg and cLine[0:1]==' ':
                    iPW=iPW+1
                    cWords=cLine.split()
                    PW[iPW,0]=int(cWords[0]) + float(cWords[1])/86400
                    PW[iPW,1]=int(cWords[3]) + float(cWords[4])/86400
                    for i in range(nDeg+1):
                        PW[iPW,i+2]=float(cWords[i+6])
            # Read the observations (OmC)
            fOb.seek(0); lBeg=False; XEOP=[[],[],[],[]]
            for cLine in fOb:
                if '+FITRES '+cEOP[k] in cLine:
                    lBeg=True
                elif '-FITRES '+cEOP[k] in cLine:
                    break
                elif lBeg and cLine[0:1]==' ':
                    cWords=cLine.split()
                    rMJD=int(cWords[0]) + float(cWords[1])/86400
                    XEOP[0].append(rMJD)
                    # Observations
                    XEOP[1].append(float(cWords[3]))
                    # Residuals
                    XEOP[3].append(float(cWords[4]))
                    # Cal the fitted values
                    iPW=-1
                    for i in range(nPW):
                        if (rMJD-PW[i,0])*86400 <= -1:
                            continue
                        if (rMJD-PW[i,1])*86400 >= -1:
                            continue
                        iPW=i
                        break
                    if iPW < 0:
                        sys.exit('Not found piece for epoch '+cWords[0]+' '+cWords[1])
                    else:
                        dt=rMJD-PW[iPW,0]; dX=0
                        for i in range(nDeg+1):
                            dX=dX + dt**i/math.factorial(i)*PW[iPW,i+2]
                        XEOP[2].append(dX)
        axs[k,0].plot(XEOP[0],XEOP[1],'.r',ms=2,label='Obs')
        axs[k,0].plot(XEOP[0],XEOP[2],'^g',ms=2,label='Fit')
        ax0=axs[k,0].twinx()
        ax0.plot(XEOP[0],XEOP[3],'.b',ms=2)
        ax0.set_ylabel('Residuals',fontname='Arial',fontsize=16)
        for tl in ax0.get_yticklabels():
            tl.set_fontname('Arial'); tl.set_fontsize(14)

        axs[k,0].grid(b=True,which='both',axis='both',color='darkgray',linestyle='--',
                    linewidth=0.8)
        axs[k,0].legend(ncol=2,loc='upper right',bbox_to_anchor=(1.0,1.0),
                        framealpha=0.6,prop={'family':'Arial','size':14})
        axs[k,0].ticklabel_format(axis='y',style='sci',useOffset=False,useMathText=True)
        axs[k,0].set_ylabel(cEOP[k],fontname='Arial',fontsize=16)
        for tl in axs[k,0].get_yticklabels():
            tl.set_fontname('Arial'); tl.set_fontsize(14)

    axs[k,0].set_xlabel('Modified Julian Day',fontname='Arial',fontsize=16)
    axs[k,0].xaxis.set_major_formatter(formatterx)
    for tl in axs[k,0].get_xticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)

    strTmp=OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotHFEOP(fInp,OutFilePrefix,OutFileSuffix):
    '''
    Plot the output from `hfeop`
    '''
    EOP=[]
    for i in range(17):
        EOP.append([])
    with open(fInp,mode='rt') as fOb:
        for cLine in fOb:
            if (cLine[0:1]=='#') or (len(cLine) < 5):
                continue
            cWords=cLine.split()
            rMJD=float(cWords[ 0])
            EOP[ 0].append(rMJD)
            # HF-xpole, 1e-6 arcsec
            EOP[ 1].append(float(cWords[ 2]))
            # HF-ypole, 1e-6 arcsec
            EOP[ 2].append(float(cWords[ 3]))
            # HF-UT1, 1e-6 sec
            EOP[ 3].append(float(cWords[ 4]))
            # HF-LOD, 1e-6 sec/day
            EOP[ 4].append(float(cWords[ 5]))
            # HF-xpole rate, 1e-6 arcsec/day
            EOP[ 5].append(float(cWords[ 6])/1000)
            # HF-ypole rate, 1e-6 arcsec/day
            EOP[ 6].append(float(cWords[ 7])/1000)
            # HF-UT1 rate, 1e-6 sec/day
            EOP[ 7].append(float(cWords[ 8])/1000)
            # HF-LOD rate, 1e-6 sec/sec/day
            EOP[ 8].append(float(cWords[ 9])/1000)
            # xpole cor, 1e-6 arcsec
            EOP[ 9].append(float(cWords[10]))
            # ypole cor, 1e-6 arcsec
            EOP[10].append(float(cWords[11]))
            # UT1 cor, 1e-6 sec
            EOP[11].append(float(cWords[12]))
            # LOD cor, 1e-6 sec per day
            EOP[12].append(float(cWords[13]))
            # UT1 zonal tide, 1e-6 sec
            EOP[13].append(float(cWords[14]))
            # LOD zonal tide, 1e-6 sec
            EOP[14].append(float(cWords[15]))
            # omega zonal tide, 1e-6 arcsec/day
            EOP[15].append(float(cWords[16]))
            EOP[16].append(np.nan)
            nRec=len(EOP[0])
            if nRec==1:
                continue
            dt=EOP[0][nRec-1]-EOP[0][nRec-2]
            # diff btw calculated LOD zonal tide from UT1 zonal tide and the reported LOD zonal tide
            EOP[16][nRec-2]=(EOP[13][nRec-1]-EOP[13][nRec-2])/dt + EOP[14][nRec-2]

    fig,axs=plt.subplots(8,2,sharex='col',squeeze=False,figsize=(18,8*4))

    cYLab=[[r'HF-xp [$\mu$as]',   r'HF-xp dot [$\mu$as/d]'],
           [r'HF-yp [$\mu$as]',   r'HF-yp dot [$\mu$as/d]'],
           [r'HF-UT1 [$\mu$s]',   r'HF-UT1 dot [$\mu$s/d]'],
           [r'HF-LOD [$\mu$s/d]', r'HF-LOD dot [$\mu$s/d**2]'],
           [r'xp-cor [$\mu$as]' , r'UT1 zonal [$\mu$s]'],
           [r'yp-cor [$\mu$as]' , r'LOD zonal [$\mu$s/d]'],
           [r'UT1-cor [$\mu$s]',  r'omega zonal [$\mu$as/d]'],
           [r'LOD-cor [$\mu$s/d]',r'dUT1+LOD zonal [$\mu$s/d]']]
    iYax=[[1,5],[2,6],[3,7],[4,8],[9,13],[10,14],[11,15],[12,16]]
    for i in range(8):
        for j in range(2):
            if iYax[i][j]==0:
                continue
            axs[i,j].plot(EOP[0],EOP[iYax[i][j]],'.r',ms=2)
            axs[i,j].set_ylabel(cYLab[i][j],fontname='Arial',fontsize=16)
            for tl in axs[i,j].get_yticklabels():
                tl.set_fontname('Arial'); tl.set_fontsize(14)
    axs[7,0].set_xlabel('Modified Julian Day',fontname='Arial',fontsize=16)
    axs[7,1].set_xlabel('Modified Julian Day',fontname='Arial',fontsize=16)
    for tl in axs[7,0].get_xticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)
    for tl in axs[7,1].get_xticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)

    strTmp=OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
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

    # InFilePrefix=r'D:/Code/PROJECT/WORK_EOP/'

    # fC04=InFilePrefix+'EOP_14_C04_IAU2000A_one_file_1962-now.txt'
    # fFinal=InFilePrefix+'finals2000A.all'

    # OutFilePrefix=r'D:/Code/PROJECT/WORK_EOP/'
    # OutFilePrefix=r'D:/Code/PROJECT/WORK2019335_ERROR/'
    OutFilePrefix=os.path.join(cDskPre0,r'PRO_2019001_2020366/D672/EOP/')

    # rMJD1=58484; rMJD2=59215
    # OutFileSuffix='Final2000A_C04'
    # PlotEOPDiff1(fC04,fFinal,rMJD1,rMJD2,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='Final2000A'
    # PlotEOP1(fFinal,'FIN',rMJD1,rMJD2,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='IERSC04'
    # PlotEOP1(fC04,'C04',rMJD1,rMJD2,OutFilePrefix,OutFileSuffix)

    # fEOPFit=r'D:/Code/PROJECT/WORK2019335_ERROR/EOPFitRes_2019335'
    # fEOPFit=r'Y:/PRO_2019001_2020366_WORK/I2_G_6/WORK2019335/EOPFitRes_2019335'
    # OutFileSuffix='eopfit_2019335.png'
    # PlotEOPFit1(fEOPFit,OutFilePrefix,OutFileSuffix)

    InFilePrefix=os.path.join(cWrkPre0,r'PRO_2019001_2020366_WORK/D672/WORK20193??/')
    # InFilePrefix=os.path.join(cWrkPre0,r'IGS/????/')
    fERPList1=glob.glob(InFilePrefix+'phb?????.erp')

    # OutFileSuffix='ERP_sig'
    # PlotEOP2('phb',fERPList1,OutFilePrefix,OutFileSuffix)

    InFilePrefix=os.path.join(cWrkPre0,r'IGS/????/')
    fERPList2=glob.glob(InFilePrefix+'igs????7.erp')

    OutFileSuffix='ERPDif_phb_igs'
    PlotEOPDiff2('phb',fERPList1,'igs',fERPList2,58818,58848,OutFilePrefix,OutFileSuffix)

    # OutFileSuffix='hfeop.png'
    # PlotHFEOP(r'D:/Code/PROJECT/WORK2019335_ERROR/hfeop_2019335',OutFilePrefix,OutFileSuffix)
