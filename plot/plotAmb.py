#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
'''
__author__ = 'hanbing'

# Standard library imports
import subprocess
import os
import sys
import os.path
import glob

# Related third party imports
import numpy as np
import matplotlib as mpl
from scipy import special
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# Local application/library specific imports
from PySRC.miscellaneous import GNSSTime

def ReadDDWLAmb(fDDWL,cSat0,cSta0,cAmb0,lNorm):
    '''
    Read DD WL amb series for specified satellites, stations and amb types

    Return:

    cSat --- Satellite list
    cAmb --- Amb IDs for each arc
       X --- epoch list for each arc
       Y --- DD WLs for each arc
    '''

    cSat=[]; cAmb=[]; X=[]; Y=[]
    with open(fDDWL,mode='rt') as fOb:
        while True:
            cLine=fOb.readline()
            if not cLine:
                break
            cLine.rstrip('\r\n')
            if cLine[0:1]=='%':
                continue
            cWords=cLine.split()
            # Check if this amb is required
            if (cAmb0[0]!='ALL' and cWords[0] not in cAmb0) or \
               (cSta0[0]!='ALL' and (cWords[1] not in cSta0 or cWords[2] not in cSta0)) or \
               (cSat0[0]!='ALL' and (cWords[3] not in cSat0 or cWords[4] not in cSat0)):
                lExclude=True
            else:
                lExclude=False
            # Amb ID
            cTmp=cWords[3]+'-'+cWords[4]+'-'+cWords[1]+'-'+cWords[2]+'-'+cWords[0]
            # Read for this arc
            rEpo=[]; DDWL=[]
            for j in range(int(cWords[6])-int(cWords[5])+1):
                cLine=fOb.readline()
                cWords=cLine.split()
                if int(cWords[1]) != 0:
                    # Exclude bad points
                    continue
                rEpo.append(float(cWords[0]))
                DDWL.append(float(cWords[2]))
            if len(DDWL)==0 or lExclude:
                continue
            if lNorm:
                DDWL=DDWL - np.rint(np.mean(DDWL))
            cAmb.append(cTmp); X.append(rEpo); Y.append(DDWL)
            if cTmp[0:3] not in cSat:
                cSat.append(cTmp[0:3])
            if cTmp[4:7] not in cSat:
                cSat.append(cTmp[4:7])
    cSat.sort()

    return cSat,cAmb,X,Y


def bdeci0(x,Sigma,n):
    '''
    Calculte the rounding region for ambiguity resolution
    Ref. Dong and Bock 1989.

      x --- absolute deviation of real-valued ambiguity to its
            nearest integer, must fall in [0, 0.5]
  Sigma --- formal uncertainty of the real-valued ambiguity
      n --- Max. integer of the search range
    '''

    CutDev=0.4
    CutSig=1/3
    sig=3*Sigma

    if x < CutDev:
        #Tapering function
        t1=1.0-x/CutDev
        if sig > CutSig:
            t2=0
        else:
            t2=(CutSig-sig)*3
        T=t1*t2
    else:
        T=0

    #Compute Q0
    Q0=0
    s0=np.sqrt(2)*sig
    for i in range(1,n+1):
        ef1=special.erfc((i-x)/s0)
        ef2=special.erfc((i+x)/s0)
        Q0=Q0 + (ef1-ef2)
    if Q0<1e-9:
        Q0=1e-9

    return Q0,T

def bdeci1(x,Sigma,CutDev,CutSig,n):
    '''
    Calculte the rounding region for ambiguity resolution
    Ref. Dong and Bock 1989. [Coded in FORTRAN]

      x --- absolute deviation of real-valued ambiguity to its
            nearest integer, must fall in [0, 0.5]
  Sigma --- formal uncertainty of the real-valued ambiguity
 CutDev --- cutoff deviation
 CutSig --- cutoff sigma
      n --- Max. integer of the search range
    '''

    sig=Sigma

    if x < CutDev:
        #Tapering function
        t1=1.0-x/CutDev
        if sig >= CutSig:
            t2=0
        else:
            t2=(CutSig-sig)*3
        T=t1*t1*t2
    else:
        T=0

    #Compute Q0
    Q0=0
    s0=np.sqrt(2)*sig
    for i in range(1,n+1):
        ef1=special.erfc((i-x)/s0)
        ef2=special.erfc((i+x)/s0)
        Q0=Q0 + (ef1-ef2)
    if Q0<1e-9:
        Q0=1e-9

    return Q0,T

def PlotAmbRoundingRegion0(OutFilePrefix,OutFileSuffix):
    '''
    Plot the rounding region for ambiguity resolution
    Ref. Dong and Bock 1989.
    '''

    CutDev=0.5
    CutSig=1.0
    alpha=0.1
    eps=1e-4
    x0=[]
    y0=[]
    x1=[]
    y1=[]
    for Dev in np.linspace(0,CutDev,num=1000,endpoint=False,dtype=np.float64):
        for Sig in np.linspace(eps,CutSig,num=10000,endpoint=False,dtype=np.float64):
            Q0,T=bdeci0(Dev,Sig,50)
            if Q0*100 <= alpha:
                x0.append(Dev)
                y0.append(Sig)
            if T/Q0 >= 1000:
                x1.append(Dev)
                y1.append(Sig)

    fig,axs=plt.subplots(1,1,figsize=(8,8))
    formatterx=mpl.ticker.StrMethodFormatter('{x:3.1f}')
    formattery=mpl.ticker.StrMethodFormatter('{x:4.2f}')

    axs.set_ylim(bottom=0,top=CutSig)
    axs.set_ylabel('Formal 1 sigma error')
    axs.yaxis.set_major_formatter(formattery)
    axs.set_xlim(left=0,right=CutDev)
    axs.set_xlabel('Absolute deviation from integer')
    axs.xaxis.set_major_formatter(formatterx)
    axs.plot(x0,y0,marker='.',markeredgecolor='r',markerfacecolor='r')
    axs.plot(x1,y1,marker='.',markeredgecolor='b',markerfacecolor='b')

    strTmp=OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotAmbRoundingRegion1(OutFilePrefix,OutFileSuffix):
    '''
    Plot the rounding region for ambiguity resolution
    Ref. Dong and Bock 1989.
    '''
    CutDev=0.5
    CutSig=1.0
    alpha=0.1
    eps=1e-4
    x0=[]
    y0=[]
    x1=[]
    y1=[]
    for Dev in np.linspace(0,CutDev,num=1000,endpoint=False,dtype=np.float64):
        for Sig in np.linspace(eps,CutSig,num=10000,endpoint=False,dtype=np.float64):
            Q0,T=bdeci1(Dev,Sig,0.15,0.15,50)
            if Q0*100 <= alpha:
                x0.append(Dev)
                y0.append(Sig)
            if T/Q0 >= 1000:
                x1.append(Dev)
                y1.append(Sig)

    fig,axs=plt.subplots(1,1,figsize=(8,8))
    formatterx=mpl.ticker.StrMethodFormatter('{x:3.1f}')
    formattery=mpl.ticker.StrMethodFormatter('{x:4.2f}')

    axs.set_ylim(bottom=0,top=CutSig)
    axs.set_ylabel('Formal 1 sigma error')
    axs.yaxis.set_major_formatter(formattery)
    axs.set_xlim(left=0,right=CutDev)
    axs.set_xlabel('Absolute deviation from integer')
    axs.xaxis.set_major_formatter(formatterx)
    axs.plot(x0,y0,marker='.',markeredgecolor='r',markerfacecolor='r')
    axs.plot(x1,y1,marker='.',markeredgecolor='b',markerfacecolor='b')

    strTmp=OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotZDWLAmb(fList,cSat0,cSta0,cAmb0,lNorm,OutFilePrefix,OutFileSuffix):
    '''
    Plot ZD WL amb file
    lNorm : Whether remove the mean integer in each arc
    '''
    cAmb=[]
    cSta=[]
    cSat=[]
    for i in range(len(fList)):
        with open(fList[i],mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:1]=='%':
                    continue
                if cLine[0:3]!='AMB':
                    continue
                cWords=cLine.split()
                if (cAmb0[0]=='ALL' or cWords[0] in cAmb0) and (cWords[0] not in cAmb):
                    cAmb.append(cWords[0])
                if (cSta0[0]=='ALL' or cWords[1] in cSta0) and (cWords[1] not in cSta):
                    cSta.append(cWords[1])
                if (cSat0[0]=='ALL' or cWords[2] in cSat0) and (cWords[2] not in cSat):
                    cSat.append(cWords[2])
    cAmb.sort()
    nAmb=len(cAmb)
    cSta.sort()
    nSta=len(cSta)
    cSat.sort()
    nSat=len(cSat)

    X=[]
    Y=[]
    for i in range(nSta):
        X.append([])
        Y.append([])
        for j in range(nSat):
            X[i].append([])
            Y[i].append([])
            for k in range(nAmb):
                X[i][j].append([])
                Y[i][j].append([])
    #Read data
    for l in range(len(fList)):
        with open(fList[l],mode='rt') as fOb:
            while True:
                cLine=fOb.readline()
                if not cLine:
                    break
                cLine.rstrip('\r\n')
                if cLine[0:1]=='%':
                    continue
                if cLine[0:3]!='AMB':
                    continue
                cWords=cLine.split()
                if cWords[0] not in cAmb:
                    k=-1
                else:
                    k=cAmb.index(cWords[0])
                if cWords[1] not in cSta:
                    i=-1
                else:
                    i=cSta.index(cWords[1])
                if cWords[2] not in cSat:
                    j=-1
                else:
                    j=cSat.index(cWords[2])
                rEpo=[]
                ZDWL=[]
                n=int(cWords[9])
                for m in range(n):
                    cLine=fOb.readline()
                    cWords=cLine.split()
                    if int(cWords[1])!=0:
                        continue
                    rEpo.append(float(cWords[0]))
                    ZDWL.append(float(cWords[2]))
                if lNorm:
                    ZDWL=ZDWL-np.rint(np.mean(ZDWL))
                if i==-1 or j==-1 or k==-1:
                    continue
                X[i][j][k].append(rEpo)
                Y[i][j][k].append(ZDWL)
    strTmp=OutFilePrefix+OutFileSuffix
    with PdfPages(strTmp) as pdf:
        for j in range(nSat):
            for i in range(nSta):
                for k in range(nAmb):
                    mAmb=len(X[i][j][k])
                    if mAmb==0:
                        continue
                    fig,axs=plt.subplots(1,1,figsize=(10,5))
                    for iAmb in range(mAmb):
                        axs.plot(X[i][j][k][iAmb],Y[i][j][k][iAmb],'.')
                    axs.text(0.05,0.95,cSat[j]+'-'+cSta[i]+'-'+cAmb[k],transform=axs.transAxes,
                             ha='left',va='top')
                    axs.set_ylabel('ZD WL [cycle]')
                    pdf.savefig(fig,bbox_inches='tight')
                    plt.close(fig)

def PlotDDWLAmb1(fDDWL,cSat0,cSta0,cAmb0,lNorm,OutFilePrefix,OutFileSuffix):
    '''
    Plot DD WL series for specified DD amb, amb-specifically

    cSat0 --- Specified satellites
    cSta0 --- Specified stations
    cAmb0 --- Specified amb types

    lNorm --- Whether remove the mean integer in each arc
    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # Get all specified DD amb
    cAmb=[]
    with open(fDDWL,mode='rt') as fOb:
        for cLine in fOb:
            if cLine[0:1]=='%':
                continue
            cWords=cLine.split()
            if cAmb0[0]!='ALL' and cWords[0] not in cAmb0:
                continue
            if cSta0[0]!='ALL' and (cWords[1] not in cSta0 or cWords[2] not in cSta0):
                continue
            if cSat0[0]!='ALL' and (cWords[3] not in cSat0 or cWords[4] not in cSat0):
                continue
            cTmp=cWords[3]+'-'+cWords[4]+'-'+cWords[1]+'-'+cWords[2]+'-'+cWords[0]
            if cTmp not in cAmb:
                cAmb.append(cTmp)
    nAmb=len(cAmb)
    if nAmb == 0:
        sys.exit('No valid amb found in '+fDDWL)
    cAmb.sort()

    X=[]; Y=[]
    for i in range(nAmb):
        X.append([]); Y.append([])

    with open(fDDWL,mode='rt') as fOb:
        while True:
            cLine=fOb.readline()
            if not cLine:
                break
            cLine.rstrip('\r\n')
            if cLine[0:1]=='%':
                continue
            cWords=cLine.split()
            cTmp=cWords[3]+'-'+cWords[4]+'-'+cWords[1]+'-'+cWords[2]+'-'+cWords[0]
            if cTmp not in cAmb:
                iAmb=-1
            else:
                iAmb=cAmb.index(cTmp)
            rEpo=[]; DDWL=[]
            # Read each epoch for this arc
            for j in range(int(cWords[6])-int(cWords[5])+1):
                cLine=fOb.readline()
                cWords=cLine.split()
                if int(cWords[1]) != 0:
                    # Exclude bad points
                    continue
                rEpo.append(float(cWords[0]))
                DDWL.append(float(cWords[2]))
            if len(DDWL)==0 or iAmb==-1:
                continue
            if lNorm:
                DDWL=DDWL - np.rint(np.mean(DDWL))
            X[iAmb].append(rEpo); Y[iAmb].append(DDWL)

    strTmp=os.path.join(OutFilePrefix,OutFileSuffix+'.pdf')
    with PdfPages(strTmp) as pdf:
        for i in range(nAmb):
            nArc=len(X[i])
            if nArc==0:
                continue
            fig,axs=plt.subplots(1,1,squeeze=False,figsize=(8,3))
            for j in range(nArc):
                axs[0,0].plot(X[i][j],Y[i][j],'.',ms=3)
                if j != 0:
                    continue
                axs[0,0].text(0.02,0.98,cAmb[i],transform=axs[0,0].transAxes,ha='left',va='top',
                              family='Arial',size=16,weight='bold')
                axs[0,0].set_ylabel('DD WL [cycle]',fontname='Arial',fontsize=16)
                for tl in axs[0,0].get_yticklabels():
                    tl.set_fontname('Arial'); tl.set_fontsize(14)
            axs[0,0].set_xlabel(r'Epoch',fontname='Arial',fontsize=16)
            for tl in axs[0,0].get_xticklabels():
                tl.set_fontname('Arial'); tl.set_fontsize(14)
            pdf.savefig(fig,bbox_inches='tight')
            plt.close(fig)

def PlotDDWLAmb2(fDDWL,cSat0,cSta0,cAmb0,lNorm,OutFilePrefix,OutFileSuffix):
    '''
    Plot DD WL series for specified DD amb, sat-specifically

    cSat0 --- Specified satellites
    cSta0 --- Specified stations
    cAmb0 --- Specified amb types

    lNorm --- Whether remove the mean integer in each arc
    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # Get all specified DD amb
    cSat,cAmb,rEpo,DDWL=ReadDDWLAmb(fDDWL,cSat0,cSta0,cAmb0,lNorm)
    nSat=len(cSat); nAmb=len(cAmb)
    if nAmb == 0:
        sys.exit('No valid amb found in '+fDDWL)
    fig,axs=plt.subplots(nSat,1,sharex='col',squeeze=False,figsize=(12,3*nSat))

    for i in range(nSat):
        axs[i,0].text(0.02,0.98,cSat[i],transform=axs[i,0].transAxes,ha='left',va='top',
                      family='Arial',size=16,weight='bold')
        axs[i,0].set_ylabel('DD WL [cycle]',fontname='Arial',fontsize=16)
        for tl in axs[i,0].get_yticklabels():
            tl.set_fontname('Arial'); tl.set_fontsize(14)

        for j in range(nAmb):
            if cSat[i] not in cAmb[j]:
                continue
            axs[i,0].plot(rEpo[j],DDWL[j],'.')

    axs[i,0].set_xlabel(r'Epoch',fontname='Arial',fontsize=16)
    for tl in axs[i,0].get_xticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    # strTmp=OutFilePrefix+OutFileSuffix+'.svg'
    # fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotDDAmb1(fDDAmb,lPlotSatPair,OutFilePrefix,OutFileSuffix):
    '''
    Based on DD amb file, plot the distribution of Deviation VS Sigma of
    DD WL for each satellite

    lPlotSatPair --- Whether addtionally plot the same info for each satellite pair
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # Get all amb type, stations and satellites
    cAmb=[]; cSta=[]; cSat=[]; cSatPair=[]
    with open(fDDAmb,mode='rt') as fOb:
        for cLine in fOb:
            if cLine[0:1]=='%' or len(cLine)<3:
                continue
            cWords=cLine.split()
            if cWords[0] not in cAmb:
                cAmb.append(cWords[0])
            if cWords[1] not in cSta:
                cSta.append(cWords[1])
            if cWords[2] not in cSta:
                cSta.append(cWords[2])
            if cWords[3] not in cSat:
                cSat.append(cWords[3])
            if cWords[4] not in cSat:
                cSat.append(cWords[4])
            cTmp1=cWords[3]+'-'+cWords[4]
            cTmp2=cWords[4]+'-'+cWords[3]
            if cTmp1 not in cSatPair and cTmp2 not in cSatPair:
                cSatPair.append(cTmp1)
    cAmb.sort(); nAmb=len(cAmb)
    cSta.sort(); nSta=len(cSta)
    cSat.sort(); nSat=len(cSat)
    cSatPair.sort(); nSatPair=len(cSatPair)

    strTmp=os.path.join(OutFilePrefix,OutFileSuffix+'.pdf')
    with PdfPages(strTmp) as pdf:
        #1) Plot DD WL amb for each satellite
        fig,axs=plt.subplots(nSat,nAmb*3,sharex='col',squeeze=False,figsize=(nAmb*3*3,nSat*3))
        # fig.subplots_adjust(hspace=0.1)
        # fig.subplots_adjust(wspace=0.1)
        #Average deviation
        AvgDev1=np.zeros((nSat,nAmb))
        #Average absolute deviation
        AvgDev2=np.zeros((nSat,nAmb))
        #Average sigma
        AvgSig=np.zeros((nSat,nAmb))
        #Fixable rate
        FixRat=np.zeros((nSat,nAmb))
        for i in range(nSat):
            for j in range(nAmb):
                DDWL=[[],[]]; DDWL1=[[],[]]; DDWL2=[[],[]]
                with open(fDDAmb,mode='rt') as fOb:
                    for cLine in fOb:
                        if cLine[0:1]=='%' or len(cLine)<3:
                            continue
                        cWords=cLine.split()
                        if cWords[0] != cAmb[j]:
                            continue
                        if cWords[3] != cSat[i] and cWords[4] != cSat[i]:
                            continue
                        # Deviation to its integer
                        if cWords[21]=='T':
                            # Fixable
                            DDWL[0].append(float(cWords[19])-int(cWords[24]))
                            DDWL[1].append(float(cWords[20]))
                            DDWL1[0].append(float(cWords[19])-int(cWords[24]))
                            DDWL1[1].append(float(cWords[20]))
                        else:
                            # Non-fixable
                            DDWL[0].append(float(cWords[19])-np.rint(float(cWords[19])))
                            DDWL[1].append(float(cWords[20]))
                            DDWL2[0].append(float(cWords[19])-np.rint(float(cWords[19])))
                            DDWL2[1].append(float(cWords[20]))
                MeaDev1=np.mean(DDWL[0]); AvgDev1[i,j]=MeaDev1
                MeaDev2=np.mean(np.abs(DDWL[0])); AvgDev2[i,j]=MeaDev2
                MeaSig=np.ma.masked_greater(DDWL[1],1).mean(); AvgSig[i,j]=MeaSig
                #1.1) Dev && Sig distribution
                axs[i,3*j].plot(DDWL1[0],DDWL1[1],'.g',ms=3)
                axs[i,3*j].plot(DDWL2[0],DDWL2[1],'.r',ms=3)
                axs[i,3*j].grid(b=True,which='major',axis='y',color='darkgray',linestyle='--',
                                linewidth=0.4)
                axs[i,3*j].set_xlim(left=-0.6,right=0.6)
                axs[i,3*j].set_ylim(bottom=0,top=1.0)
                axs[i,3*j].text(0.02,0.98,cSat[i]+'-'+cAmb[j],transform=axs[i,3*j].transAxes,
                                ha='left',va='top',fontname='Arial')
                FixRat[i,j]=len(DDWL1[0])/len(DDWL[0])
                axs[i,3*j].text(0.98,0.98,'{:>5d}/{:>5.4f}'.format(len(DDWL[0]),FixRat[i,j]),
                                transform=axs[i,3*j].transAxes,ha='right',va='top',fontname='Arial')
                axs[i,3*j].set_ylabel('Sig',fontname='Arial')
                for tl in axs[i,3*j].get_yticklabels():
                    tl.set_fontname('Arial')
                if i == nSat-1:
                    axs[i,3*j].set_xlabel('Dev',fontname='Arial')
                    for tl in axs[i,3*j].get_xticklabels():
                        tl.set_fontname('Arial')

                #1.2) Histogram of Deviation
                axs[i,3*j+1].hist(DDWL[0],bins=50)
                axs[i,3*j+1].set_xlim(left=-0.6,right=0.6)
                axs[i,3*j+1].text(0.98,0.98,'{:>6.3f}/{:>5.3f}'.format(MeaDev1,MeaDev2),
                                  transform=axs[i,3*j+1].transAxes,ha='right',va='top',
                                  fontname='Arial')
                for tl in axs[i,3*j+1].get_yticklabels():
                    tl.set_fontname('Arial')
                if i == nSat-1:
                    axs[i,3*j+1].set_xlabel('Dev',fontname='Arial')
                    for tl in axs[i,3*j+1].get_xticklabels():
                        tl.set_fontname('Arial')

                #1.3) Histogram of Sigma
                axs[i,3*j+2].hist(DDWL[1],bins=50,range=(0.0,0.2))
                axs[i,3*j+2].set_xlim(left=0,right=0.2)
                axs[i,3*j+2].text(0.98,0.98,'{:>5.3f}'.format(MeaSig),
                                  transform=axs[i,3*j+2].transAxes,ha='right',va='top',
                                  fontname='Arial')
                for tl in axs[i,3*j+2].get_yticklabels():
                    tl.set_fontname('Arial')
                if i == nSat-1:
                    axs[i,3*j+2].set_xlabel('Sig',fontname='Arial')
                    for tl in axs[i,3*j+2].get_xticklabels():
                        tl.set_fontname('Arial')
        pdf.savefig(fig,bbox_inches='tight')
        plt.close(fig)
        #2) Statistics for each satellite, fixable WL rate and average sigma
        fig,axs=plt.subplots(nAmb,1,sharex='col',squeeze=False,figsize=(nSat*0.5,nAmb*3))
        w=1/(1+1); x=np.arange(nSat)
        for i in range(nAmb):
            axs[i,0].text(0.02,0.98,cAmb[i],transform=axs[i,0].transAxes,
                          ha='left',va='top',fontname='Arial')
            axs[i,0].set_ylim(bottom=0,top=1)
            axs[i,0].bar(x+(0-1/2)*w,FixRat[:,i],w,align='edge')
            axs[i,0].set_ylabel('Fixable WL Rate',fontname='Arial')
            for tl in axs[i,0].get_yticklabels():
                tl.set_fontname('Arial')
            axs[i,0].grid(b=True,which='major',axis='y',color='darkgray',linestyle='--',
                          linewidth=0.4)
            axs[i,0].set_axisbelow(True)

            axe=axs[i,0].twinx()
            axe.plot(x,AvgSig[:,i],'.r',ms=10)
            axe.set_ylabel('Mean Sig',fontname='Arial')
            for tl in axe.get_yticklabels():
                tl.set_fontname('Arial')
        axs[i,0].set_xlim(left=-1,right=nSat)
        axs[i,0].set_xticks(x)
        axs[i,0].set_xticklabels(cSat,fontdict={'fontname':'Arial'})
        pdf.savefig(fig,bbox_inches='tight')
        plt.close(fig)
        if not lPlotSatPair:
            return

        #2) Plot DD WL amb for each satellite pair
        #Average deviation
        AvgDev1=np.zeros((nSatPair,nAmb))
        #Average absolute deviation
        AvgDev2=np.zeros((nSatPair,nAmb))
        #Average sigma
        AvgSig=np.zeros((nSatPair,nAmb))

        fig,axs=plt.subplots(nSatPair,nAmb*3,sharex='col',squeeze=False,figsize=(nAmb*3*3,nSatPair*3))
        # fig.subplots_adjust(hspace=0.1)
        # fig.subplots_adjust(wspace=0.1)
        for i in range(nSatPair):
            for j in range(nAmb):
                DDWL=[[],[]]; DDWL1=[[],[]]; DDWL2=[[],[]]
                with open(fDDAmb,mode='rt') as fOb:
                    for cLine in fOb:
                        if cLine[0:1]=='%' or len(cLine)<3:
                            continue
                        cWords=cLine.split()
                        if cWords[0] != cAmb[j]:
                            continue
                        if cWords[3] != cSatPair[i][0:3] and cWords[3] != cSatPair[i][4:7]:
                            continue
                        if cWords[4] != cSatPair[i][0:3] and cWords[4] != cSatPair[i][4:7]:
                            continue
                        if cWords[21]=='T':
                            # Fixable
                            DDWL[0].append(float(cWords[19])-int(cWords[24]))
                            DDWL[1].append(float(cWords[20]))
                            DDWL1[0].append(float(cWords[19])-int(cWords[24]))
                            DDWL1[1].append(float(cWords[20]))
                        else:
                            # Non-fixable
                            DDWL[0].append(float(cWords[19])-np.rint(float(cWords[19])))
                            DDWL[1].append(float(cWords[20]))
                            DDWL2[0].append(float(cWords[19])-np.rint(float(cWords[19])))
                            DDWL2[1].append(float(cWords[20]))
                MeaDev1=np.mean(DDWL[0]); AvgDev1[i,j]=MeaDev1
                MeaDev2=np.mean(np.abs(DDWL[0])); AvgDev2[i,j]=MeaDev2
                MeaSig=np.ma.masked_greater(DDWL[1],1).mean(); AvgSig[i,j]=MeaSig
                #2.1) Dev && Sig distribution
                axs[i,3*j].plot(DDWL1[0],DDWL1[1],'.g',ms=3)
                axs[i,3*j].plot(DDWL2[0],DDWL2[1],'.r',ms=3)
                axs[i,3*j].grid(b=True,which='major',axis='y',color='darkgray',linestyle='--',
                                linewidth=0.4)
                axs[i,3*j].set_xlim(left=-0.6,right=0.6)
                axs[i,3*j].set_ylim(bottom=0,top=1.0)
                axs[i,3*j].text(0.02,0.98,cSatPair[i]+'-'+cAmb[j],transform=axs[i,3*j].transAxes,
                                ha='left',va='top',fontname='Arial')
                axs[i,3*j].text(0.98,0.98,'{:>5d}/{:>5.4f}'.format(len(DDWL[0]),len(DDWL1[0])/len(DDWL[0])),
                                transform=axs[i,3*j].transAxes,ha='right',va='top',fontname='Arial')
                axs[i,3*j].set_ylabel('Sig',fontname='Arial')
                for tl in axs[i,3*j].get_yticklabels():
                    tl.set_fontname('Arial')
                if i == nSatPair-1:
                    axs[i,3*j].set_xlabel('Dev',fontname='Arial')
                    for tl in axs[i,3*j].get_xticklabels():
                        tl.set_fontname('Arial')

                #2.2) Histogram of Deviation
                axs[i,3*j+1].hist(DDWL[0],bins=50)
                axs[i,3*j+1].set_xlim(left=-0.6,right=0.6)
                axs[i,3*j+1].text(0.98,0.98,'{:>6.3f}/{:>5.3f}'.format(MeaDev1,MeaDev2),
                                  transform=axs[i,3*j+1].transAxes,ha='right',va='top',
                                  fontname='Arial')
                for tl in axs[i,3*j+1].get_yticklabels():
                    tl.set_fontname('Arial')
                if i == nSatPair-1:
                    axs[i,3*j+1].set_xlabel('Dev',fontname='Arial')
                    for tl in axs[i,3*j+1].get_xticklabels():
                        tl.set_fontname('Arial')

                #2.3) Histogram of Sigma
                axs[i,3*j+2].hist(DDWL[1],bins=50,range=(0.0,0.2))
                axs[i,3*j+2].set_xlim(left=0,right=0.2)
                axs[i,3*j+2].text(0.98,0.98,'{:>5.3f}'.format(MeaSig),
                                  transform=axs[i,3*j+2].transAxes,ha='right',va='top',
                                  fontname='Arial')
                for tl in axs[i,3*j+2].get_yticklabels():
                    tl.set_fontname('Arial')
                if i == nSatPair-1:
                    axs[i,3*j+2].set_xlabel('Sig',fontname='Arial')
                    for tl in axs[i,3*j+2].get_xticklabels():
                        tl.set_fontname('Arial')
        pdf.savefig(fig,bbox_inches='tight')
        plt.close(fig)

def PlotDDAmb2(fDDAmbList,OutFilePrefix,OutFileSuffix):
    '''
    Based on DD amb file, plot DD WL/NL fixable rate series for each satellite

    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nFile=len(fDDAmbList)
    # Get the dimension
    cAmb=[]; cSta=[]; cSat=[]; rEpo=[]
    for i in range(nFile):
        # Epoch according to the file name
        YYYY=int(os.path.basename(fDDAmbList[i])[-7:-3])
        DOY=int(os.path.basename(fDDAmbList[i])[-3:])
        rEpo.append(GNSSTime.doy2mjd(YYYY,DOY))
        with open(fDDAmbList[i],mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:1]=='%' or len(cLine)<3:
                    continue
                cWords=cLine.split()
                if cWords[0] not in cAmb:
                    cAmb.append(cWords[0])
                if cWords[1] not in cSta:
                    cSta.append(cWords[1])
                if cWords[2] not in cSta:
                    cSta.append(cWords[2])
                if cWords[3] not in cSat:
                    cSat.append(cWords[3])
                if cWords[4] not in cSat:
                    cSat.append(cWords[4])
    cAmb.sort(); nAmb=len(cAmb)
    cSta.sort(); nSta=len(cSta)
    cSat.sort(); nSat=len(cSat)
    # Allocate arraies
    nDD=np.zeros((nSat,nAmb,nFile),dtype=np.int32)
    nDDWLFix=np.zeros((nSat,nAmb,nFile),dtype=np.int32)
    rDDWLFix=np.zeros((nSat,nAmb,nFile)); rDDWLFix[:,:,:]=np.nan
    # Average WL fixable rate for each satellite
    AvgWL=np.zeros((nSat,nAmb)); AvgWL[:,:]=np.nan
    nDDNLFix=np.zeros((nSat,nAmb,nFile),dtype=np.int32)
    rDDNLFix=np.zeros((nSat,nAmb,nFile)); rDDNLFix[:,:,:]=np.nan
    # Average NL fixable rate for each satellite
    AvgNL=np.zeros((nSat,nAmb)); AvgNL[:,:]=np.nan

    # Read files
    for i in range(nFile):
        with open(fDDAmbList[i],mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:1]=='%' or len(cLine)<3:
                    continue
                cWords=cLine.split()
                iAmb=cAmb.index(cWords[0])
                iSat1=cSat.index(cWords[3])
                iSat2=cSat.index(cWords[4])
                nDD[iSat1,iAmb,i]=nDD[iSat1,iAmb,i]+1
                nDD[iSat2,iAmb,i]=nDD[iSat2,iAmb,i]+1
                if cWords[21]=='T':
                    # WL
                    nDDWLFix[iSat1,iAmb,i]=nDDWLFix[iSat1,iAmb,i]+1
                    nDDWLFix[iSat2,iAmb,i]=nDDWLFix[iSat2,iAmb,i]+1
                if cWords[27]=='T':
                    # NL
                    nDDNLFix[iSat1,iAmb,i]=nDDNLFix[iSat1,iAmb,i]+1
                    nDDNLFix[iSat2,iAmb,i]=nDDNLFix[iSat2,iAmb,i]+1
        for iSat in range(nSat):
            for iAmb in range(nAmb):
                if nDD[iSat,iAmb,i]==0:
                    continue
                #Ratio of fixable rate
                rDDWLFix[iSat,iAmb,i]=nDDWLFix[iSat,iAmb,i]/nDD[iSat,iAmb,i]*100
                rDDNLFix[iSat,iAmb,i]=nDDNLFix[iSat,iAmb,i]/nDD[iSat,iAmb,i]*100

    fig,axs=plt.subplots(nSat,2,sharex='col',sharey='row',squeeze=False,figsize=(2*6,nSat*3))

    for i in range(nSat):
        axs[i,0].set_ylim(bottom=0,top=100)
        axs[i,0].set_ylabel('Fixable Rate [%]',fontname='Arial',fontsize=16)
        for tl in axs[i,0].get_yticklabels():
            tl.set_fontname('Arial'); tl.set_fontsize(14)

        # WL
        axs[i,0].text(0.02,0.98,cSat[i]+' WL',transform=axs[i,0].transAxes,
                      ha='left',va='top',fontname='Arial',fontsize=16)
        axs[i,0].grid(b=True,which='both',axis='y',color='darkgray',linestyle='--',
                     linewidth=0.4)
        for j in range(nAmb):
            #Average of fixable WL ratio
            AvgWL[i,j]=np.nanmean(rDDWLFix[i,j,:])
            axs[i,0].plot(rEpo,rDDWLFix[i,j,:],'o--',lw=1,ms=3,label=cAmb[j])

        # NL
        axs[i,1].text(0.02,0.98,cSat[i]+' NL',transform=axs[i,1].transAxes,
                      ha='left',va='top',fontname='Arial',fontsize=16)
        axs[i,1].grid(b=True,which='both',axis='y',color='darkgray',linestyle='--',
                     linewidth=0.4)
        for j in range(nAmb):
            #Average of fixable NL ratio
            AvgNL[i,j]=np.nanmean(rDDNLFix[i,j,:])
            axs[i,1].plot(rEpo,rDDNLFix[i,j,:],'o--',lw=1,ms=3,label=cAmb[j])
    axs[i,0].set_xlabel(r'Modified Julian Day',fontname='Arial',fontsize=16)
    for tl in axs[i,0].get_xticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)
    axs[i,1].set_xlabel(r'Modified Julian Day',fontname='Arial',fontsize=16)
    for tl in axs[i,1].get_xticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    strTmp=OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)


def PlotAmbCon(cSer,fList,OutFilePrefix,OutFileSuffix):
    '''
    Plot ambcon file
    1) NL fix rate
    2) WL fix rate
    3) Number of inde DD amb
    4) Number of DD amb
    5) Number of ZD amb
    6) Number of sat
    7) Number of sta
    '''

    fig,axs=plt.subplots(7,1,sharex='col',squeeze=False,figsize=(8,7*2))
    fig.subplots_adjust(hspace=0.1)

    axs[0,0].set_ylabel('NL fix (%)')
    axs[1,0].set_ylabel('WL fix (%)')
    axs[2,0].set_ylabel('# of inde DD amb')
    axs[3,0].set_ylabel('# of DD amb')
    axs[4,0].set_ylabel('# of ZD amb')
    axs[5,0].set_ylabel('# of sat')
    axs[6,0].set_ylabel('# of sta')

    nSer=len(cSer)
    for k in range(nSer):
        nFile=len(fList[k])
        #Epoch
        Epo=np.zeros(nFile)
        #Number of ZD amb
        nZD=np.zeros(nFile,dtype=np.int32)
        #Number of DD amb
        nDD=np.zeros(nFile,dtype=np.int32)
        #Number of independent DD amb
        nDDInde=np.zeros(nFile,dtype=np.int32)
        #Number of fixed independent DD WL amb
        nDDWLFix=np.zeros(nFile,dtype=np.int32)
        #Number of fixed independent DD (NL) amb
        nDDNLFix=np.zeros(nFile,dtype=np.int32)
        #Number of sat
        nSat=np.zeros(nFile,dtype=np.uint8)
        #Number of sta
        nSta=np.zeros(nFile,dtype=np.uint16)

        for i in range(nFile):
            with open(fList[k][i],mode='rt') as fOb:
                nLine=0
                for cLine in fOb:
                    nLine=nLine+1
                    if nLine==1:
                        cWords=cLine[26:].split()
                        nDDNLFix[i]=int(cWords[0])
                    if nLine==2:
                        cWords=cLine[26:].split()
                        nDDWLFix[i]=int(cWords[0])
                    if nLine==3:
                        cWords=cLine.split()
                        Epo[i]=int(cWords[0])+float(cWords[1])/86400
                        nSat[i]=int(cWords[4])
                        nSta[i]=int(cWords[5])
                        nZD[i]=int(cWords[8])
                        nDD[i]=int(cWords[10])
                        nDDInde[i]=int(cWords[11])
                    if nLine>=4:
                        break
        #(NL) fix rate
        axs[0,0].plot(Epo,np.divide(nDDNLFix,nDDInde)*100,marker='o',label=cSer[k])
        #WL fix rate
        axs[1,0].plot(Epo,np.divide(nDDWLFix,nDDInde)*100,marker='o',label=cSer[k])
        #Number of independent DD amb
        axs[2,0].plot(Epo,nDDInde,marker='o',label=cSer[k])
        #Number of DD amb
        axs[3,0].plot(Epo,nDD,marker='o',label=cSer[k])
        #Number of ZD amb
        axs[4,0].plot(Epo,nZD,marker='o',label=cSer[k])
        #Number of satellite
        axs[5,0].plot(Epo,nSat,marker='o',label=cSer[k])
        #Number of station
        axs[6,0].plot(Epo,nSta,marker='o',label=cSer[k])
    axs[0,0].legend(loc='upper right')
    axs[1,0].legend(loc='upper right')
    axs[2,0].legend(loc='upper right')
    axs[3,0].legend(loc='upper right')
    axs[4,0].legend(loc='upper right')
    axs[5,0].legend(loc='upper right')
    axs[6,0].legend(loc='upper right')

    strTmp=OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    import argparse

    # OutFilePrefix='D:/Code/PROJECT/WORK2019001/'

    # OutFileSuffix='bdeci0.pdf'
    # PlotAmbRoundingRegion0(OutFilePrefix,OutFileSuffix)

    # OutFileSuffix='bdeci1.pdf'
    # PlotAmbRoundingRegion1(OutFilePrefix,OutFileSuffix)

    # for Sig in np.linspace(1e-4,0.15,num=150000,endpoint=False,dtype=np.float64):
    #     Q0,T=bdeci1(0.14,Sig,0.15,0.15,50)
    #     # if abs(T/Q0-1000)<=1:
    #     #     print('{:>7.5f} {:>19.3f}'.format(Sig,T/Q0))

    #     print('{:>7.5f} {:>19.3f}'.format(Sig,T/Q0))

    #     # if abs(Q0*100-0.1)<=0.05:
    #     #     print('{:>7.5f} {:>19.3f}'.format(Sig,Q0*100))

    #     # print('{:>7.5f} {:>19.3f}'.format(Sig,Q0*100))

    # fList=[]
    # fList.append(r'D:/Code/PROJECT/WORK2019001/zdwl_2019001')

    # cSat=['ALL']
    # cSta=['ALL']
    # cAmb=['AMBC13']
    # OutFileSuffix='ZDWL.pdf'
    # PlotZDWLAmb(fList,cSat,cSta,cAmb,False,OutFilePrefix,OutFileSuffix)

    fDDWL=r'Y:/PRO_2019001_2020366_WORK/C41/WORK2019335/ddwl_2019335'
    cSat=['ALL']
    cSta=['ALL']
    cAmb=['CC13']
    # OutFilePrefix=r'Y:/PRO_2019001_2020366_WORK/C41/WORK2019335'
    OutFilePrefix='Z:/PRO_2019001_2020366/C41/AMB/'
    OutFileSuffix='DDWLAmb_2019335'
    # PlotDDWLAmb1(fDDWL,cSat,cSta,cAmb,True,OutFilePrefix,OutFileSuffix)
    PlotDDWLAmb2(fDDWL,cSat,cSta,cAmb,True,OutFilePrefix,OutFileSuffix)
    fDDAmb=r'Y:/PRO_2019001_2020366_WORK/C41/WORK2019347/ddamb_2019347'
    # OutFileSuffix='DDAmb1_2019347'
    # PlotDDAmb1(fDDAmb,False,OutFilePrefix,OutFileSuffix)

    # InFilePrefix=r'Y:/PRO_2019001_2020366_WORK/C41/WORK2019???/'
    # fDDAmbList=glob.glob(InFilePrefix+'ddamb_2019???')
    # OutFileSuffix='DDAmb2'
    # PlotDDAmb2(fDDAmbList,OutFilePrefix,OutFileSuffix)

    # fList=[]
    # fList.append(r'D:/Code/PROJECT/WORK2019001/ddamb_2019001')
    # OutFileSuffix='DDAmb1.pdf'

    # cSer=[]
    # fList=[]
    # cSer.append('test3_55')
    # InFilePrefix=r'Y:/MGEX_POD_C_B1I+B3I_BDS2+BDS3_ISL_test3_55/2019/AMBCON_POD/'
    # fList.append(glob.glob(InFilePrefix+'ambcon_*'))
    # cSer.append('test3')
    # InFilePrefix=r'Y:/MGEX_POD_C_B1I+B3I_BDS2+BDS3_ISL_test3/2019/AMBCON_POD'
    # fList.append(glob.glob(InFilePrefix+'ambcon_*'))
    # cSer.append('test4_55')
    # InFilePrefix=r'Y:/MGEX_POD_C_B1I+B3I_BDS2+BDS3_ISL_test4_55/2019/AMBCON_POD/'
    # fList.append(glob.glob(InFilePrefix+'ambcon_*'))
    # cSer.append('test4')
    # InFilePrefix=r'Y:/MGEX_POD_C_B1I+B3I_BDS2+BDS3_ISL_test4/2019/AMBCON_POD/'
    # fList.append(glob.glob(InFilePrefix+'ambcon_*'))
    # OutFilePrefix='D:/Code/PROJECT/WORK_ISL/'
    # OutFileSuffix='Ambcon.pdf'
    # PlotAmbCon(cSer,fList,OutFilePrefix,OutFileSuffix)
