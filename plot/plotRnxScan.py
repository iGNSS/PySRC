#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Plot some results from scaning RINEX observation files on MGEX data center(s)
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

# Local application/library specific imports
from PySRC.miscellaneous import GNSSTime

def PlotStaNum0(cSrc,fPathSrc,MJD0,nDay,OutFilePrefix,OutFileSuffix):
    '''
    Plot the number of stations from different Data Centers that can
    observe each satellite system

        cSrc --- List of Data Centers
    fPathSrc --- List of path for the scan result files for each DC

    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # ANY, GPS, BDS, GAL, GLO
    fig,axs=plt.subplots(5,1,sharex='col',sharey='col',squeeze=False,figsize=(14,2*5))
    fig.subplots_adjust(hspace=0.1)

    nSrc=len(cSrc)
    for i in range(nSrc):
        # Number of stations for each system in each file
        nSta=np.zeros((nDay,5),dtype=np.uint16)
        x=[]
        for j in range(nDay):
            #Get the date of this file from the file name
            MJD=MJD0+j
            YYYY,DOY=GNSSTime.mjd2doy(MJD)
            MO,DD=GNSSTime.doy2dom(YYYY,DOY)
            x.append(datetime.datetime(YYYY,MO,DD))
            fScan=os.path.join(fPathSrc[i],'obs_{:4d}{:03d}_'.format(YYYY,DOY)+cSrc[i])
            if not os.path.isfile(fScan):
                print(fScan+' does not exist!')
                continue
            # Station list for each system
            cSta=[[],[],[],[],[]]
            with open(fScan,mode='rt') as fOb:
                for cLine in fOb:
                    if cLine[0:4] == 'MET=':
                        # Start of a station within the file
                        cTmp=cLine[4:8].upper()
                        if cTmp in cSta[0]:
                            print('Duplicated in '+fScan+' '+cLine[0:8])
                        else:
                            cSta[0].append(cTmp)
                    elif cLine[0:3] == 'STP':
                        # Start of a GNSS with the station
                        if cLine[9:12] == 'GXX':
                            if cLine[4:8] in cSta[1]:
                                print('Duplicated in '+fScan+' '+cLine[0:12])
                            else:
                                cSta[1].append(cLine[4:8])
                        elif cLine[9:12] == 'CXX':
                            if cLine[4:8] in cSta[2]:
                                print('Duplicated in '+fScan+' '+cLine[0:12])
                            else:
                                cSta[2].append(cLine[4:8])
                        elif cLine[9:12] == 'EXX':
                            if cLine[4:8] in cSta[3]:
                                print('Duplicated in '+fScan+' '+cLine[0:12])
                            else:
                                cSta[3].append(cLine[4:8])
                        elif cLine[9:12] == 'RXX':
                            if cLine[4:8] in cSta[4]:
                                print('Duplicated in '+fScan+' '+cLine[0:12])
                            else:
                                cSta[4].append(cLine[4:8])
                        else:
                            continue
                    else:
                        # Skip other records
                        continue
            nSta[j,0]=len(cSta[0])
            nSta[j,1]=len(cSta[1])
            nSta[j,2]=len(cSta[2])
            nSta[j,3]=len(cSta[3])
            nSta[j,4]=len(cSta[4])
        cSys=['ANY','GPS','BDS','GAL','GLO']
        for j in range(5):
            axs[j,0].plot(x,nSta[:,j],lw=3,label=cSrc[i])
            if i==0:
                axs[j,0].text(0.01,0.99,cSys[j],transform=axs[j,0].transAxes,
                              ha='left',va='top',family='Arial',size=16,weight='bold')
                axs[j,0].set_ylabel('Sta #',fontname='Arial',fontsize=16)
                for tl in axs[j,0].get_yticklabels():
                    tl.set_fontname('Arial'); tl.set_fontsize(14)
            if i==(nSrc-1):
                axs[j,0].legend(loc='upper right',framealpha=0.6,bbox_to_anchor=(1.0,1.0),
                                prop={'family':'Arial','size':14},borderaxespad=0.1,
                                columnspacing=1.0,handlelength=1.0,handletextpad=0.4)
    YR=mdates.YearLocator()
    Mo=mdates.MonthLocator()
    for j in range(5):
        axs[j,0].xaxis.set_major_locator(YR)
        axs[j,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axs[j,0].xaxis.set_minor_locator(Mo)
        axs[j,0].xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    for tl in axs[j,0].get_xticklabels(which='major'):
        tl.set_fontname('Arial'); tl.set_fontsize(14); tl.set_fontweight('bold')
    for tl in axs[j,0].get_xticklabels(which='minor'):
        tl.set_fontname('Arial'); tl.set_fontsize(10)

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    strTmp=OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def GetObsTypSat1(fList,cSat):
    '''
    Get the obs types for a specific satellite in each file

    fList ---
    cSat  --- specified satellite
    '''

    # All obs typs
    cObsTypAll=[]
    # Obs types in each file
    cObsTyp=[]
    for i in range(len(fList)):
        cObsTyp.append([])
        with open(fList[i],mode='rt') as fOb:
            cLine=fOb.readline()
            while cLine:
                if cLine[0:3] == 'STP' and cLine[9:12] == cSat[0:1]+'XX':
                    #Get the obs type list
                    cTyp=cLine[12:].split()
                    cLine=fOb.readline()
                    while cLine:
                        if cLine[0:3] != 'STO':
                            # End for this system
                            break
                        if cLine[9:12] != cSat:
                            cLine=fOb.readline()
                            continue
                        if '*****' in cLine:
                            print(fList[i]+' '+cLine)
                            cLine=fOb.readline()
                            continue
                        nObs=cLine[12:].split()
                        for j in range(len(cTyp)):
                            if int(nObs[j])<=0:
                                continue
                            if cTyp[j] in cObsTyp[i]:
                                continue
                            cObsTyp[i].append(cTyp[j])
                            if cTyp[j] not in cObsTypAll:
                                cObsTypAll.append(cTyp[j])
                        cLine=fOb.readline()
                else:
                    cLine=fOb.readline()
            cObsTyp[i].sort()
    cObsTypAll.sort()
    return cObsTypAll,cObsTyp

def WriteObsTypSat1(fList,cSatList,OutFilePrefix,OutFileSuffix):
    '''
    Get the obs types for specific satellite list

       fList  ---
    cSatList  --- specified satellite list
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    fOut=open(OutFilePrefix+os.path.splitext(OutFileSuffix)[0],'w')

    for i in range(len(cSatList)):
        cObsTypAll,cObsTyp=GetObsTypSat1(fList,cSatList[i])
        if len(cObsTypAll)==0:
            continue
        cTmp=''
        for j in range(len(cObsTypAll)):
            if j==0:
                cTmp=cObsTypAll[j]
            else:
                cTmp=cTmp+'+'+cObsTypAll[j]
        fOut.write(cSatList[i]+' {:>2d} '.format(len(cObsTypAll))+cTmp+'\n')
    fOut.close()


def GetStaNum1(fList,cSat,cObs,mObs):
    '''
    Get the stations that can observe a specific satellite
    with a specific min number of obs on specified obs types in each file

    fList ---
    cSat  --- specified satellite
    cObs  --- specified observation type list
    mObs  --- min epoch numbers for each observation type
    '''

    if len(cObs) != len(mObs):
        sys.exit('GetStaNum1: Unmatched lists of cObs and mObs!')

    x=[]; rMJD=[]; cSta=[]
    for i in range(len(fList)):
        #Get the date of this file from the file name
        YYYY=int(os.path.basename(fList[i])[4:8])
        DOY=int(os.path.basename(fList[i])[8:11])
        MO,DD=GNSSTime.doy2dom(YYYY,DOY)
        x.append(datetime.datetime(YYYY,MO,DD))
        rMJD.append(GNSSTime.doy2mjd(YYYY,DOY))
        #List of selected stations for this file
        cSta.append([])
        with open(fList[i],mode='rt') as fOb:
            cLine=fOb.readline()
            while cLine:
                if cLine[0:3] == 'STP' and cLine[9:12] == cSat[0:1]+'XX':
                    # index of the specified obs type in the file
                    iObs=np.zeros(len(cObs),dtype=np.int8)
                    #Get the observation type list
                    cTyp=cLine[12:].split()
                    # whether this is a valid station
                    lValid=True
                    for j in range(len(cObs)):
                        if cObs[j] not in cTyp:
                            # If miss any required obs type
                            lValid=False
                            break
                        iObs[j]=cTyp.index(cObs[j])
                    if lValid:
                        cLine=fOb.readline()
                        while cLine:
                            if cLine[0:3] != 'STO':
                                lValid=False
                                break
                            if cLine[9:12] != cSat:
                                cLine=fOb.readline()
                                continue
                            if '*****' in cLine:
                                print(fList[i]+' '+cLine)
                                cLine=fOb.readline()
                                continue
                            nObs=cLine[12:].split()
                            for j in range(len(cObs)):
                                if int(nObs[iObs[j]])<mObs[j]:
                                    # If non-enough obs number for any obs type
                                    lValid=False
                                    break
                            if lValid:
                                cSta[i].append(cLine[4:8])
                            cLine=fOb.readline()
                            break
                    else:
                        cLine=fOb.readline()
                else:
                    cLine=fOb.readline()
        cSta[i].sort()
    return x,rMJD,cSta

def GetStaNum2(fList,cSatList,cObs,mObs,mSat):
    '''
    Get the stations that can observe a min set of specific satellites
    with a specific min number of obs on specified obs types in each file

    fList ---
 cSatList --- specified satellite list
    cObs  --- specified observation type list
    mObs  --- min epoch numbers for each observation type
    mSat  --- min satellite numbers for a station to be selected
    '''

    if len(cObs) != len(mObs):
        sys.exit('GetStaNum2: Unmatched lists of cObs and mObs!')

    # Firstly, get the daily valid stations for each satellite
    cSta1=[]
    for i in range(len(cSatList)):
        x,rMJD,cSta0=GetStaNum1(fList,cSatList[i],cObs,mObs)
        cSta1.append(cSta0)

    # Secondly, get the daily union of station set among all satellites
    # for a daily station set that can observe any of required satellites
    cSta0=[]
    for i in range(len(fList)):
        cSta0.append([])
        for j in range(len(cSatList)):
            for k in range(len(cSta1[j][i])):
                if cSta1[j][i][k] in cSta0[i]:
                    continue
                cSta0[i].append(cSta1[j][i][k])
    # Thirdly, check daily for each station in the union set to see
    # how many times it appears in those satellite-specific station sets
    x=[]; rMJD=[]; cSta=[]
    for i in range(len(fList)):
        #Get the date of this file from the file name
        YYYY=int(os.path.basename(fList[i])[4:8])
        DOY=int(os.path.basename(fList[i])[8:11])
        MO,DD=GNSSTime.doy2dom(YYYY,DOY)
        x.append(datetime.datetime(YYYY,MO,DD))
        rMJD.append(GNSSTime.doy2mjd(YYYY,DOY))
        cSta.append([])
        #Check each station in this file
        for j in range(len(cSta0[i])):
            #Count the number of satellites observed by this station
            l=0
            #Check each satellite whether it was observed by this station
            for k in range(len(cSatList)):
                if cSta0[i][j] not in cSta1[k][i]:
                    continue
                l=l+1
                if l >= mSat:
                    #This station passes the check
                    cSta[i].append(cSta0[i][j])
                    break
        cSta[i].sort()
    # Sort along the date
    x0=[]; rMJD0=[]; cSta0=[]
    ind=np.argsort(rMJD)
    for i in range(ind.size):
        x0.append(x[ind[i]])
        rMJD0.append(rMJD[ind[i]])
        cSta0.append(cSta[ind[i]])
    return x0, rMJD0, cSta0

def PlotStaNum1(fList,cSatList,cObs,mObs,OutFilePrefix,OutFileSuffix):
    '''
    Plot the number of stations that could observe the specific satellites
    with a specific min number of obs on specified obs types in each (daily) file.

    For each satellite, a separate line will be plotted/stacked within the same axis.

    cSatList --- Specify the satellite
    cObs     --- specified observation type list
    mObs     --- min epoch numbers for each observation type
    '''

    if len(cObs) != len(mObs):
        sys.exit('PlotStaNum1: Unmatched lists of cObs and mObs!')
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nSat=len(cSatList)
    fig,axs=plt.subplots(1,1,squeeze=False,figsize=(12,nSat*0.5))

    # Max number of stations for each satellite, which decide the max y-axis limit
    nStaMax=300
    Yticks=[0]; Ytilab=['0']
    axs[0,0].set_ylim(bottom=0,top=nStaMax*nSat)
    for i in range(nSat):
        x,rMJD,cSta=GetStaNum1(fList,cSatList[i],cObs,mObs)
        nSta=np.zeros(len(fList)); nSta[:]=np.nan
        for j in range(len(fList)):
            if len(cSta[j]) == 0:
                continue
            nSta[j]=len(cSta[j])+i*nStaMax
        Yticks.append((i+1)*nStaMax)
        Ytilab.append('{:>3d}'.format(nStaMax))
        axs[0,0].plot(x,nSta,lw=3,label=cSatList[i])
        axs[0,0].text(0.01,1.0/nSat*(i+0.98),cSatList[i],transform=axs[0,0].transAxes,
                      ha='left',va='top',family='Arial',size=16)
    axs[0,0].set_ylabel('Sta #',fontname='Arial',fontsize=16)
    axs[0,0].set_yticks(Yticks,minor=False)
    axs[0,0].set_yticklabels(Ytilab,fontdict={'fontsize':14,'fontname':'Arial'})
    axs[0,0].grid(b=True,which='both',axis='y',color='darkgray',linestyle='--',
                  linewidth=0.8)
    YR=mdates.YearLocator()
    Mo=mdates.MonthLocator()
    axs[0,0].xaxis.set_major_locator(YR)
    axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    for tl in axs[0,0].get_xticklabels(which='major'):
        tl.set_fontname('Arial'); tl.set_fontsize(14); tl.set_fontweight('bold')
    axs[0,0].xaxis.set_minor_locator(Mo)
    axs[0,0].xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    for tl in axs[0,0].get_xticklabels(which='minor'):
        tl.set_fontname('Arial'); tl.set_fontsize(10)

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    strTmp=OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotStaNum2(fList,cSatList,cObsSer,mObsSer,OutFilePrefix,OutFileSuffix):
    '''
    Plot the number of stations that could observe the specific satellites
    with a specific min number of obs on specified obs type in each (daily) file.

    For each satellite, a separate axis with multi-series would be plotted.

    cSatList --- Specify the satellites
    cObsSer  --- Specify the observation types for each series
    mObsSer  ---
    '''

    nSer=len(cObsSer)
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nSat=len(cSatList)
    fig,axs=plt.subplots(nSat,1,sharex='col',squeeze=False,figsize=(12,nSat*2))
    fig.subplots_adjust(hspace=0.1)

    for iSer in range(nSer):
        for iSat in range(nSat):
            x,rMJD,cSta=GetStaNum1(fList,cSatList[iSat],cObsSer[iSer],mObsSer[iSer])
            nSta=np.zeros(len(fList)); nSta[:]=np.nan
            for i in range(len(fList)):
                if len(cSta[i]) == 0:
                    continue
                nSta[i]=len(cSta[i])
            for i in range(len(cObsSer[iSer])):
                if i == 0:
                    cTmp=cObsSer[iSer][i]
                else:
                    cTmp=cTmp+'+'+cObsSer[iSer][i]
            axs[iSat,0].plot(x,nSta,lw=3,label=cTmp)
            if iSer == (nSer-1):
                axs[iSat,0].set_ylim(bottom=0,top=300)
                axs[iSat,0].set_ylabel('Sta #',fontname='Arial',fontsize=16)
                for tl in axs[iSat,0].get_yticklabels():
                    tl.set_fontname('Arial'); tl.set_fontsize(14)
                axs[iSat,0].text(0.01,0.98,cSatList[iSat],transform=axs[iSat,0].transAxes,
                                 ha='left',va='top',family='Arial',size=16)
                axs[iSat,0].grid(b=True,which='both',axis='y',color='darkgray',
                                 linestyle='--',linewidth=0.8)
                axs[iSat,0].legend(ncol=nSer,loc='upper center',bbox_to_anchor=(0.5,1.0),
                                   framealpha=0.3,prop={'family':'Arial','size':14})
    YR=mdates.YearLocator()
    Mo=mdates.MonthLocator()
    axs[nSat-1,0].xaxis.set_major_locator(YR)
    axs[nSat-1,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    for tl in axs[nSat-1,0].get_xticklabels(which='major'):
        tl.set_fontname('Arial'); tl.set_fontsize(14); tl.set_fontweight('bold')
    axs[nSat-1,0].xaxis.set_minor_locator(Mo)
    axs[nSat-1,0].xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    for tl in axs[nSat-1,0].get_xticklabels(which='minor'):
        tl.set_fontname('Arial'); tl.set_fontsize(10)

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    strTmp=OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotStaNum3(fList,cSatList,cObsSer,mObsSer,mSatSer,lWriteOut,
                OutFilePrefix,OutFileSuffix):
    '''
    Plot the number of stations that could observe a min set of specified satellites
    with a specific min number of obs on specified obs types in each (daily) file.

    Multi-series would be plotted within the same axis.

 cSatList   --- specified satellite list
  cObsSer   --- specified observation type list for each series
  mObsSer   --- min epoch numbers for each observation type for each series
  mSatSer   --- min satellite numbers for a station to be selected for each series

  lWriteOut --- Whether write out the station lists for each file
    '''

    nSer=len(cObsSer)
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    fig,axs=plt.subplots(1,1,squeeze=False,figsize=(12,3))

    axs[0,0].set_ylim(bottom=0,top=300)
    axs[0,0].set_ylabel('Station Number',fontname='Arial',fontsize=16)

    for iSer in range(nSer):
        x,rMJD,cSta=GetStaNum2(fList,cSatList,cObsSer[iSer],mObsSer[iSer],mSatSer[iSer])
        nSta=np.zeros(len(fList)); nSta[:]=np.nan
        if lWriteOut:
            # Check the directories of output station list files
            for i in range(len(cObsSer[iSer])):
                if i == 0:
                    cTmp=cObsSer[iSer][i]
                else:
                    cTmp=cTmp+cObsSer[iSer][i]
            fPathSer=os.path.join(OutFilePrefix,cTmp)
            if not os.path.isdir(fPathSer):
                os.makedirs(fPathSer)
        for i in range(len(fList)):
            if len(cSta[i]) == 0:
                continue
            nSta[i]=len(cSta[i])
            if not lWriteOut:
                continue
            # Write out the station list
            fSta=os.path.join(fPathSer,os.path.basename(fList[i])[-13:-6]+'_StaList')
            with open(fSta,'w') as fOb:
                fOb.write('# {:>3d} stations'.format(nSta[i])+'\n')
                for j in range(nSta[i]):
                    fOb.write(' '+cSta[i][j]+'\n')
        for i in range(len(cObsSer[iSer])):
            if i == 0:
                cTmp=cObsSer[iSer][i]
            else:
                cTmp=cTmp+'+'+cObsSer[iSer][i]
        axs[0,0].plot(x,nSta,lw=3,label=cTmp)
    axs[0,0].grid(which='both',axis='y',c='darkgray',ls='--',lw=0.8)
    axs[0,0].legend(ncol=nSer,loc='upper center',bbox_to_anchor=(0.5,1.0),
                    framealpha=0.3,prop={'family':'Arial','size':14})
    for tl in axs[0,0].get_yticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)
    YR=mdates.YearLocator()
    Mo=mdates.MonthLocator()
    axs[0,0].xaxis.set_major_locator(YR)
    axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    for tl in axs[0,0].get_xticklabels(which='major'):
        tl.set_fontname('Arial'); tl.set_fontsize(12); tl.set_fontweight('bold')
    axs[0,0].xaxis.set_minor_locator(Mo)
    axs[0,0].xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    for tl in axs[0,0].get_xticklabels(which='minor'):
        tl.set_fontname('Arial'); tl.set_fontsize(10)

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    strTmp=OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotStaNum4(fList,cSatListSer,cObsSer,mObsSer,mSatSer,lWriteOut,OutFilePrefix,OutFileSuffix):
    '''
    Plot the number of stations that could simultaneously observe a min set of
    different specified satellite lists with a specific min number of obs on specified
    obs types in each (daily) file.

    cSatListSer --- Specified satellite lists for each series
    cObsSer     --- Specified obs type lists for each series
    mObsSer     --- Specified min number of obs for each series
    mSatSer     --- Specified min number of sat for each series

    lWriteOut   --- Whether write out the station lists for each file

    '''

    nSer=len(cObsSer)
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    cSta=[]
    for iSer in range(nSer):
        # Get the required station sets for each specified satellite list
        x,rMJD,cSta1=GetStaNum2(fList,cSatListSer[iSer],cObsSer[iSer],mObsSer[iSer],mSatSer[iSer])
        if iSer==0:
            for i in range(len(fList)):
                cSta.append(cSta1[i])
        else:
            # Get stations that exist in both current and last station set
            for i in range(len(fList)):
                #Common station set until now
                cSta0=cSta[i]; cSta[i]=[]
                for j in range(len(cSta1[i])):
                    if cSta1[i][j] not in cSta0:
                        continue
                    #Add to the common station list
                    cSta[i].append(cSta1[i][j])

    #Count the number of stations
    nSta=np.zeros(len(fList)); nSta[:]=np.nan
    for i in range(len(fList)):
        if len(cSta[i]) == 0:
            continue
        nSta[i]=len(cSta[i])
        if not lWriteOut:
            continue
        # Write out the station list
        fSta=os.path.join(OutFilePrefix,os.path.basename(fList[i])[-13:-6]+'_StaList')
        with open(fSta,'w') as fOb:
            fOb.write('# {:>3d} stations'.format(nSta[i])+'\n')
            for j in range(nSta[i]):
                fOb.write(' '+cSta[i][j]+'\n')

    fig,axs=plt.subplots(1,1,squeeze=False,figsize=(12,4))

    axs[0,0].set_ylim(bottom=0,top=300)
    axs[0,0].set_ylabel('Sta #',fontname='Arial',fontsize=16)
    for tl in axs[0,0].get_yticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)

    axs[0,0].plot(x,nSta,lw=3)
    axs[0,0].grid(b=True,which='both',axis='y',color='darkgray',linestyle='--',
                  linewidth=0.8)
    YR=mdates.YearLocator()
    Mo=mdates.MonthLocator()
    axs[0,0].xaxis.set_major_locator(YR)
    axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    for tl in axs[0,0].get_xticklabels(which='major'):
        tl.set_fontname('Arial'); tl.set_fontsize(14); tl.set_fontweight('bold')
    axs[0,0].xaxis.set_minor_locator(Mo)
    axs[0,0].xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    for tl in axs[0,0].get_xticklabels(which='minor'):
        tl.set_fontname('Arial'); tl.set_fontsize(10)

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    strTmp=OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotStaNum5(cSer,fPathSer,MJD0,nDay,OutFilePrefix,OutFileSuffix):
    '''
    Plot the number of stations in station list file for each series

    fPathSer --- the path list for all series
    '''

    nSer=len(cSer)
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    fig,axs=plt.subplots(1,1,squeeze=False,figsize=(12,4))
    axs[0,0].set_ylim(bottom=0,top=300)
    # axs[0,0].set_prop_cycle(color =['r','r','r','r','r','r','r','r','r',
    #                                 'g','g','g','g','g','g','g','g','g',
    #                                 'b','b','b','b','b','b','b','b','b'],
    #                         marker=['.','v','^','<','>','*','x','d','X',
    #                                 '.','v','^','<','>','*','x','d','X',
    #                                 '.','v','^','<','>','*','x','d','X'])

    for iSer in range(nSer):
        x=[]; nSta=[]
        for i in range(nDay):
            MJD=MJD0+i
            YYYY,DOY=GNSSTime.mjd2doy(MJD)
            MO,DD=GNSSTime.doy2dom(YYYY,DOY)
            x.append(datetime.datetime(YYYY,MO,DD))
            fSta=os.path.join(fPathSer[iSer],'{:4d}{:03d}_Stalist'.format(YYYY,DOY))
            if not os.path.isfile(fSta):
                print(fSta+' does not exist!')
                nSta.append(np.nan)
            else:
                nSta.append(0)
                with open(fSta,mode='rt') as fOb:
                    for cLine in fOb:
                        if cLine[0:1] != ' ':
                            continue
                        if len(cLine) < 5:
                            continue
                        nSta[i]=nSta[i]+1
        axs[0,0].plot(x,nSta,lw=3,label=cSer[iSer])
    axs[0,0].set_ylabel('Sta #',fontname='Arial',fontsize=16)
    for tl in axs[0,0].get_yticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)

    axs[0,0].grid(b=True,which='both',axis='y',color='darkgray',linestyle='--',
                  linewidth=0.8)
    axs[0,0].legend(ncol=4,loc='upper center',bbox_to_anchor=(0.5,1.0),framealpha=0.3,
                    prop={'family':'Arial','size':14},borderaxespad=0.1,
                    columnspacing=1.0,handlelength=1.0,handletextpad=0.4)

    YR=mdates.YearLocator()
    Mo=mdates.MonthLocator()
    axs[0,0].xaxis.set_major_locator(YR)
    axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    for tl in axs[0,0].get_xticklabels(which='major'):
        tl.set_fontname('Arial'); tl.set_fontsize(14); tl.set_fontweight('bold')
    axs[0,0].xaxis.set_minor_locator(Mo)
    axs[0,0].xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    for tl in axs[0,0].get_xticklabels(which='minor'):
        tl.set_fontname('Arial'); tl.set_fontsize(10)

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    strTmp=OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotStaNum6(fPathSNX,fPathSta,MJD0,nDay,lWriteOut,OutFilePrefix,OutFileSuffix):
    '''
    Plot the number of stations that exist simultaneously in the
    station list file and the IGS SINEX file

    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    if lWriteOut:
        fOutPath=os.path.join(OutFilePrefix,'CommonWithSNX')
        if not os.path.isdir(fOutPath):
            os.makedirs(fOutPath)

    x=[]; nStaLst=[]; nStaSNX=[]
    for i in range(nDay):
        MJD=MJD0+i
        YYYY,DOY=GNSSTime.mjd2doy(MJD)
        WK,WKD=GNSSTime.doy2wkd(YYYY,DOY)
        YR=GNSSTime.year2yr(YYYY)
        MO,DD=GNSSTime.doy2dom(YYYY,DOY)
        x.append(datetime.datetime(YYYY,MO,DD))

        fSNX=os.path.join(fPathSNX,'{:04d}'.format(WK),
                          'igs{:02d}P{:04d}{:01d}.ssc'.format(YR,WK,WKD))
        fSta=os.path.join(fPathSta,'{:4d}{:03d}_Stalist'.format(YYYY,DOY))

        if not os.path.isfile(fSta):
            print(fSta+' does not exist!')
            nStaLst.append(np.nan); nStaSNX.append(np.nan)
        else:
            nStaLst.append(0); cStaLst=[]
            # Read the station list file
            with open(fSta,mode='rt') as fOb:
                for cLine in fOb:
                    if cLine[0:1] != ' ':
                        continue
                    if len(cLine) < 4:
                        continue
                    cStaLst.append(cLine[1:5])
                    nStaLst[i]=nStaLst[i]+1
            if not os.path.isfile(fSNX):
                print(fSNX+' does not exist!')
                nStaSNX.append(np.nan)
            else:
                nStaSNX.append(0); cStaSNX=[]
                #Station list from SNX file
                with open(fSNX,mode='rt',encoding='utf_8') as fOb:
                    lBeg=False
                    for cLine in fOb:
                        if cLine[0:8] == '+SITE/ID':
                            lBeg=True
                        elif cLine[0:8] == '-SITE/ID':
                            exit
                        elif lBeg and cLine[0:1]==' ':
                            if (cLine[1:5] in cStaLst) and (cLine[1:5] not in cStaSNX):
                                cStaSNX.append(cLine[1:5])
                                nStaSNX[i]=nStaSNX[i]+1
                if lWriteOut:
                    fOut=os.path.join(fOutPath,'{:04d}{:03d}_StaListComWithSNX'.format(YYYY,DOY))
                    with open(fOut,'w') as fOb:
                        fOb.write('# {:>3d} common stations'.format(nStaSNX[i])+'\n')
                        for j in range(nStaSNX[i]):
                            fOb.write(' '+cStaSNX[j]+'\n')

    fig,axs=plt.subplots(1,1,squeeze=False,figsize=(12,4))

    axs[0,0].set_ylim(bottom=0,top=300)
    axs[0,0].plot(x,nStaLst,lw=3,label='In station list')
    axs[0,0].plot(x,nStaSNX,lw=3,label='Common with SNX')

    axs[0,0].set_ylabel('Sta #',fontname='Arial',fontsize=16)
    for tl in axs[0,0].get_yticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)
    axs[0,0].legend(ncol=2,loc='upper center',bbox_to_anchor=(0.5,1.0),framealpha=0.3,
                    prop={'family':'Arial','size':14},borderaxespad=0.1,
                    columnspacing=1.0,handlelength=1.0,handletextpad=0.4)

    axs[0,0].grid(b=True,which='both',axis='y',color='darkgray',linestyle='--',
                  linewidth=0.8)
    YR=mdates.YearLocator()
    Mo=mdates.MonthLocator()
    axs[0,0].xaxis.set_major_locator(YR)
    axs[0,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    for tl in axs[0,0].get_xticklabels(which='major'):
        tl.set_fontname('Arial'); tl.set_fontsize(14); tl.set_fontweight('bold')
    axs[0,0].xaxis.set_minor_locator(Mo)
    axs[0,0].xaxis.set_minor_formatter(mdates.DateFormatter('%b'))
    for tl in axs[0,0].get_xticklabels(which='minor'):
        tl.set_fontname('Arial'); tl.set_fontsize(10)

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    strTmp=OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotStaNum7(fScan,fSta,lPrint,OutFilePrefix,OutFileSuffix):
    '''
    Plot the number stations equipped with each Receiver and Antenna

    fSta --- Specified station list. If given, only listed stations would be
             checked.
   lPrint--- Whether print the station list belonging to each antenna type
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    cStaLst=[]
    if not os.path.isfile(fSta):
        print(fSta+' does not exist!')
    else:
        with open(fSta,mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:1]!=' ':
                    continue
                if len(cLine) < 5:
                    continue
                if cLine[1:5].upper() in cStaLst:
                    continue
                else:
                    cStaLst.append(cLine[1:5].upper())

    cSta=[]; cRec=[]; cRecSta=[]; cAnt=[]; cAntNum=[]; cAntSta=[]
    with open(fScan,mode='rt') as fOb:
        for cLine in fOb:
            if cLine[0:4] != 'MET=':
                continue
            cWords=cLine[4:].split(sep='=')
            # Sta name, only the first 4-ch of the marker name
            Sta=cWords[0].lstrip().rstrip()[0:4].upper()
            if (len(cStaLst)>0) and (Sta not in cStaLst):
                continue
            if Sta in cSta:
                print('Duplicated in '+fScan+' '+cLine[0:8])
            else:
                cSta.append(Sta)
            # Rec type
            Rec=cWords[4].lstrip().rstrip().upper()
            if Rec not in cRec:
                cRec.append(Rec); cRecSta.append([])
            iRec=cRec.index(Rec); cRecSta[iRec].append(Sta)
            # Ant Num && type
            AntNum=cWords[6].lstrip().rstrip().upper()
            Ant=cWords[7].lstrip().rstrip().upper()
            # For antenna, only the type name is used to discriminate
            if Ant not in cAnt:
                cAnt.append(Ant); cAntNum.append(AntNum)
                cAntSta.append([])
            iAnt=cAnt.index(Ant); cAntSta[iAnt].append(Sta)
    print('{:>3d} Stations'.format(len(cSta)))

    fig,axs=plt.subplots(2,1,squeeze=False,figsize=(12,10))
    fig.subplots_adjust(hspace=0.4)

    # Station number of each rec type
    print('{:<3s} {:<20s} {:<20s}'.format('Num','RecType','StaNum'))
    nRec=len(cRec); cRec0=cRec.copy(); cRec0.sort()
    x=np.arange(nRec); y=np.zeros(nRec,dtype=np.int32)
    for i in range(nRec):
        y[i]=len(cRecSta[cRec.index(cRec0[i])])
        print('{:>03d} {:<20s} {:>3d}'.format(i+1,cRec0[i],y[i]))
    print(' ')

    axs[0,0].set_xlim(left=-1,right=nRec)
    w=1/(1+1)
    axs[0,0].bar(x+(0-1/2)*w,y[:],w,align='edge')
    axs[0,0].grid(b=True,which='both',axis='y',color='darkgray',linestyle='--',
                  linewidth=0.8)
    axs[0,0].set_axisbelow(True)
    axs[0,0].set_xlabel('Receiver Type',fontname='Arial',fontsize=16)
    axs[0,0].set_xticks(x)
    axs[0,0].set_xticklabels(cRec0,rotation=330,c='darkblue',
            fontdict={'fontsize':8,'fontname':'Arial',
                      'horizontalalignment':'left'})
    axs[0,0].set_ylabel('Sta #',fontname='Arial',fontsize=16)
    for tl in axs[0,0].get_yticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)

    # Station number of each ant type
    print('{:<3s} {:<20s} {:<20s}'.format('Num','AntType','StaNum'))
    nAnt=len(cAnt); cAnt0=cAnt.copy(); cAnt0.sort()
    x=np.arange(nAnt); y=np.zeros(nAnt,dtype=np.int32)
    for i in range(nAnt):
        j=cAnt.index(cAnt0[i]); y[i]=len(cAntSta[j])
        strTmp='{:>03d} {:<20s} {:<20s} {:>3d}'.format(i+1,cAnt0[i],cAntNum[j],y[i])
        if lPrint:
            for k in range(y[i]):
                strTmp=strTmp+' '+cAntSta[j][k]
        print(strTmp)

    axs[1,0].set_xlim(left=-1,right=nAnt)
    w=1/(1+1)
    axs[1,0].bar(x+(0-1/2)*w,y[:],w,align='edge')
    axs[1,0].grid(b=True,which='both',axis='y',color='darkgray',linestyle='--',
                  linewidth=0.8)
    axs[1,0].set_axisbelow(True)
    axs[1,0].set_xlabel('Antenna Type',fontname='Arial',fontsize=16)
    axs[1,0].set_xticks(x)
    axs[1,0].set_xticklabels(cAnt0,rotation=330,c='darkblue',
            fontdict={'fontsize':8,'fontname':'Arial',
                      'horizontalalignment':'left'})
    axs[1,0].set_ylabel('Sta #',fontname='Arial',fontsize=16)
    for tl in axs[1,0].get_yticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    strTmp=OutFilePrefix+OutFileSuffix+'.svg'
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

    cSrc=[]; fList=[]
    cSatListSer=[]; cObsSer=[]; mObsSer=[]; mSatSer=[]
    # InFilePrefix=r'Y:/MGEX/'
    # InFilePrefix=r'D:/Code/PROJECT/WORK_BDSOBS/'
    # cSrc.append('cddis'); fList.append(r'Y:/MGEX/cddis')
    # cSrc.append('bkg'); fList.append(r'Y:/MGEX/bkg')
    # cSrc.append('ign'); fList.append(r'Y:/MGEX/ign')

    OutFilePrefix=os.path.join(cWrkPre0,r'MGEX/StaList/2019001_2020366/')
    # OutFileSuffix='StaNum0'
    # PlotStaNum0(cSrc,fList,58484,730,OutFilePrefix,OutFileSuffix)

    # InFilePrefix=r'Y:/MGEX/2019/'
    InFilePrefix=os.path.join(cWrkPre0,r'MGEX/cddis/')
    fList=glob.glob(InFilePrefix+'obs_*_cddis')
    # fList=glob.glob(InFilePrefix+'scan_*')

    # cSatList=['C01','C02','C03','C04','C05','C06','C07','C08','C09','C10',
    #           'C11','C12','C13','C14','C15','C16','C17','C18','C19','C20',
    #           'C21','C22','C23','C24','C25','C26','C27','C28','C29','C30',
    #           'C31','C32','C33','C34','C35','C36','C37']

    # cSatList=['C01','C02','C03','C04','C05','C06','C07','C08','C09','C10',
    #           'C11','C12','C13','C14','C15','C16','C17','C18','C19','C20',
    #           'C21','C22','C23','C24','C25','C26','C27','C28','C29','C30',
    #           'C31','C32','C33','C34','C35','C36','C37','C38','C39','C40',
    #           'C41','C42','C43','C44','C45','C46','C47','C48','C49','C50',
    #           'C51','C52','C53','C54','C55','C56','C57','C58','C59','C60',
    #           'C61','C62','C63','C64','C65','C66','C67','C68','C69','C70']
    # OutFileSuffix='ObsTypSat'
    # WriteObsTypSat1(fList,cSatList,OutFilePrefix,OutFileSuffix)

    # BDS-2
    cSatList=['C01','C02','C03','C04','C05','C06','C07','C08','C09','C10',
              'C11','C12','C13','C14','C15','C16','C17']
    cSatListSer.append(cSatList)
    # cObs=['L2I','L6I','C2I','C6I']
    cObs=['L2I','L7I','C2I','C7I']
    mObs=[500,500,500,500]; mSat=1
    cObsSer.append(cObs); mObsSer.append(mObs); mSatSer.append(mSat)

    # # BDS-3
    # cSatList=['C18','C19','C20',
    #           'C21','C22','C23','C24','C25','C26','C27','C28','C29','C30',
    #           'C31','C32','C33','C34','C35','C36','C37','C38','C39','C40',
    #           'C41','C42','C43','C44','C45','C46','C47','C48','C49','C50',
    #           'C51','C52','C53','C54','C55','C56','C57','C58','C59','C60',
    #           'C61']
    cSatListSer.append(cSatList)
    cObs=['L2I','L6I','C2I','C6I']
    mObs=[500,500,500,500]; mSat=1
    cObsSer.append(cObs); mObsSer.append(mObs); mSatSer.append(mSat)

    # OutFileSuffix='StaNum_BDS3_L2IL7IC2IC7I.pdf'
    # PlotStaNum1(fList,cSatList,cObs,mObs,OutFilePrefix,OutFileSuffix)

    # OutFileSuffix='StaNum_BDS3.pdf'
    # PlotStaNum2(fList,cSatList,cObsSer,mObsSer,OutFilePrefix,OutFileSuffix)

    OutFileSuffix='StaNum_BDS2_Min1'
    PlotStaNum3(fList,cSatList,cObsSer,mObsSer,mSatSer,False,OutFilePrefix,OutFileSuffix)

    # OutFilePrefix=r'Y:/MGEX/StaList/BDS2_L2IL7I_BDS3_L2IL6I/'
    # OutFileSuffix='StaNum_BDS2_L2IL7I_BDS3_L2IL6I'
    # PlotStaNum4(fList,cSatListSer,cObsSer,mObsSer,mSatSer,False,OutFilePrefix,OutFileSuffix)


    # cSer=['BDS-2_L2IL6I',
    #       'BDS-2_L2IL7I',
    #       'BDS-3_L2IL6I',
    #       'BDS-3_L2IL7I',
    #       'BDS-2_L2IL6I_BDS-3_L2IL6I',
    #       'BDS-2_L2IL7I_BDS-3_L2IL6I',
    #       'ALL']
    # fPathSer=[r'Y:/MGEX/StaList/BDS2_L2IL6I',
    #           r'Y:/MGEX/StaList/BDS2_L2IL7I',
    #           r'Y:/MGEX/StaList/BDS3_L2IL6I',
    #           r'Y:/MGEX/StaList/BDS3_L2IL7I',
    #           r'Y:/MGEX/StaList/BDS2_L2IL6I_BDS3_L2IL6I',
    #           r'Y:/MGEX/StaList/BDS2_L2IL7I_BDS3_L2IL6I',
    #           r'Y:/MGEX/StaList/ALL']

    # OutFileSuffix='StaNum_All'
    # PlotStaNum5(cSer,fPathSer,58484,730,OutFilePrefix,OutFileSuffix)

    # OutFileSuffix='StaNum_CommonWithSNX'
    # PlotStaNum6(r'Y:/IGS',r'Y:/MGEX/StaList/ALL',58484,730,False,OutFilePrefix,OutFileSuffix)

    fScan=r'Y:/MGEX/cddis/obs_2019335_cddis'
    fSta=r'Y:/MGEX/StaList/BDS2_L2IL6I_BDS3_L2IL6I/2019335_StaList'
    OutFileSuffix='StaNum_RecAnt'
    # PlotStaNum7(fScan,fSta,True,OutFilePrefix,OutFileSuffix)
