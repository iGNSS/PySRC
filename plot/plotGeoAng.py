#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Make some plots based on the output during OrbGeometryAngle
'''
__author__ = 'hanbing'

# Standard library imports
import math
import os.path
import os
import sys
import glob

# Related third party imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def GetAng(fList,cSVN0,cTyp,iRec):
    '''
    Read the specified info for specfied satellites

    cSVN0 --- Specified the SVNs of satellites required
     cTyp --- 'ANG' or 'SRP'
     iRec --- Zero-started index list of the records required for each satellite
              Should no less than 8, beacuse the first 8 words (i.e., 0-7) are
              always the same. Starting from index 8, for 'ANG' record
              #  8, beta
              #  9, E
              # 10, Eps
              # 11, u
              # 12, nominal yaw
              # 13, u0
              # 14, u0-us
              # 15, Ecl yaw
              # 16, 1st of the Quaterions in CRS
              # 17, 2nd of the Quaterions in CRS
              # 18, 3rd of the Quaterions in CRS
              # 19, 4th of the Quaterions in CRS
              # 20, 1st of the Quaterions in TRS
              # 21, 2nd of the Quaterions in TRS
              # 22, 3rd of the Quaterions in TRS
              # 23, 4th of the Quaterions in TRS
              # 24, orbital period

    Return:
     cSat --- the combination list of "SVN Blk"
        X --- Data list For each satellite, except for the first two, the
              following columns are in the order as required word list
    '''

    if cTyp!='SRP' and cTyp!='ANG':
        sys.exit('Unknow info type!')
    nRec=len(iRec)
    if nRec==0:
        sys.exit('Non-specified record index list!')
    # The first two are always the Epoch && PRN
    cSat=[]; X=[]

    for i in range(len(fList)):
        with open(fList[i],mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:3] != cTyp:
                    continue
                cWords=cLine.split()
                if cSVN0[0]!='ALL' and cWords[2] not in cSVN0:
                    continue
                # SVN-blk type
                cSat0=cWords[2]+' '+cWords[3]
                if cSat0 not in cSat:
                    cSat.append(cSat0)
                    for j in range(2+nRec):
                        X.append([])
                iSat=cSat.index(cSat0)
                #Epoch, MJD
                X[iSat*(2+nRec)].append(int(cWords[5]) + float(cWords[6])/86400)
                #PRN
                X[iSat*(2+nRec)+1].append(cWords[1])
                for j in range(nRec):
                    X[iSat*(2+nRec)+2+j].append(float(cWords[iRec[j]]))
    return cSat,X

def PlotAng1(fList,cSat0,lPlot,lReport,OutFilePrefix,OutFileSuffix):
    '''
    Plot yaw attitude for each satellite one by one

    Input :
    cSat0 --- Specified PRN list
    lPlot --- bool list of selected angles to be plotted
              [0], beta angle
              [1], E
              [2], Eps
              [3], u
              [4], YawN
              [5], u0
              [6], DeltaU
              [7], YawE
              [8], YawE-YawN
              [9], orbit frequency
  lReport --- Whether report the eclipsing seasons according to beta angle
    '''

    cSat=[]; X=[]; nX=10
    for i in range(len(fList)):
        with open(fList[i],mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:3] != 'ANG':
                    continue
                cWords=cLine.split()
                if cSat0[0]!='ALL' and cWords[1] not in cSat0:
                    continue
                if cWords[1]+'-'+cWords[2] not in cSat:
                    #Satellite PRN-SVN
                    cSat.append(cWords[1]+'-'+cWords[2])
                    for j in range(nX):
                        X.append([])
                iSat=cSat.index(cWords[1]+'-'+cWords[2])
                #Epoch, MJD
                X[iSat*nX   ].append(int(cWords[5]) + float(cWords[6])/86400)
                #Beta
                X[iSat*nX+ 1].append(float(cWords[8]))
                #E
                X[iSat*nX+ 2].append(float(cWords[9]))
                #Eps
                X[iSat*nX+ 3].append(float(cWords[10]))
                #u
                X[iSat*nX+ 4].append(float(cWords[11]))
                #YawN
                X[iSat*nX+ 5].append(float(cWords[12]))
                #u0
                X[iSat*nX+ 6].append(float(cWords[13]))
                #DeltaU
                X[iSat*nX+ 7].append(float(cWords[14]))
                #YawE
                X[iSat*nX+ 8].append(float(cWords[15]))
                #Orbit Period, in days
                X[iSat*nX+ 9].append(float(cWords[24]))
    nSat=len(cSat); cSat1=cSat.copy(); cSat1.sort();

    fig,axs=plt.subplots(nSat,1,sharex='col',squeeze=False,figsize=(12,3*nSat))
    fig.subplots_adjust(hspace=0.1)
    formatterx=mpl.ticker.StrMethodFormatter('{x:7.1f}')

    for i in range(nSat):
        j=cSat.index(cSat1[i]); Ang=np.array(X[nX*j:nX*j+nX])
        ix=np.argsort(Ang[0])
        #Report some special periods for each satellite
        if lReport:
            lEnter1=False
            for k in range(Ang[0].size):
                if np.abs(Ang[1][ix[k]]) >= 4:
                    lEnter2=False
                else:
                    lEnter2=True
                if (lEnter1 and not lEnter2) or (not lEnter1 and lEnter2):
                    print(cSat[j]+' {:>10.4f} {:>6.2f}'.format(Ang[0][ix[k]],Ang[1][ix[k]]))
                lEnter1=lEnter2

        axs[i,0].text(0.02,0.98,cSat[j],color='darkred',weight='bold',family='Arial',size=14,
                    transform=axs[i,0].transAxes,ha='left',va='top')
        if len(lPlot)>=1 and lPlot[0]:
            # Beta angle
            axs[i,0].plot(Ang[0][ix],Ang[1][ix],ls='',marker='.',ms=4,label=r'$\beta$')
        if len(lPlot)>=2 and lPlot[1]:
            # E angle
            axs[i,0].plot(Ang[0][ix],Ang[2][ix],ls='',marker='.',ms=4,label='E')
        if len(lPlot)>=3 and lPlot[2]:
            # Epsilon angle
            axs[i,0].plot(Ang[0][ix],Ang[3][ix],ls='',marker='.',ms=4,label=r'$\epsilon$')
        if len(lPlot)>=4 and lPlot[3]:
            # u angle
            axs[i,0].plot(Ang[0][ix],Ang[4][ix],ls='',marker='.',ms=4,label=r'$\mu$')
        if len(lPlot)>=5 and lPlot[4]:
            # Nominal Yaw
            axs[i,0].plot(Ang[0][ix],Ang[5][ix],ls='-',lw=1,marker='.',ms=2,label=r'$\Psi_n$')
        if len(lPlot)>=6 and lPlot[5]:
            # u0 angle
            axs[i,0].plot(Ang[0][ix],Ang[6][ix],ls='',marker='.',ms=4,label=r'$\mu$0')
        if len(lPlot)>=7 and lPlot[6]:
            # delta u angle
            axs[i,0].plot(Ang[0][ix],Ang[7][ix],ls='',marker='.',ms=4,label=r'$\Delta\mu$')
        if len(lPlot)>=8 and lPlot[7]:
            # actual Yaw
            axs[i,0].plot(Ang[0][ix],Ang[8][ix],ls='-',lw=1,marker='.',ms=2,label=r'$\Psi_e$')
        if len(lPlot)>=9 and lPlot[8]:
            #Yaw difference, Normalize to [0,PI]
            y=np.abs(Ang[8][ix]-Ang[5][ix])
            for j in range(y.size):
                if y[j] > 180:
                    y[j]=y[j]-180
            axs[i,0].plot(Ang[0][ix],y,ls='',marker='.',ms=2,label=r'$\Delta\Psi$')
        if len(lPlot)>=10 and lPlot[9]:
            # Orbit period, in cycles per day
            axs[i,0].plot(Ang[0][ix],1/Ang[9][ix],ls='-',lw=1,marker='.',ms=2,label=r'Orb freq')
        axs[i,0].ticklabel_format(axis='y',style='sci',useOffset=False,useMathText=True)
        axs[i,0].set_ylabel('',fontname='Arial',fontsize=16)
        for tl in axs[i,0].get_yticklabels():
            tl.set_fontname('Arial'); tl.set_fontsize(14)
        axs[i,0].legend(loc='upper right',labelcolor='mfc',framealpha=0.6,
                        labelspacing=0.1,borderpad=0.1,prop={'family':'Arial','size':14})
    for tl in axs[i,0].get_xticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)
    axs[i,0].set_xlabel('Modified Julian Day',fontname='Arial',fontsize=16)
    axs[i,0].xaxis.set_major_formatter(formatterx)

    strTmp=OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotAng2(fList,iPlot,cSVN0,OutFilePrefix,OutFileSuffix):
    '''
    Plot the specified satellite yaw attitude data, one axe for all satellite

    Input :
    iPlot --- Zero-started index the record required for each satellite
              Should no less than 8, beacuse the first 8 words (i.e., 0-7) are
              always the same. Starting from index 8, for 'ANG' record
              #  8, beta
              #  9, E
              # 10, Eps
              # 11, u
              # 12, nominal yaw
              # 13, u0
              # 14, u0-us
              # 15, Ecl yaw
              # 16, 1st of the Quaterions in CRS
              # 17, 2nd of the Quaterions in CRS
              # 18, 3rd of the Quaterions in CRS
              # 19, 4th of the Quaterions in CRS
              # 20, 1st of the Quaterions in TRS
              # 21, 2nd of the Quaterions in TRS
              # 22, 3rd of the Quaterions in TRS
              # 23, 4th of the Quaterions in TRS
              # 24, orbital period
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    cSVN=[]; X=[]
    for i in range(len(fList)):
        with open(fList[i],mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:3] != 'ANG':
                    continue
                cWords=cLine.split()
                if cSVN0[0]!='ALL' and cWords[2] not in cSVN0:
                    continue
                if cWords[2] not in cSVN:
                    #Satellite SVNs
                    cSVN.append(cWords[2])
                    # Epoch and angles
                    X.append([]); X.append([])
                iSat=cSVN.index(cWords[2])
                #Epoch, MJD
                X[iSat*2  ].append(int(cWords[5]) + float(cWords[6])/86400)
                X[iSat*2+1].append(float(cWords[iPlot]))
    cSat0=cSVN.copy(); cSat0.sort(); nSat=len(cSVN)

    fig,axs=plt.subplots(1,1,squeeze=False,figsize=(10,4))

    yLab=[r'$\beta$ [deg]',r'E [deg]',r'$\epsilon$ [deg]',r'$\mu$ [deg]',
          r'$\Psi_n$ [deg]',r'$\mu$0 [deg]',r'$\Delta$$\mu$ [deg]',r'$\Psi_e$ [deg]']
    axs[0,0].set_ylabel(yLab[iPlot-8],fontname='Arial',fontsize=16)
    axs[0,0].grid(which='major',axis='y',c='darkgray',ls='--',lw=0.4)
    axs[0,0].set_axisbelow(True)

    for i in range(nSat):
        j=cSVN.index(cSat0[i])
        if cSVN[j]=='C003':
            cLab=cSVN[j]+ ' (BDS-2G)'
        elif cSVN[j]=='C005':
            cLab=cSVN[j]+ ' (BDS-2I)'
        elif cSVN[j]=='C007':
            cLab=cSVN[j]+ ' (BDS-2I)'
        elif cSVN[j]=='C008':
            cLab=cSVN[j]+ ' (BDS-2I)'
        elif cSVN[j]=='C224':
            cLab=cSVN[j]+ ' (BDS-3I)'
        elif cSVN[j]=='C012':
            cLab=cSVN[j]+ ' (BDS-2M)'
        elif cSVN[j]=='C015':
            cLab=cSVN[j]+ ' (BDS-2M)'
        elif cSVN[j]=='C209':
            cLab=cSVN[j]+ ' (BDS-3M)'
        else:
            cLab=cSVN[j]
        # Sort the data
        ind=np.argsort(X[j*2])
        Ang=np.zeros((ind.size,2))
        for k in range(ind.size):
            Ang[k,0]=X[j*2][ind[k]]
            Ang[k,1]=X[j*2+1][ind[k]]
        axs[0,0].plot(Ang[:,0],Ang[:,1],ls='-',lw=1,marker='.',ms=3,label=cLab)

    axs[0,0].legend(ncol=1,loc='center left',bbox_to_anchor=(1.0,0.5),labelcolor='mfc',framealpha=0.6,
                    labelspacing=0.1,borderpad=0.1,handletextpad=0.1,prop={'family':'Arial','size':14})
    for tl in axs[0,0].get_yticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)

    axs[0,0].xaxis.set_major_formatter('{x: >7.1f}')
    for tl in axs[0,0].get_xticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)
    axs[0,0].set_xlabel('Modified Julian Day',fontname='Arial',fontsize=16)

    strTmp=OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    strTmp=OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotAng3(fList,cSVN,OutFilePrefix,OutFileSuffix):
    '''
    Plot yaw attitude for the specified satellite, one axe for every revolution

    Input :
    cSVN --- the specified satellite, SVN
    '''

    X=[[],[],[],[],[],[],[],[],[]]
    for i in range(len(fList)):
        with open(fList[i],mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:1] != ' ':
                    continue
                cWords=cLine.split()
                if cWords[1] != cSVN:
                    continue
                #Epoch
                X[0].append(float(cWords[4]))
                #Beta
                X[1].append(float(cWords[5]))
                #E
                X[2].append(float(cWords[6]))
                #Eps
                X[3].append(float(cWords[7]))
                #u
                X[4].append(float(cWords[8]))
                #YawN
                X[5].append(float(cWords[9]))
                #u0
                X[6].append(float(cWords[10]))
                #DeltaU
                X[7].append(float(cWords[11]))
                #YawE
                X[8].append(float(cWords[12]))
    if len(X[0]) == 0:
        sys.exit(cSVN+' not found!')
    Ang=np.array(X); ix=np.argsort(Ang[0])
    #Count how many times the satellite passed the point where u=90 deg
    iPass=[]; nRev=0
    for i in range(Ang[0].size):
        if i != 0:
            if uLast<=90 and Ang[4][ix[i]]>90:
                iPass.append(i)
        uLast=Ang[4][ix[i]]
    #Number of revolutions
    nRev=len(iPass)-1

    fig,axs=plt.subplots(nRev,1,squeeze=False,figsize=(4,4*nRev))

    for i in range(nRev):
        #Start && end point of this revolution
        iRev=ix[iPass[i]:iPass[i+1]]

        axs[i,0].plot(Ang[0][iRev],Ang[5][iRev],ls='-',lw=1,marker='.',ms=2,label=r'$\Psi_n$')
        axs[i,0].plot(Ang[0][iRev],Ang[8][iRev],ls='-',lw=1,marker='.',ms=2,label=r'$\Psi_e$')

        #Normalize the yaw error into [0,PI]
        y=np.abs(Ang[8][iRev]-Ang[5][iRev])
        for k in range(y.size):
            if y[k] > 180:
                y[k]=y[k]-180
        axs[i,0].plot(Ang[0][iRev],y,ls='',marker='^',ms=4,label=r'$\Delta\Psi$')

        strTmp='{:>6.2f}'.format(Ang[1][ix[iPass[i]]])
        strTmp=strTmp+r'$^\circ\leqslant\beta\leqslant$'
        strTmp=strTmp+'{:>6.2f}'.format(Ang[1][ix[iPass[i+1]-1]])
        strTmp=strTmp+r'$^\circ$'
        axs[i,0].text(0.5,1.0,strTmp,transform=axs[i,0].transAxes,ha='center',va='bottom',
                      fontdict={'fontsize':14,'fontname':'Arial'})
        axs[i,0].text(0.0,0.5,cSVN,color='darkred',weight='bold',family='Arial',size=14,
                      transform=axs[i,0].transAxes,ha='left',va='center')

        axs[i,0].legend(loc='center right',bbox_to_anchor=(1.0,0.5),labelcolor='mfc',framealpha=0.6,
                        labelspacing=0.1,borderpad=0.1,handletextpad=0.3,handlelength=0.5,
                        columnspacing=0.8,prop={'family':'Arial','size':14})
        for tl in axs[i,0].get_yticklabels():
            tl.set_fontname('Arial'); tl.set_fontsize(14)
        axs[i,0].set_ylabel('[deg]',fontname='Arial',fontsize=16)
        axs[i,0].set_xticklabels([],minor=False)
        for tl in axs[i,0].get_xticklabels():
            tl.set_fontname('Arial'); tl.set_fontsize(14)
        axs[i,0].set_xlabel(' ',fontname='Arial',fontsize=16)

    strTmp=OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotAng4(fList,cSVN0,OutFilePrefix,OutFileSuffix):
    '''
    Plot the orbit periods for specified satellites


    '''
    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    # PRN, SVN, blk type
    cSat=[[],[],[]]; X=[]
    for i in range(len(fList)):
        with open(fList[i],mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:3] != 'ANG':
                    continue
                cWords=cLine.split()
                if cSVN0[0]!='ALL' and cWords[2] not in cSVN0:
                    continue
                if cWords[2] not in cSat[1]:
                    # New satellite, PRN, SVN, blk type
                    cSat[0].append(cWords[1])
                    cSat[1].append(cWords[2])
                    cSat[2].append(cWords[3])
                    # Epoch, period
                    X.append([]); X.append([])
                iSat=cSat[1].index(cWords[2])
                #Epoch, rmjd
                X[iSat*2  ].append(int(cWords[5]) + float(cWords[6])/86400)
                #orbit period, in days
                X[iSat*2+1].append(float(cWords[24]))
    nSat=len(cSat[0]); cSVN=cSat[1].copy(); cSVN.sort()

    fig,axs=plt.subplots(1,1,squeeze=False,figsize=(12,4))

    # Report to the terminal
    print('{: <3s} {: >4s} {: >9s} {: >9s}'.format('PRN','SVN','Mean[d]','STD[s]'))
    for i in range(nSat):
        j=cSat[1].index(cSVN[i])
        ind=np.argsort(X[j*2])
        t=np.array(X[j*2])[ind]; OP=np.array(X[j*2+1])[ind]
        # Mean frequency in cycles per day
        Mea=1/np.mean(OP)
        # Variation of orbit periods in seconds
        STD=np.std(OP)*86400
        axs[0,0].plot(t,OP,'o-',ms=2,lw=0.5,label=cSVN[i])
        print('{:>3s} {:>4s} {:>9.6f} {:>9.2f}'.format(cSat[0][j],cSat[1][j],Mea,STD))

    axs[0,0].grid(which='major',axis='y',color='darkgray',linestyle='--',linewidth=0.4)
    axs[0,0].set_axisbelow(True)

    axs[0,0].legend(loc='center left',bbox_to_anchor=(1.0,0.5),labelcolor='mfc',framealpha=0.6,
                    labelspacing=0.1,borderpad=0.1,handletextpad=0.1,prop={'family':'Arial','size':14})
    axs[0,0].ticklabel_format(axis='y',style='sci',useOffset=False,useMathText=True)
    for tl in axs[0,0].get_yticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)
    axs[0,0].set_ylabel('Orbit periods [day]',fontname='Arial',fontsize=16)

    for tl in axs[0,0].get_xticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)
    axs[0,0].set_xlabel('Modified Julian Day',fontname='Arial',fontsize=16)
    formatterx=mpl.ticker.StrMethodFormatter('{x:7.1f}')
    axs[0,0].xaxis.set_major_formatter(formatterx)

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
        cWrkPre0=r'Y:/'

        cDskPre0=r'Z:/'
    else:
        # GFZ section cluster
        cWrkPre0=r'/wrk/hanbing/'

        cDskPre0=r'/dsk/hanbing/'
    print('Run On '+cWhere)

    # The correspoinding relation between PRN and SVN may change as time goes
    cPRN=['C19','C20','C21','C22','C23','C24','C25','C26','C27','C28',
          'C29','C30','C32','C33','C34','C35','C36','C37','C38','C39',
          'C40']
    cSVN=['C201','C202','C206','C205','C209','C210','C212','C211','C203','C204',
          'C207','C208','C213','C214','C216','C215','C218','C219','C220','C221',
          'C224']
    # # BDS-2/BDS-3 IGSOs
    # cSVN=['C005','C007','C008','C009','C010','C017','C019','C220','C221','C224']
    # # BDS-2/BDS-3 MEOs on Plane A
    # cSVN=['C012','C013','C203','C204','C207','C208','C216','C215','C226','C225']
    # # BDS-2/BDS-3 MEOs on Plane B
    # cSVN=['C015','C201','C202','C206','C205','C213','C214','C227','C228']
    # # BDS-2/BDS-3 MEOs on Plane C
    # cSVN=['C209','C210','C218','C219','C223','C222','C212','C211']
    # Typical satellites from each plane
    # cSVN=['C003','C005','C007','C008','C224','C012','C015','C209']

    # InFilePrefix=r'D:/Code/PROJECT/WORK_ANG/'
    InFilePrefix=os.path.join(cDskPre0,r'GNSS/PROJECT/OrbGeometry/ANG/')
    fList=glob.glob(InFilePrefix+'GeoAng_20?????_C??')
    # fList += glob.glob(InFilePrefix+'GeoAng_202000[1-9]_C2[5-9]')

    # OutFilePrefix=r'D:/Code/PROJECT/WORK_ANG/'
    OutFilePrefix=os.path.join(cDskPre0,r'PRO_2019001_2020366/Common/')

    # OutFileSuffix='Orbit_Freq.pdf'
    # lPlot=[False,False,False,False,False,False,False,False,False,True]
    # PlotAng1(fList,['ALL'],lPlot,False,OutFilePrefix,OutFileSuffix)
    OutFileSuffix='BDS_beta'
    PlotAng2(fList,8,['C003','C005','C007','C008','C224','C012','C015','C209'],OutFilePrefix,OutFileSuffix)

    # PlotAng3(fList,'C012',OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='BDS-2_OrbitPeriod_2019335_2019365'
    # PlotAng4(fList,cSVN,OutFilePrefix,OutFileSuffix)
