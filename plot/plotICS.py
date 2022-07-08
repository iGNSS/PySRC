#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Plot orbit ICS files
'''
__author__ = 'hanbing'

# Standard library imports
import os
import sys
import os.path
import glob

# Related third party imports
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

#
from astropy.stats import sigma_clipped_stats

from PySRC.miscellaneous import GNSSTime

def GetICSSat(fICSList,cSat):
    '''
    Get ICS (excluding the initial state vector) series for a satellite.

    NOTE: Here we assume that the ICS set for this satellite
          is same in all files.
    '''

    nFile=len(fICSList)

    #Get the ICS set
    cPar=[]; xPar=[]
    for i in range(nFile):
        if fICSList[i][-3:] != cSat:
            continue
        with open(fICSList[i],mode='rt') as fOb:
            nLine=0
            for cLine in fOb:
                nLine=nLine+1
                if nLine <= 10:
                    # Skip the first 10 lines which are file header and initial state vector
                    continue
                if cLine[0:10] == 'END of SAT':
                    break
                cWords=cLine.split()
                if cWords[0] not in cPar:
                    cPar.append(cWords[0])
                    # Epoch, estimates && post-sigma
                    xPar.append([]); xPar.append([]); xPar.append([])
                iPar=cPar.index(cWords[0])
                if cWords[0]=='AVPulse' or cWords[0]=='CVPulse' or cWords[0]=='RVPulse':
                    # nm -> um
                    rUnit=1e-3
                else:
                    rUnit=1
                #Epoch
                xPar[3*iPar  ].append(float(cWords[3])+float(cWords[4])/86400.0)
                #Estimate
                xPar[3*iPar+1].append(float(cWords[1])*rUnit)
                #Sigma of estimate
                xPar[3*iPar+2].append(float(cWords[2])*rUnit)
    return cPar,xPar

def PlotICSSat1(fList,cSatList,iAng,fGeoAngPath,MJD1,MJD2,OutFilePrefix,OutFileSuffix):
    '''
    Plot ICS series for specific satellite list, i.e.
    one figure including all parameters for each satellite and
    all figures are stored into a PDF.

       fList --- ics-files list
    cSatList --- PRN list of specified satellites
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    nSat=len(cSatList)

    # Read in the geometric angles if required
    if iAng != 0:
        Ang=[]
        for j in range(nSat):
            # Epoch, beta, delta u
            Ang.append([]); Ang.append([]); Ang.append([])
        for i in range(MJD1,MJD2+1):
            YYYY,DOY=GNSSTime.mjd2doy(i)
            for j in range(nSat):
                fAng=fGeoAngPath+'/GeoAng_'+'{:04d}{:03d}'.format(YYYY,DOY)+'_'+cSatList[j]
                if not os.path.isfile(fAng):
                    print(fAng+' does not exist!')
                    continue
                with open(fAng,mode='rt') as fOb:
                    for cLine in fOb:
                        if cLine[0:3] != 'ANG':
                            continue
                        if len(cLine)<5:
                            continue
                        cWords=cLine[40:].split()
                        #Epoch (rMJD)
                        Ang[3*j  ].append(int(cWords[0])+float(cWords[1])/86400)
                        #Beta in deg
                        Ang[3*j+1].append(float(cWords[3]))
                        #delta u, orbit angle from Noon, in deg
                        Ang[3*j+2].append(float(cWords[9]))

    strTmp=os.path.join(OutFilePrefix,OutFileSuffix+'.pdf')
    with PdfPages(strTmp) as pdf:
        for i in range(len(cSatList)):
            cPar,xPar=GetICSSat(fList,cSatList[i])
            nPar=len(cPar)
            if nPar == 0:
                print('Not found for '+cSatList[i])
                continue
            fig,axs=plt.subplots(nPar,1,sharex='col',squeeze=False,figsize=(8,3*nPar))
            fig.subplots_adjust(hspace=0.1)

            for j in range(nPar):
                axs[j,0].errorbar(xPar[3*j],xPar[3*j+1],yerr=xPar[3*j+2],marker='.',ls='--',
                                  ms=6,capsize=3,lw=1)
                axs[j,0].text(0.02,0.98,cSatList[i],color='darkred',weight='bold',
                              family='Arial',size=14,transform=axs[j,0].transAxes,ha='left',va='top')
                axs[j,0].yaxis.set_major_formatter('{x: >.2f}')
                axs[j,0].grid(which='major',axis='y',c='darkgray',ls='--',lw=0.4)
                axs[j,0].set_axisbelow(True)
                axs[j,0].set_ylabel(cPar[j],fontname='Arial',fontsize=16)
                for tl in axs[j,0].get_yticklabels():
                    tl.set_fontname('Arial'); tl.set_fontsize(14)
                if iAng==1:
                    # Plot beta angles at the right y-axes
                    axe=axs[j,0].twinx()
                    axe.set_ylim(bottom=-90,top=90)
                    axe.set_ylabel(r'$\beta$ [deg]',fontname='Arial',fontsize=16,color='goldenrod')
                    for tl in axe.get_yticklabels():
                        tl.set_fontname('Arial'); tl.set_fontsize(14); tl.set_color('goldenrod')
                    axe.plot(Ang[3*i],Ang[3*i+1],'.-',c='goldenrod',ms=1,label=r'$\beta$')
                    #Label out the period between +/- 4 deg
                    axe.fill_between(Ang[3*i],0,1,where=np.abs(Ang[3*i+1])<=4,alpha=0.5,
                                    color='silver',transform=axe.get_xaxis_transform())
            axs[j,0].set_xlabel('Modified Julian Day',fontname='Arial',fontsize=16)
            for tl in axs[j,0].get_xticklabels():
                tl.set_fontname('Arial'); tl.set_fontsize(14)

            pdf.savefig(fig,bbox_inches='tight')
            plt.close(fig)

def PlotICSSat2(fList,cSatList,OutFilePrefix,OutFileSuffix):
    '''
    Plot ICS series for specific satellite list in one figure, i.e.,
    each satellite takes one row and each parameter take one column within
    the row.
    '''

    #Global union parameter table
    cPar=[]
    #Satellite set
    cSat=[]; xPar=[]
    #Number of parameters for each satellite
    nPar=[]
    #Global index of parameters for each satellite
    iPar=[]

    for i in range(len(cSatList)):
        cPar0,xPar0=GetICSSat(fList,cSatList[i])
        nPar0=len(cPar0)
        if nPar0 == 0:
            print('Not found for '+cSatList[i])
            continue
        cSat.append(cSatList[i])
        nPar.append(nPar0); iPar.append([])
        for j in range(nPar0):
            if cPar0[j] not in cPar:
                cPar.append(cPar0[j])
            iPar[len(cSat)-1].append(cPar.index(cPar0[j]))
        xPar.append(xPar0)
    nParAll=len(cPar); nSat=len(cSat)

    fig,axs=plt.subplots(nSat,nParAll,sharex='col',
                         squeeze=False,figsize=(nParAll*8,3*nSat))
    fig.subplots_adjust(hspace=0.1)
    fig.subplots_adjust(wspace=0.1)

    for i in range(nSat):
        for j in range(nPar[i]):
            #Global index of this parameter
            k=iPar[i][j]

            #Set the y-axis limit for different parameters
            if cPar[k]=='Kd_BERN':
                axs[i,k].set_ylim(bottom=-150,top=-50)
            elif cPar[k]=='Kds2_BERN':
                axs[i,k].set_ylim(bottom=-10,top=10)
            elif cPar[k]=='Kdc2_BERN':
                axs[i,k].set_ylim(bottom=-10,top=10)
            elif cPar[k]=='Kds4_BERN':
                axs[i,k].set_ylim(bottom=-10,top=10)
            elif cPar[k]=='Kdc4_BERN':
                axs[i,k].set_ylim(bottom=-10,top=10)
            elif cPar[k]=='Kds6_BERN':
                axs[i,k].set_ylim(bottom=-10,top=10)
            elif cPar[k]=='Kdc6_BERN':
                axs[i,k].set_ylim(bottom=-10,top=10)
            elif cPar[k]=='Ky_BERN':
                axs[i,k].set_ylim(bottom=-2,top=2)
            elif cPar[k]=='Kb_BERN':
                axs[i,k].set_ylim(bottom=-5,top=5)
            elif cPar[k]=='Kbs1_BERN':
                axs[i,k].set_ylim(bottom=-15,top=15)
            elif cPar[k]=='Kbc1_BERN':
                axs[i,k].set_ylim(bottom=-15,top=15)
            elif cPar[k]=='BOXW_SB':
                axs[i,k].set_ylim(bottom=-90,top=90)

            #Cal the robusted mean, median and std
            Mea,Med,Sig=sigma_clipped_stats(np.array(xPar[i][3*j+1]),sigma=3,maxiters=5)
            if cPar[k]=='BOXW_SB' or cPar[k]=='Kd_BERN':
                axs[i,k].text(0.5,0.98,'{:>8.3f}/{:>8.3f} {:>6.3f}'.format(Mea,Med,Sig),
                              transform=axs[i,k].transAxes,ha='center',va='top',
                              fontdict={'fontsize':14,'fontname':'Arial'})
            else:
                axs[i,k].text(0.5,0.98,'{:>7.3f}/{:>7.3f} {:>6.3f}'.format(Mea,Med,Sig),
                              transform=axs[i,k].transAxes,ha='center',va='top',
                              fontdict={'fontsize':14,'fontname':'Arial'})

            axs[i,k].errorbar(xPar[i][3*j],xPar[i][3*j+1],yerr=xPar[i][3*j+2],marker='.',ls='',
                              ms=6,capsize=3,label=cPar[k])
            axs[i,k].text(0.02,0.98,cPar[k],color='darkgreen',weight='bold',family='Arial',size=14,
                          transform=axs[i,k].transAxes,ha='left',va='top')
            axs[i,k].text(0.98,0.98,cSat[i],color='darkred',weight='bold',family='Arial',size=14,
                          transform=axs[i,k].transAxes,ha='right',va='top')
            for tl in axs[i,k].get_yticklabels():
                tl.set_fontname('Arial'); tl.set_fontsize(14)

    strTmp=OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

def PlotICSSat3(fICSList,cSatList,cParList,OutFilePrefix,OutFileSuffix):
    '''
    Plot ICS series for specified parameters of specified satellites, i.e.,
    each satellite takes one row and all specified parameters for this satellite
    are plotted within the axis for this satellite.
    '''

    nSat=len(cSatList)
    fig,axs=plt.subplots(nSat,1,sharex='col',squeeze=False,figsize=(8,3*nSat))
    # fig.subplots_adjust(hspace=0.1)

    for i in range(nSat):
        cPar0,xPar0=GetICSSat(fICSList,cSatList[i])
        lValidSat=False
        for j in range(len(cParList)):
            if cParList[j] not in cPar0:
                continue
            lValidSat=True
            k=cPar0.index(cParList[j])
            axs[i,0].errorbar(xPar0[3*k],xPar0[3*k+1],yerr=xPar0[3*k+2],fmt='o--',ms=3,lw=1,
                              capsize=3,label=cParList[j])
            axs[i,0].grid(b=True,which='both',axis='y',color='darkgray',linestyle='--',
                          linewidth=0.8)
        axs[i,0].set_ylabel('',fontname='Arial',fontsize=16)
        for tl in axs[i,0].get_yticklabels():
            tl.set_fontname('Arial'); tl.set_fontsize(14)
        axs[i,0].text(0.02,0.98,cSatList[i],color='darkgreen',weight='bold',family='Arial',size=14,
                      transform=axs[i,0].transAxes,ha='left',va='top')
        if not lValidSat:
            continue
        axs[i,0].legend(loc='upper left',bbox_to_anchor=(1.0,1.0),framealpha=0.6,
                        prop={'family':'Arial','size':14})
    axs[i,0].set_xlabel('Modified Julian Day',fontname='Arial',fontsize=16)
    for tl in axs[i,0].get_xticklabels():
        tl.set_fontname('Arial'); tl.set_fontsize(14)

    strTmp=OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp,dpi=900,transparent=True,bbox_inches='tight')
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

    fGeoAngPath=os.path.join(cDskPre0,r'GNSS/PROJECT/OrbGeometry/ANG')

    InFilePrefix=os.path.join(cWrkPre0,r'PRO_2019001_2020366_WORK/E11/')
    fICSList=glob.glob(InFilePrefix+'2019/ICS_POD/???/ics_2019*')

    OutFilePrefix=os.path.join(cDskPre0,r'PRO_2019001_2020366/E11/ICS/')

    cSatList1=['C01','C02','C03','C04','C05','C06','C07','C08','C09','C10',
               'C11','C12','C13','C14','C15','C16','C17','C18','C19','C20',
               'C21','C22','C23','C24','C25','C26','C27','C28','C29','C30',
               'C31','C32','C33','C34','C35','C36','C37','C38','C39','C40',
               'C41','C42','C43','C44','C45','C46','C47','C48','C49','C50',
               'C51','C52','C53','C54','C55','C56','C57','C58','C59']
    cSatList2=['G01','G02','G03','G04','G05','G06','G07','G08','G09','G10',
               'G11','G12','G13','G14','G15','G16','G17','G18','G19','G20',
               'G21','G22','G23','G24','G25','G26','G27','G28','G29','G30',
               'G31','G32']
    cParList1=['AVPulse','CVPulse','RVPulse']

    OutFileSuffix='ICS_2019335_2019365_NoAng'
    PlotICSSat1(fICSList,cSatList1,0,fGeoAngPath,58818,58848,OutFilePrefix,OutFileSuffix)

    # OutFileSuffix='All_ICS.png'
    # PlotICSSat2(fICSList,cSatList2,OutFilePrefix,OutFileSuffix)

    OutFilePrefix=r'D:/Code/PROJECT/WORK2019335_ERROR/'
    OutFileSuffix='VPulse.pdf'
    # PlotICSSat3(fICSList,cSatList2,cParList1,OutFilePrefix,OutFileSuffix)
