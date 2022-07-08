#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Plot the log file(s) from pre-processing of GNSS data by turboedit
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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

def GetMP(fList,cSat0,cTyp0):
    '''Extract MP series for cSat0 and cTyp0

    fList : log file list
    '''

    cSat=[]
    cTyp=[]
    cArc=[]
    for i in range(len(fList)):
        with open(fList[i],mode='r',encoding='UTF-16') as fOb:
            for cLine in fOb:
                if cLine[0:5] != 'M1M2 ':
                    continue
                cWords=cLine.split()
                if cSat0[0] != 'ALL' and cWords[1] not in cSat0:
                    continue
                if cTyp0[0] != 'ALL ' and cWords[2] not in cTyp0:
                    continue
                if cWords[1] not in cSat:
                    cSat.append(cWords[1])
                    cTyp.append([])
                    cArc.append([])
                iSat=cSat.index(cWords[1])
                if cWords[2] not in cTyp[iSat]:
                    cTyp[iSat].append(cWords[2])
                    cArc[iSat].append([])
                iTyp=cTyp[iSat].index(cWords[2])
                if cWords[3] not in cArc[iSat][iTyp]:
                    cArc[iSat][iTyp].append(cWords[3])
    MP=[]
    for i in range(len(cSat)):
        MP.append([])
        for j in range(len(cTyp[i])):
            MP[i].append([])
            for k in range(len(cArc[i][j])):
                MP[i][j].append([])
                for l in range(7):
                    MP[i][j][k].append([])
    for i in range(len(fList)):
        with open(fList[i],mode='r',encoding='UTF-16') as fOb:
            for cLine in fOb:
                if cLine[0:5] != 'M1M2 ':
                    continue
                cWords=cLine.split()
                if cSat0[0] != 'ALL' and cWords[1] not in cSat0:
                    continue
                if cTyp0[0] != 'ALL ' and cWords[2] not in cTyp0:
                    continue
                iSat=cSat.index(cWords[1])
                iTyp=cTyp[iSat].index(cWords[2])
                iArc=cArc[iSat][iTyp].index(cWords[3])
                #Epoch
                MP[iSat][iTyp][iArc][0].append(int(cWords[4]))
                #Azim at station
                MP[iSat][iTyp][iArc][1].append(float(cWords[5]))
                #Elev at station
                MP[iSat][iTyp][iArc][2].append(float(cWords[6]))
                #Azim at satellite
                MP[iSat][iTyp][iArc][3].append(float(cWords[7]))
                #Nadir at satellite
                MP[iSat][iTyp][iArc][4].append(float(cWords[8]))
                #MP 1
                MP[iSat][iTyp][iArc][5].append(float(cWords[9]))
                #MP 2
                MP[iSat][iTyp][iArc][6].append(float(cWords[10]))

    return cSat,cTyp,cArc,MP

def GetComb(fList,cSat0,cTyp0,Comb):
    '''Extract comb series for cSat0 and cTyp0

    fList : log file list
    '''

    cSat=[]
    cTyp=[]
    cArc=[]
    for i in range(len(fList)):
        with open(fList[i],mode='r',encoding='UTF-16') as fOb:
            for cLine in fOb:
                cWords=cLine.split()
                if cWords[0] != Comb:
                    continue
                if cSat0[0] != 'ALL' and cWords[1] not in cSat0:
                    continue
                if cTyp0[0] != 'ALL ' and cWords[2] not in cTyp0:
                    continue
                if cWords[1] not in cSat:
                    cSat.append(cWords[1])
                    cTyp.append([])
                    cArc.append([])
                iSat=cSat.index(cWords[1])
                if cWords[2] not in cTyp[iSat]:
                    cTyp[iSat].append(cWords[2])
                    cArc[iSat].append([])
                iTyp=cTyp[iSat].index(cWords[2])
                if cWords[3] not in cArc[iSat][iTyp]:
                    cArc[iSat][iTyp].append(cWords[3])
    MP=[]
    for i in range(len(cSat)):
        MP.append([])
        for j in range(len(cTyp[i])):
            MP[i].append([])
            for k in range(len(cArc[i][j])):
                MP[i][j].append([])
                for l in range(6):
                    MP[i][j][k].append([])
    for i in range(len(fList)):
        with open(fList[i],mode='r',encoding='UTF-16') as fOb:
            for cLine in fOb:
                cWords=cLine.split()
                if cWords[0] != Comb:
                    continue
                if cSat0[0] != 'ALL' and cWords[1] not in cSat0:
                    continue
                if cTyp0[0] != 'ALL ' and cWords[2] not in cTyp0:
                    continue
                iSat=cSat.index(cWords[1])
                iTyp=cTyp[iSat].index(cWords[2])
                iArc=cArc[iSat][iTyp].index(cWords[3])
                #Epoch
                MP[iSat][iTyp][iArc][0].append(int(cWords[4]))
                #Azim at station
                MP[iSat][iTyp][iArc][1].append(float(cWords[5]))
                #Elev at station
                MP[iSat][iTyp][iArc][2].append(float(cWords[6]))
                #Azim at satellite
                MP[iSat][iTyp][iArc][3].append(float(cWords[7]))
                #Nadir at satellite
                MP[iSat][iTyp][iArc][4].append(float(cWords[8]))
                #Comb value
                MP[iSat][iTyp][iArc][5].append(float(cWords[9]))

    return cSat,cTyp,cArc,MP

def PlotMP(fLogList,cSat0,cTyp0,lNorm,OutFilePrefix,OutFileSuffix):
    '''
    Plot MP series for specific satellites and observation types

    lNorm : Whether remove the mean in each arc
    '''

    cSat,cTyp,cArc,MP=GetMP(fLogList,cSat0,cTyp0)

    strTmp=OutFilePrefix+OutFileSuffix
    with PdfPages(strTmp) as pdf:
        #For each satellites
        for i in range(len(cSat)):
            nTyp=len(cTyp[i])
            fig,axs=plt.subplots(nTyp,4,squeeze=False,figsize=(4*4,nTyp*1.5))
            fig.subplots_adjust(hspace=0.1)
            for j in range(nTyp):
                nArc=len(cArc[i][j])
                for k in range(nArc):
                    if lNorm:
                        #Remove arc mean
                        MP[i][j][k][5]=MP[i][j][k][5]-np.mean(MP[i][j][k][5])
                        MP[i][j][k][6]=MP[i][j][k][6]-np.mean(MP[i][j][k][6])
                    axs[j,0].plot(MP[i][j][k][0],MP[i][j][k][5],'.',markersize=2)
                    axs[j,1].plot(MP[i][j][k][4],MP[i][j][k][5],'.',markersize=2)
                    axs[j,2].plot(MP[i][j][k][0],MP[i][j][k][6],'.',markersize=2)
                    axs[j,3].plot(MP[i][j][k][4],MP[i][j][k][6],'.',markersize=2)
                axs[j,0].text(0.01,0.99,cSat[i]+' '+cTyp[i][j]+' MP1',
                              transform=axs[j,0].transAxes,ha='left',va='top')
                # axs[j,0].set_ylabel('[m]')
                if lNorm:
                    x1,x2=axs[j,0].get_xlim()
                    axs[j,0].hlines(0,x1,x2,colors=['darkgray'],linestyle='dashed')
                axs[j,1].text(0.01,0.99,cSat[i]+' '+cTyp[i][j]+' MP1',
                              transform=axs[j,1].transAxes,ha='left',va='top')
                axs[j,1].set_xlim(left=0,right=15)
                if lNorm:
                    x1,x2=axs[j,1].get_xlim()
                    axs[j,1].hlines(0,x1,x2,colors=['darkgray'],linestyle='dashed')
                axs[j,2].text(0.01,0.99,cSat[i]+' '+cTyp[i][j]+' MP2',
                              transform=axs[j,2].transAxes,ha='left',va='top')
                if lNorm:
                    x1,x2=axs[j,2].get_xlim()
                    axs[j,2].hlines(0,x1,x2,colors=['darkgray'],linestyle='dashed')
                axs[j,3].text(0.01,0.99,cSat[i]+' '+cTyp[i][j]+' MP2',
                              transform=axs[j,3].transAxes,ha='left',va='top')
                axs[j,3].set_xlim(left=0,right=15)
                if lNorm:
                    x1,x2=axs[j,3].get_xlim()
                    axs[j,3].hlines(0,x1,x2,colors=['darkgray'],linestyle='dashed')
            axs[j,0].set_xlabel('Epoch')
            axs[j,1].set_xlabel('Nadir [deg]')
            axs[j,2].set_xlabel('Epoch')
            axs[j,3].set_xlabel('Nadir [deg]')
            pdf.savefig(fig,bbox_inches='tight')
            plt.close(fig)

def PlotS1S2(fLogList,cSat0,cTyp0,OutFilePrefix,OutFileSuffix):
    '''
    Plot S1/S2 series (clock jump observable) for specific satellites and observation types

    '''

    cSat=[]
    cTyp=[]
    for i in range(len(fLogList)):
        with open(fLogList[i],mode='r',encoding='UTF-16') as fOb:
            for cLine in fOb:
                if cLine[0:5] != 'S1S2 ':
                    continue
                cWords=cLine.split()
                if cSat0[0] != 'ALL' and cWords[1] not in cSat0:
                    continue
                if cWords[1] not in cSat:
                    cSat.append(cWords[1])
                    cTyp.append([])
    cSat.sort()
    for i in range(len(fLogList)):
        with open(fLogList[i],mode='r',encoding='UTF-16') as fOb:
            for cLine in fOb:
                if cLine[0:5] != 'S1S2 ':
                    continue
                cWords=cLine.split()
                if cWords[1] not in cSat:
                    continue
                if cTyp0[0] != 'ALL ' and cWords[2] not in cTyp0:
                    continue
                iSat=cSat.index(cWords[1])
                if cWords[2] not in cTyp[iSat]:
                    cTyp[iSat].append(cWords[2])
    S=[]
    for i in range(len(cSat)):
        S.append([])
        for j in range(len(cTyp[i])):
            S[i].append([])
            for k in range(4):
                S[i][j].append([])
    for i in range(len(fLogList)):
        with open(fLogList[i],mode='r',encoding='UTF-16') as fOb:
            for cLine in fOb:
                if cLine[0:5] != 'S1S2 ':
                    continue
                cWords=cLine.split()
                if cSat0[0] != 'ALL' and cWords[1] not in cSat0:
                    continue
                if cTyp0[0] != 'ALL ' and cWords[2] not in cTyp0:
                    continue
                iSat=cSat.index(cWords[1])
                iTyp=cTyp[iSat].index(cWords[2])
                #Epoch
                S[iSat][iTyp][0].append(int(cWords[3]))
                #Elev at station
                S[iSat][iTyp][3].append(float(cWords[5]))
                #S1
                S[iSat][iTyp][1].append(float(cWords[8]))
                #S2
                S[iSat][iTyp][2].append(float(cWords[9]))

    strTmp=OutFilePrefix+OutFileSuffix
    with PdfPages(strTmp) as pdf:
        #For each satellites
        for i in range(len(cSat)):
            nTyp=len(cTyp[i])
            fig,axs=plt.subplots(nTyp,1,squeeze=False,figsize=(6,nTyp*3))
            fig.subplots_adjust(hspace=0.1)
            for j in range(nTyp):
                #S1
                p1, =axs[j,0].plot(S[i][j][0],S[i][j][1],'.',ms=2,mec='maroon',mfc='maroon',label='S1')
                #S2
                p2, =axs[j,0].plot(S[i][j][0],S[i][j][2],'.',ms=2,mec='blueviolet',mfc='blueviolet',label='S2')
                #Elev
                axe=axs[j,0].twinx()
                p3, =axe.plot(S[i][j][0],S[i][j][3],'.',ms=2,mec='gold',mfc='gold',label='Elev')
                lines=[p1,p2,p3]
                axe.set_ylabel('[deg]')

                axs[j,0].legend(lines,[l.get_label() for l in lines],ncol=3,loc='lower center')
                axs[j,0].text(0.01,0.99,cSat[i],transform=axs[j,0].transAxes,ha='left',va='top')
                axs[j,0].set_ylabel('[m]')
            axs[j,0].set_xlabel('Epoch')
            pdf.savefig(fig,bbox_inches='tight')
            plt.close(fig)

def PlotComb(fLogList,cSat0,cTyp0,Comb,lNorm,OutFilePrefix,OutFileSuffix):
    '''
    Plot comb series for specific satellites and observation types

    lNorm : Whether remove the mean in each arc
    '''

    cSat,cTyp,cArc,MP=GetComb(fLogList,cSat0,cTyp0,Comb)

    strTmp=OutFilePrefix+OutFileSuffix
    with PdfPages(strTmp) as pdf:
        #For each satellites
        for i in range(len(cSat)):
            nTyp=len(cTyp[i])
            fig,axs=plt.subplots(nTyp,1,squeeze=False,figsize=(12,nTyp*1.5))
            fig.subplots_adjust(hspace=0.1)
            for j in range(nTyp):
                nArc=len(cArc[i][j])
                for k in range(nArc):
                    if lNorm:
                        #Remove arc mean
                        MP[i][j][k][5]=MP[i][j][k][5]-np.mean(MP[i][j][k][5])
                    if cArc[i][j][k]=='999999':
                        axs[j,0].plot(MP[i][j][k][0],MP[i][j][k][5],'rx',markersize=3)
                    else:
                        axs[j,0].plot(MP[i][j][k][0],MP[i][j][k][5],'.',markersize=2)
                axs[j,0].text(0.01,0.99,cSat[i]+' '+cTyp[i][j]+' '+Comb,
                              transform=axs[j,0].transAxes,ha='left',va='top')
                # axs[j,0].set_ylabel('[m]')
                if lNorm:
                    x1,x2=axs[j,0].get_xlim()
                    axs[j,0].hlines(0,x1,x2,colors=['darkgray'],linestyle='dashed')
            axs[j,0].set_xlabel('Epoch')
            pdf.savefig(fig,bbox_inches='tight')
            plt.close(fig)

def PlotP4(fLog,cSat0,OutFilePrefix,OutFileSuffix):
    '''
    '''

    cSat=[]
    with open(fLog,mode='r',encoding='UTF-16') as fOb:
        for cLine in fOb:
            if cLine[0:7] != 'Polyfit':
                continue
            cWords=cLine.split()
            if cSat0[0] != 'ALL' and cWords[1] not in cSat0:
                continue
            if cWords[1] not in cSat:
                cSat.append(cWords[1])
    cSat.sort()

    strTmp=OutFilePrefix+OutFileSuffix
    with PdfPages(strTmp) as pdf:
        #For each satellites
        for i in range(len(cSat)):
            #Read data for this sat
            with open(fLog,mode='r',encoding='UTF-16') as fOb:
                cLine=fOb.readline()
                while cLine:
                    if cLine[0:7] != 'Polyfit':
                        cLine=fOb.readline()
                        continue
                    cWords=cLine.split()
                    if cWords[1] != cSat[i]:
                        cLine=fOb.readline()
                        continue
                    nObs=int(cWords[3])
                    Flag=np.zeros(nObs,dtype=np.int8)
                    Epo=np.zeros(nObs)
                    P4=np.zeros(nObs)
                    V4=np.zeros(nObs)
                    F4=np.zeros(nObs)
                    Fit=np.zeros(nObs)
                    Q4=np.zeros(nObs)
                    dx=float(cWords[4])
                    ft=float(cWords[5])
                    nDg=int(cWords[6])
                    a=np.zeros(nDg+1)
                    for j in range(nDg+1):
                        a[j]=float(cWords[7+j])
                    for j in range(nObs):
                        cLine=fOb.readline()
                        cWords=cLine.split()
                        Flag[j]=int(cWords[3])
                        Epo[j]=float(cWords[4])
                        if Flag[j]==9:
                            P4[j]=np.nan
                            V4[j]=np.nan
                        else:
                            P4[j]=float(cWords[5])
                            V4[j]=float(cWords[7])
                        F4[j]=a[nDg]
                        for k in range(nDg-1,-1,-1):
                            F4[j]=F4[j]*(Epo[j]-dx)*ft + a[k]
                        Fit[j]=float(cWords[6])
                        if cWords[8] == '9999.9999':
                            Q4[j]=np.nan
                        else:
                            Q4[j]=float(cWords[8])
                    break
            fig,axs=plt.subplots(3,1,sharex='col',squeeze=False,figsize=(12,3*3))
            fig.subplots_adjust(hspace=0.1)
            axs[0,0].plot(Epo,P4,'.',markersize=2)
            axs[0,0].plot(Epo,F4,'b.-',markersize=2)
            axs[0,0].text(0.01,0.99,cSat[i],transform=axs[0,0].transAxes,ha='left',va='top')
            axs[0,0].text(0.99,0.99,'P4 polyfit',transform=axs[0,0].transAxes,ha='right',va='top')

            axs[1,0].plot(Epo,V4,'r.',markersize=2)
            axs[1,0].text(0.99,0.99,'P4 fit res',transform=axs[1,0].transAxes,ha='right',va='top')

            axs[2,0].plot(Epo,Q4,'g.',markersize=2)
            axs[2,0].text(0.99,0.99,'L4-P4',transform=axs[2,0].transAxes,ha='right',va='top')


            pdf.savefig(fig,bbox_inches='tight')
            plt.close(fig)

def PlotP3(fLog,cSat0,lNorm,OutFilePrefix,OutFileSuffix):
    '''
    '''

    cSat=[]
    with open(fLog,mode='r',encoding='UTF-16') as fOb:
        for cLine in fOb:
            if cLine[0:5] != 'P3Res':
                continue
            cWords=cLine.split()
            if cSat0[0] != 'ALL' and cWords[1] not in cSat0:
                continue
            if cWords[1] not in cSat:
                cSat.append(cWords[1])
    cSat.sort()

    strTmp=OutFilePrefix+OutFileSuffix
    with PdfPages(strTmp) as pdf:
        #For each satellites
        for i in range(len(cSat)):
            #Read data for this sat
            P3Res=[[],[],[],[],[],[]]
            with open(fLog,mode='r',encoding='UTF-16') as fOb:
                for cLine in fOb:
                    if cLine[0:5] != 'P3Res':
                        continue
                    cWords=cLine.split()
                    if cWords[1] != cSat[i]:
                        continue
                    if int(cWords[3]) == 1:
                        continue
                    #Epoch
                    P3Res[0].append(int(cWords[4]))
                    P3Res[1].append(float(cWords[5]))
                    P3Res[2].append(float(cWords[6]))
                    P3Res[3].append(float(cWords[7]))
                    P3Res[4].append(float(cWords[8]))
                    #P3 res
                    P3Res[5].append(float(cWords[9]))
            if lNorm:
                P3Res[5]=P3Res[5]-np.mean(P3Res[5])
            fig,axs=plt.subplots(1,1,sharex='col',squeeze=False,figsize=(12,3))
            fig.subplots_adjust(hspace=0.1)
            axs[0,0].plot(P3Res[0],P3Res[5],'.',markersize=2)
            axs[0,0].text(0.01,0.99,cSat[i],transform=axs[0,0].transAxes,ha='left',va='top')
            axs[0,0].text(0.99,0.99,'P3 res',transform=axs[0,0].transAxes,ha='right',va='top')

            pdf.savefig(fig,bbox_inches='tight')
            plt.close(fig)

def PlotTECR(fLog,cSat0,OutFilePrefix,OutFileSuffix):
    '''
    Plot TECR series
    '''

    cSat=[]
    with open(fLog,mode='r',encoding='UTF-16') as fOb:
        for cLine in fOb:
            if cLine[0:10] != 'TECRSeries':
                continue
            cWords=cLine.split()
            if cSat0[0] != 'ALL' and cWords[1] not in cSat0:
                continue
            if cWords[1] not in cSat:
                cSat.append(cWords[1])
    cSat.sort()

    strTmp=OutFilePrefix+OutFileSuffix
    with PdfPages(strTmp) as pdf:
        #For each satellites
        for i in range(len(cSat)):
            #Read data for this sat
            TECR=[[],[]]
            with open(fLog,mode='r',encoding='UTF-16') as fOb:
                for cLine in fOb:
                    if cLine[0:10] != 'TECRSeries':
                        continue
                    cWords=cLine.split()
                    if cWords[1] != cSat[i]:
                        continue
                    #Epoch
                    TECR[0].append(int(cWords[3]))
                    #TECR
                    TECR[1].append(float(cWords[4]))
            fig,axs=plt.subplots(1,1,sharex='col',squeeze=False,figsize=(12,3))
            fig.subplots_adjust(hspace=0.1)
            axs[0,0].plot(TECR[0],TECR[1],'.',markersize=2)
            axs[0,0].text(0.01,0.99,cSat[i],transform=axs[0,0].transAxes,ha='left',va='top')
            axs[0,0].text(0.99,0.99,'TECR',transform=axs[0,0].transAxes,ha='right',va='top')

            pdf.savefig(fig,bbox_inches='tight')
            plt.close(fig)

def PlotEpoDif(fLogList,cSat0,cTyp0,OutFilePrefix,OutFileSuffix):
    '''
    Plot L4 and L6 epoch dif series
    '''
    cSat=[]
    for i in range(len(fLogList)):
        with open(fLogList[i],mode='r',encoding='UTF-16') as fOb:
            for cLine in fOb:
                if cLine[0:8] != 'L4EpoDif' and cLine[0:8] != 'L6EpoDif':
                    continue
                cWords=cLine.split()
                if cSat0[0] != 'ALL' and cWords[1] not in cSat0:
                    continue
                if cWords[1] not in cSat:
                    cSat.append(cWords[1])
    cSat.sort()
    cTyp=[]
    for i in range(len(cSat)):
        cTyp.append([])
    for i in range(len(fLogList)):
        with open(fLogList[i],mode='r',encoding='UTF-16') as fOb:
            for cLine in fOb:
                if cLine[0:8] != 'L4EpoDif' and cLine[0:8] != 'L6EpoDif':
                    continue
                cWords=cLine.split()
                if cWords[1] not in cSat:
                    continue
                if cTyp0[0] != 'ALL ' and cWords[2] not in cTyp0:
                    continue
                iSat=cSat.index(cWords[1])
                if cWords[2] not in cTyp[iSat]:
                    cTyp[iSat].append(cWords[2])
    MP=[]
    for i in range(len(cSat)):
        MP.append([])
        for j in range(len(cTyp[i])):
            MP[i].append([])
            for k in range(2):
                MP[i][j].append([])
                for l in range(2):
                    MP[i][j][k].append([])
    for i in range(len(fLogList)):
        with open(fLogList[i],mode='r',encoding='UTF-16') as fOb:
            for cLine in fOb:
                if cLine[0:8] != 'L4EpoDif' and cLine[0:8] != 'L6EpoDif':
                    continue
                cWords=cLine.split()
                if cWords[1] not in cSat:
                    continue
                iSat=cSat.index(cWords[1])
                if cWords[2] not in cTyp[iSat]:
                    continue
                iTyp=cTyp[iSat].index(cWords[2])
                if cLine[0:8] == 'L4EpoDif':
                    #Epoch
                    MP[iSat][iTyp][0][0].append(int(cWords[3]))
                    #L4 epoch dif
                    MP[iSat][iTyp][0][1].append(float(cWords[8]))
                else:
                    #Epoch
                    MP[iSat][iTyp][1][0].append(int(cWords[3]))
                    #L6 epoch dif
                    MP[iSat][iTyp][1][1].append(float(cWords[8]))
    
    strTmp=OutFilePrefix+OutFileSuffix
    with PdfPages(strTmp) as pdf:
        #For each satellites
        for i in range(len(cSat)):
            nTyp=len(cTyp[i])
            fig,axs=plt.subplots(2,nTyp,sharex='col',squeeze=False,figsize=(nTyp*6,2*1.5))
            fig.subplots_adjust(hspace=0.1)
            for j in range(nTyp):
                #L4 epoch dif
                axs[0,j].plot(MP[i][j][0][0],MP[i][j][0][1],'.',markersize=2)
                axs[0,j].text(0.01,0.99,cSat[i]+' '+cTyp[i][j]+' L4',
                              transform=axs[0,j].transAxes,ha='left',va='top')
                if j==0:
                    axs[0,j].set_ylabel('[m]')
                x1,x2=axs[0,j].get_xlim()
                axs[0,j].set_xlim(left=x1,right=x2)
                axs[0,j].hlines(0,x1,x2,colors=['darkgray'],linestyle='dashed')

                #L6 epoch dif
                Mea=np.mean(MP[i][j][1][1])
                Med=np.median(MP[i][j][1][1])
                Std=np.std(MP[i][j][1][1])
                RMS=0.0
                MAD=np.median(np.abs(MP[i][j][1][1]-Med))/0.6745
                nL6=len(MP[i][j][1][1])
                for l in range(nL6):
                    RMS=RMS+MP[i][j][1][1][l]*MP[i][j][1][1][l]
                RMS=np.sqrt(RMS/nL6)
                axs[1,j].plot(MP[i][j][1][0],MP[i][j][1][1],'.',markersize=2)
                axs[1,j].text(0.01,0.99,cSat[i]+' '+cTyp[i][j]+' L6',
                              transform=axs[1,j].transAxes,ha='left',va='top')
                cStr=cSat[i]+' '+cTyp[i][j]+' DifL6 '+ \
                'Mea={:>9.3f} STD={:>8.3f} RMS={:>8.3f} Med={:>9.3f} MAD={:>8.3f}'.format(Mea,Std,RMS,Med,MAD)
                print(cStr)
                if j==0:
                    axs[1,j].set_ylabel('[m]')
                x1,x2=axs[1,j].get_xlim()
                axs[1,j].set_xlim(left=x1,right=x2)
                axs[1,j].hlines(0,x1,x2,colors=['darkgray'],linestyle='dashed')
                axs[1,j].set_xlabel('Epoch')
            pdf.savefig(fig,bbox_inches='tight')
            plt.close(fig)

if __name__ == '__main__':
    import argparse

    InFilePrefix=r'D:/Code/PROJECT/WORK2019001/'
    fLogList=glob.glob(InFilePrefix+'abpo_log')
    OutFilePrefix=r'D:/Code/PROJECT/WORK2019001/'
    cSat=['ALL']
    cTyp=['LC13']
    # OutFileSuffix='MP.pdf'
    # PlotMP(fLogList,cSat,cTyp,True,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='CombDif.pdf'
    # PlotEpoDif(fLogList,cSat,cTyp,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='CombL4.pdf'
    # PlotComb(fLogList,cSat,cTyp,'L4Series',False,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='CombL6.pdf'
    # PlotComb(fLogList,cSat,cTyp,'L6Series',False,OutFilePrefix,OutFileSuffix)

    # fLog=r'D:/Code/PROJECT/WORK2019164/ffmj_log'
    # OutFileSuffix='CombP4.pdf'
    # PlotP4(fLog,cSat,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='CombP3.pdf'
    # PlotP3(fLog,cSat,True,OutFilePrefix,OutFileSuffix)
    # OutFileSuffix='TECR.pdf'
    # PlotTECR(fLog,cSat,OutFilePrefix,OutFileSuffix)
    OutFileSuffix='S1S2.pdf'
    PlotS1S2(fLogList,cSat,cTyp,OutFilePrefix,OutFileSuffix)


