#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Compare the models for BDS satellite-induced code pseudorange variations
proposed by different papers
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

def GDV(cBlk,Elev,iModel):
    '''
    Calculate the GDV correction for a specific BDS satellite type

    cBlk --- BDS-2G/BDS-2I/BDS-2M
    Elev --- Elevation angle, degrees
   iMode --- 1, Wan2015 model; 2, Lou2017 model
    '''

    #Model 1 Wan2015
    M1=np.array([(-0.55,-0.71,-0.27,-0.47,-0.40,-0.22),
                 (-0.40,-0.36,-0.23,-0.38,-0.31,-0.15),
                 (-0.34,-0.33,-0.21,-0.32,-0.26,-0.13),
                 (-0.23,-0.19,-0.15,-0.23,-0.18,-0.10),
                 (-0.15,-0.14,-0.11,-0.11,-0.06,-0.04),
                 (-0.04,-0.03,-0.04, 0.06, 0.09, 0.05),
                 ( 0.09, 0.08, 0.05, 0.34, 0.28, 0.14),
                 ( 0.19, 0.17, 0.14, 0.69, 0.48, 0.27),
                 ( 0.27, 0.24, 0.19, 0.97, 0.64, 0.36),
                 ( 0.35, 0.33, 0.32, 1.05, 0.69, 0.47)])
    #Model 2 Lou2017
    M2=np.array([(-0.436,-0.275,-0.048,-0.590,-0.257,-0.102,-0.946,-0.598,-0.177),
                 ( 1.158, 1.087, 0.566, 1.624, 0.995, 0.748, 2.158, 1.635, 0.652),
                 (-0.333,-0.452,-0.185,-0.645,-0.381,-0.307,-0.642,-0.556,-0.178)])

    dCor=np.zeros(3)
    if iModel==1:
        if cBlk=='BDS-2I' or cBlk=='BDS-2M':
            i=np.int8(np.floor(Elev/10.0))
            if i==9:
                i=8
            if cBlk=='BDS-2I':
                j=0
            else:
                j=3
            alpha=Elev/10.0-i
            for k in range(3):
                dCor[k]=M1[i,j+k] + alpha*(M1[i+1,j+k]-M1[i,j+k])
    else:
        a1=np.deg2rad(Elev)
        a2=a1*a1
        a3=a2*a1
        if cBlk=='BDS-2G':
            i=0
        elif cBlk=='BDS-2I':
            i=3
        else:
            i=6
        for k in range(3):
            j=i+k
            dCor[k]=M2[0,j]*a1 + M2[1,j]*a2 + M2[2,j]*a3
    return dCor

def PlotGDV(lNorm,OutFilePrefix,OutFileSuffix):
    '''
    Plot the two GDV models

    lNorm --- Whether align the two models (at elev=0 deg)
    '''

    B1=[]
    for i in range(10):
        B1.append([])
    B2=[]
    for i in range(10):
        B2.append([])

    #Sampling
    dN=np.zeros(6)
    for Elev in range(90):
        #Model 1 Wan2015
        #BDS-2G B1-B3 (NONE)
        B1[0].append(Elev)
        B1[1].append(np.nan)
        B1[2].append(np.nan)
        B1[3].append(np.nan)

        #BDS-2I B1-B3
        dCor=GDV('BDS-2I',Elev,1)
        if lNorm and Elev==0:
            dN[0:3]=dCor[0:3]
        B1[4].append(dCor[0]-dN[0])
        B1[5].append(dCor[1]-dN[1])
        B1[6].append(dCor[2]-dN[2])

        #BDS-2M B1-B3
        dCor=GDV('BDS-2M',Elev,1)
        if lNorm and Elev==0:
            dN[3:6]=dCor[0:3]
        B1[7].append(dCor[0]-dN[3])
        B1[8].append(dCor[1]-dN[4])
        B1[9].append(dCor[2]-dN[5])

        #Model 2
        dCor=GDV('BDS-2G',Elev,2)
        B2[0].append(Elev)
        B2[1].append(dCor[0])
        B2[2].append(dCor[1])
        B2[3].append(dCor[2])
        dCor=GDV('BDS-2I',Elev,2)
        B2[4].append(dCor[0])
        B2[5].append(dCor[1])
        B2[6].append(dCor[2])
        dCor=GDV('BDS-2M',Elev,2)
        B2[7].append(dCor[0])
        B2[8].append(dCor[1])
        B2[9].append(dCor[2])

    fig,axs=plt.subplots(3,3,sharex='col',sharey='row',squeeze=False,figsize=(3*4,3*3))
    fig.subplots_adjust(hspace=0.1)

    #BDS-2G
    axs[0,0].text(0.01,0.99,'BDS-2G B1',transform=axs[0,0].transAxes,ha='left',va='top')
    axs[0,0].plot(B1[0],B1[1],label='Wan2015',linestyle='-',linewidth=2.5)
    axs[0,0].plot(B2[0],B2[1],label='Lou2017',linestyle='--',linewidth=2.5)
    axs[0,0].legend(loc='lower right')
    axs[0,0].set_xlim(left=0,right=90)
    axs[0,0].set_ylim(bottom=-0.5,top=1.5)
    axs[0,0].hlines(0,0,90,colors=['darkgray'],linestyle='dashed')
    axs[0,0].set_ylabel('[m]')

    axs[0,1].text(0.01,0.99,'BDS-2G B2',transform=axs[0,1].transAxes,ha='left',va='top')
    axs[0,1].plot(B1[0],B1[2],label='Wan2015',linestyle='-',linewidth=2.5)
    axs[0,1].plot(B2[0],B2[2],label='Lou2017',linestyle='--',linewidth=2.5)
    axs[0,1].legend(loc='lower right')
    axs[0,1].set_xlim(left=0,right=90)
    axs[0,1].set_ylim(bottom=-0.5,top=1.5)
    axs[0,1].hlines(0,0,90,colors=['darkgray'],linestyle='dashed')

    axs[0,2].text(0.01,0.99,'BDS-2G B3',transform=axs[0,2].transAxes,ha='left',va='top')
    axs[0,2].plot(B1[0],B1[3],label='Wan2015',linestyle='-',linewidth=2.5)
    axs[0,2].plot(B2[0],B2[3],label='Lou2017',linestyle='--',linewidth=2.5)
    axs[0,2].legend(loc='lower right')
    axs[0,2].set_xlim(left=0,right=90)
    axs[0,2].set_ylim(bottom=-0.5,top=1.5)
    axs[0,2].hlines(0,0,90,colors=['darkgray'],linestyle='dashed')

    #BDS-2I
    axs[1,0].text(0.01,0.99,'BDS-2I B1',transform=axs[1,0].transAxes,ha='left',va='top')
    axs[1,0].plot(B1[0],B1[4],label='Wan2015',linestyle='-',linewidth=2.5)
    axs[1,0].plot(B2[0],B2[4],label='Lou2017',linestyle='--',linewidth=2.5)
    axs[1,0].legend(loc='lower right')
    axs[1,0].set_xlim(left=0,right=90)
    axs[1,0].set_ylim(bottom=-0.5,top=1.5)
    axs[1,0].hlines(0,0,90,colors=['darkgray'],linestyle='dashed')
    axs[1,0].set_ylabel('[m]')

    axs[1,1].text(0.01,0.99,'BDS-2I B2',transform=axs[1,1].transAxes,ha='left',va='top')
    axs[1,1].plot(B1[0],B1[5],label='Wan2015',linestyle='-',linewidth=2.5)
    axs[1,1].plot(B2[0],B2[5],label='Lou2017',linestyle='--',linewidth=2.5)
    axs[1,1].legend(loc='lower right')
    axs[1,1].set_xlim(left=0,right=90)
    axs[1,1].set_ylim(bottom=-0.5,top=1.5)
    axs[1,1].hlines(0,0,90,colors=['darkgray'],linestyle='dashed')

    axs[1,2].text(0.01,0.99,'BDS-2I B3',transform=axs[1,2].transAxes,ha='left',va='top')
    axs[1,2].plot(B1[0],B1[6],label='Wan2015',linestyle='-',linewidth=2.5)
    axs[1,2].plot(B2[0],B2[6],label='Lou2017',linestyle='--',linewidth=2.5)
    axs[1,2].legend(loc='lower right')
    axs[1,2].set_xlim(left=0,right=90)
    axs[1,2].set_ylim(bottom=-0.5,top=1.5)
    axs[1,2].hlines(0,0,90,colors=['darkgray'],linestyle='dashed')

    #BDS-2M
    axs[2,0].text(0.01,0.99,'BDS-2M B1',transform=axs[2,0].transAxes,ha='left',va='top')
    axs[2,0].plot(B1[0],B1[7],label='Wan2015',linestyle='-',linewidth=2.5)
    axs[2,0].plot(B2[0],B2[7],label='Lou2017',linestyle='--',linewidth=2.5)
    axs[2,0].legend(loc='lower right')
    axs[2,0].set_xlim(left=0,right=90)
    axs[2,0].set_ylim(bottom=-0.5,top=1.5)
    axs[2,0].hlines(0,0,90,colors=['darkgray'],linestyle='dashed')
    axs[2,0].set_ylabel('[m]')
    axs[2,0].set_xlabel('Elevation [deg]')

    axs[2,1].text(0.01,0.99,'BDS-2M B2',transform=axs[2,1].transAxes,ha='left',va='top')
    axs[2,1].plot(B1[0],B1[8],label='Wan2015',linestyle='-',linewidth=2.5)
    axs[2,1].plot(B2[0],B2[8],label='Lou2017',linestyle='--',linewidth=2.5)
    axs[2,1].legend(loc='lower right')
    axs[2,1].set_xlim(left=0,right=90)
    axs[2,1].set_ylim(bottom=-0.5,top=1.5)
    axs[2,1].hlines(0,0,90,colors=['darkgray'],linestyle='dashed')
    axs[2,1].set_xlabel('Elevation [deg]')

    axs[2,2].text(0.01,0.99,'BDS-2M B3',transform=axs[2,2].transAxes,ha='left',va='top')
    axs[2,2].plot(B1[0],B1[9],label='Wan2015',linestyle='-',linewidth=2.5)
    axs[2,2].plot(B2[0],B2[9],label='Lou2017',linestyle='--',linewidth=2.5)
    axs[2,2].legend(loc='lower right')
    axs[2,2].set_xlim(left=0,right=90)
    axs[2,2].set_ylim(bottom=-0.5,top=1.5)
    axs[2,2].hlines(0,0,90,colors=['darkgray'],linestyle='dashed')
    axs[2,2].set_xlabel('Elevation [deg]')

    strTmp=OutFilePrefix+OutFileSuffix
    fig.savefig(strTmp,transparent=True,bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    import argparse

    OutFilePrefix=r'D:/Code/PROJECT/WORK2019164/'
    OutFileSuffix='GDV.png'
    PlotGDV(True,OutFilePrefix,OutFileSuffix)