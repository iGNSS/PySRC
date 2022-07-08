#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Get information from ctrl-file
'''
__author__ = 'hanbing'

# Standard library imports
import os
import sys
import os.path
import glob
import datetime
import math

# Local application/library specific imports
from PySRC.miscellaneous import GNSSTime


def GetStaRec0(fCtrl,cSta):
    '''
    Get the rec && ant info for specified station list

    cSta --- Station list, in upper-case

    Return List: ('****' means not found)
    # [0], Rec type
    # [1], Rec no.
    # [2], Rec firmware version
    # [3], Ant type
    '''

    # Rec type, Rec no., Rec firmware version, Ant type
    cRec=[[],[],[],[]]
    for i in range(len(cSta)):
        for j in range(4):
            cRec[j].append('****')

    with open(fCtrl,mode='rt') as fOb:
        lBeg=False
        for cLine in fOb:
            if cLine[0:12] == '+GPS station':
                lBeg=True
            elif cLine[0:12] == '-GPS station':
                break
            elif lBeg:
                if cLine[0:1] !=' ':
                    continue
                if cLine[1:5] not in cSta:
                    continue
                iSta=cSta.index(cLine[1:5])
                if cLine[6:9]=='ENU':
                    #Antenna type
                    cRec[3][iSta]=cLine[55:75]
                elif cLine[6:9]=='RCV':
                    #Receiver type
                    cRec[0][iSta]=cLine[10:30]
                    #Receiver number
                    cRec[1][iSta]=cLine[31:43]
                    #Receiver firmware version
                    cRec[2][iSta]=cLine[44:64]

    # Check if all rec && ant info are found
    for i in range(len(cSta)):
        if cRec[0][i]=='****':
            print('Rec type not found for '+cSta[i])
        if cRec[3][i]=='****':
            print('Ant type not found for '+cSta[i])

    return cRec

def GetStaRec1(fCtrl):
    '''
    Get the rec && ant info for all stations

    Return List:

    # [0], Rec type
    # [1], Rec no.
    # [2], Rec firmware version
    # [3], Ant type

    and station list
    '''

    # Rec type, Rec no., Rec firmware version, Ant type
    cRec=[[],[],[],[]]; cSta=[]
    with open(fCtrl,mode='rt') as fOb:
        lBeg=False
        for cLine in fOb:
            if cLine[0:12] == '+GPS station':
                lBeg=True
            elif cLine[0:12] == '-GPS station':
                break
            elif lBeg:
                if cLine[0:1] !=' ':
                    continue
                if cLine[1:5] not in cSta:
                    cSta.append(cLine[1:5])
                    for i in range(4):
                        cRec[i].append('****')
                iSta=cSta.index(cLine[1:5])
                if cLine[6:9]=='ENU':
                    #Antenna type
                    cRec[3][iSta]=cLine[55:75]
                elif cLine[6:9]=='RCV':
                    #Receiver type
                    cRec[0][iSta]=cLine[10:30]
                    #Receiver number
                    cRec[1][iSta]=cLine[31:43]
                    #Receiver firmware version
                    cRec[2][iSta]=cLine[44:64]

    # Check if all rec && ant info are found
    for i in range(len(cSta)):
        if cRec[0][i]=='****':
            print('Rec type not found for '+cSta[i])
        if cRec[3][i]=='****':
            print('Ant type not found for '+cSta[i])

    return cSta,cRec

def PrintStaRec0(fPathWrk,MJD0,nDay,iRec,iPrint,OutFilePrefix,OutFileSuffix):
    '''
    Print the equipment history for a session based on ctrl-files

    fPathWrk --- the directory of the work

        iRec --- Specify which equippent info is required
                 # [0], Rec type
                 # [1], Rec no.
                 # [2], Rec firmware version
                 # [3], Ant type
      iPrint --- What info should be printed
                 # 0, the equipment list
                 # 1, the equipment change

    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)
    fOut=open(OutFilePrefix+os.path.splitext(OutFileSuffix)[0],'w')

    cSta=[]; cRec=[]
    for i in range(nDay):
        MJD=MJD0+i
        YYYY,DOY=GNSSTime.mjd2doy(MJD)
        fCtr=os.path.join(fPathWrk,'WORK{:4d}{:03d}'.format(YYYY,DOY),'cf_net')
        cSta0,cRec0=GetStaRec1(fCtr)
        for j in range(len(cSta0)):
            if cSta0[j] not in cSta:
                cSta.append(cSta0[j]); cRec.append([])
                for k in range(nDay):
                    cRec[len(cSta)-1].append('****')
            iSta=cSta.index(cSta0[j])
            cRec[iSta][i]=cRec0[iRec][j]

    # Print
    if iPrint==0:
        cStr='        '
        for i in range(len(cSta)):
            cStr=cStr+' {:<20s}'.format(cSta[i])
        fOut.write(cStr+'\n')
        for i in range(nDay):
            MJD=MJD0+i
            YYYY,DOY=GNSSTime.mjd2doy(MJD)
            cStr='{:4d} {:03d}'.format(YYYY,DOY)
            for j in range(len(cSta)):
                cStr=cStr+' {: <20s}'.format(cRec[j][i])
            fOut.write(cStr+'\n')
    elif iPrint==1:
        # Only print the equipment change
        for i in range(len(cSta)):
            # Start check from the day after the first available day
            k=0
            for j in range(nDay):
                if cRec[i][j]=='****':
                    continue
                k=j; break
            for j in range(k+1,nDay):
                if cRec[i][j] in cRec[i][0:j] or cRec[i][j]=='****':
                    continue
                MJD=MJD0+j
                YYYY,DOY=GNSSTime.mjd2doy(MJD)
                cStr=cSta[i]+' {:4d}{:03d}'.format(YYYY,DOY)+' {:<20s}'.format(cRec[i][j])
                fOut.write(cStr+'\n')
    fOut.close()

def CombineStaListGMT(fList,OutFilePrefix,OutFileSuffix):
    '''
    Combine the station lists to generate a comprehensive list. The input
    station lists are in the GMT format, i.e., for each station line,
    1X,F5.1,1X,F6.1,1X,A24,1X,A4

    The output is aimed to plot a network map containing all stations.

    One same site may have slightly different coordinates in different lists.
    It will be coped with properly here.
    '''

    if not os.path.isdir(OutFilePrefix):
        os.makedirs(OutFilePrefix)

    nFile=len(fList)
    cSta=[]; xPos=[[],[]]; cFmt=[]
    for i in range(nFile):
        with open(fList[i],mode='rt') as fOb:
            for cLine in fOb:
                if cLine[0:1] != ' ':
                    continue
                if cLine[39:43] not in cSta:
                    # New station
                    cSta.append(cLine[39:43])
                    # Longitute
                    xPos[0].append(float(cLine[1:6]))
                    # Latitude
                    xPos[1].append(float(cLine[7:13]))
                    # Plot format
                    cFmt.append(cLine[14:38])
                else:
                    # Old station
                    iSta=cSta.index(cLine[39:43])
                    # Check the pos difference
                    if math.fabs(xPos[0][iSta]-float(cLine[1:6])) > 0.15:
                        print('Longitude diff found for '+cLine[39:43]+' in '+fList[i])
                    if math.fabs(xPos[1][iSta]-float(cLine[7:13])) > 0.15:
                        print('Latitude diff found for  '+cLine[39:43]+' in '+fList[i])
    fOut=open(OutFilePrefix+os.path.splitext(OutFileSuffix)[0],'w')
    nSta=len(cSta); cSta0=cSta.copy(); cSta0.sort()
    fOut.write('# {: >3d} stations synthesized from lists\n'.format(nSta))
    for i in range(nSta):
        j=cSta.index(cSta0[i])
        fOut.write(' {: >5.1f} {: >6.1f} {:>24s} {:>4s}\n'.format(xPos[0][j],
                   xPos[1][j],cFmt[j],cSta[j]))



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

    OutFilePrefix=os.path.join(cDskPre0,r'PRO_2019001_2020366/C4/STA/')

    fPathWrk=os.path.join(cWrkPre0,r'PRO_2019001_2020366_WORK/C4')
    OutFileSuffix='StaRecHistory'
    PrintStaRec0(fPathWrk,58818,31,0,0,OutFilePrefix,OutFileSuffix)

    InFilePrefix=os.path.join(cWrkPre0,r'PRO_2019001_2020366_WORK/C21/')
    fList=glob.glob(InFilePrefix+'WORK20193??/20193??_UsedSta')

    # OutFileSuffix='StaList_ALL'
    # CombineStaListGMT(fList,OutFilePrefix,OutFileSuffix)