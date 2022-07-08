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
import datetime
import shutil


def CopyRnxO(fCtrl,SrcPath,DesPath,DOY,YR):
    '''
    Copy a set of RnxO files from source path (SrcPath) to destination path (DesPath).
    Station list is extracted from the cf-file (fCtrl) and the date is specified via
    Day of Year (DOY) and 2-digital year (YR).
    '''

    #Get the station list
    cSta=[]
    with open(fCtrl,mode='rt') as fOb:
        lBeg=False
        for cLine in fOb:
            if cLine[0:13] == '+Station used':
                lBeg=True
            elif cLine[0:13] == '-Station used':
                break
            elif lBeg and cLine[0:1]==' ' and len(cLine)>=5:
                if cLine[1:5] not in  cSta:
                    cSta.append(cLine[1:5])

    nSta=len(cSta); nStaCopied=0
    for i in range(nSta):
        fObs=os.path.join(SrcPath,cSta[i].lower()+'{:03d}0.{:02d}o'.format(DOY,YR))
        if not os.path.isfile(fObs):
            print(fObs+' does not exist!')
        else:
            shutil.copy(fObs,DesPath)
            nStaCopied=nStaCopied+1
    return nStaCopied

if __name__ == '__main__':
    import argparse

    fCtrl=r'Y:/PRO_2019001_2020366_WORK/I2/WORK2019335_ERROR/cf_net'
    SrcPath=r'Y:/MGEX/2019/335'
    DesPath=r'Y:/PRO_2019001_2020366_WORK/I2/OBS/2019/335/'
    DOY=335; YR=19
    nSta=CopyRnxO(fCtrl,SrcPath,DesPath,DOY,YR)
    print('{:d} stations copied'.format(nSta))