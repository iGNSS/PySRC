#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Do some process on ISL observation files
'''
__author__ = 'hanbing'

# Standard library imports
import subprocess
import os
import sys
import os.path
import glob
import datetime


def CheckDupObs(fISLList,OutFile,lExclDup):
    '''
    Check duplicated ISL obs, i.e. more than one record for the same link
    at a specific epoch.

    NOTE: Here we do not discriminate the tran or recv end, i.e.
          cSat1-cSat2 and cSat2-cSat1 are regarded as the same link
    '''

    fOut=open(OutFile,mode='w')

    nFile=len(fISLList); nDupRec=0
    for i in range(nFile):
        rEpo=[]
        if lExclDup:
            fNew=open(fISLList[i]+'_new',mode='w')
        with open(fISLList[i],mode='rt') as fOb:
            for cLine in fOb:
                if len(cLine) < 30:
                    continue
                cWords=cLine.split()
                t=int(cWords[0])+float(cWords[1])/86400.0
                if len(rEpo)==0:
                    #the first epoch
                    rEpo.append(cLine[0:24])
                    cLink=[]; cLink0=[]; RelRng=[]; RelClk=[]
                elif rEpo[-1] != cLine[0:24]:
                    #New epoch
                    ## Firstly, report the result of last epoch
                    for j in range(len(cLink)):
                        if len(RelRng[j]) == 1:
                            if lExclDup:
                                fNew.write('{:24s} GPS {:7s} {:16.4f} {:16.4f}\n'.format(
                                            rEpo[-1],cLink[j],RelRng[j][0],RelClk[j][0]))
                            continue
                        #More than one obs for this link at the same epoch
                        nDupRec=nDupRec+1
                        for k in range(len(RelRng[j])):
                            if k==0:
                                #Take the first one as reference
                                continue
                            d1=RelRng[j][k]-RelRng[j][0]
                            d2=RelClk[j][k]-RelClk[j][0]
                            fOut.write('{:24s} GPS {:7s} {:16.4f} {:16.4f} {:16.4f} {:16.4f}\n'.format(
                                         rEpo[-1],cLink0[j][k],RelRng[j][k],RelClk[j][k],d1,d2))
                    ## Then, start for this epoch
                    rEpo.append(cLine[0:24])
                    cLink=[]; cLink0=[]; RelRng=[]; RelClk=[]
                if cWords[3][0:3]+'-'+cWords[3][4:7] not in cLink and \
                   cWords[3][4:7]+'-'+cWords[3][0:3] not in cLink:
                    #New link
                    cLink.append(cWords[3][0:3]+'-'+cWords[3][4:7])
                    cLink0.append([]); RelRng.append([]); RelClk.append([])
                    iLink=len(cLink)-1
                elif cWords[3][0:3]+'-'+cWords[3][4:7] in cLink:
                    #Existing link
                    iLink=cLink.index(cWords[3][0:3]+'-'+cWords[3][4:7])
                else:
                    #Existing link
                    iLink=cLink.index(cWords[3][4:7]+'-'+cWords[3][0:3])
                cLink0[iLink].append(cWords[3][0:3]+'-'+cWords[3][4:7])
                RelRng[iLink].append(float(cWords[4]))
                RelClk[iLink].append(float(cWords[5]))
            ## Report the result of the last epoch
            for j in range(len(cLink)):
                if len(RelRng[j]) == 1:
                    if lExclDup:
                        fNew.write('{:24s} GPS {:7s} {:16.4f} {:16.4f}\n'.format(
                                    rEpo[-1],cLink[j],RelRng[j][0],RelClk[j][0]))
                    continue
                nDupRec=nDupRec+1
                for k in range(len(RelRng[j])):
                    if k==0:
                        continue
                    d1=RelRng[j][k]-RelRng[j][0]
                    d2=RelClk[j][k]-RelClk[j][0]
                    fOut.write('{:24s} GPS {:7s} {:16.4f} {:16.4f} {:16.4f} {:16.4f}\n'.format(
                                rEpo[-1],cLink0[j][k],RelRng[j][k],RelClk[j][k],d1,d2))
            if lExclDup:
                fNew.close()
    fOut.close()
    print('{:>6d} records have been duplicated'.format(nDupRec))

if __name__ == '__main__':
    import argparse

    # OutFilePrefix='E:/ISL/results/'

    InFilePrefix=r'E:/ISL/'
    fISLList=[]
    fISLList=glob.glob(InFilePrefix+'ISL_2019???')
    OutFile='D:/Code/PROJECT/WORK_ISL/DupRec'
    CheckDupObs(fISLList,OutFile,True)