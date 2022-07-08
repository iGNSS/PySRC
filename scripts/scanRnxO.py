#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Scan GNSS observation RINEX files in the specified directory using
FORTRAN program 'ScanRnxObs'
'''
__author__ = 'hanbing'

# Standard library imports
import os.path
import os
import sys
import fnmatch
import subprocess

# Local application/library specific imports
from PySRC.miscellaneous import GNSSTime

if __name__ == '__main__':
    import argparse

    parser=argparse.ArgumentParser()
    parser.add_argument('--mjd',type=int,required=True,
                        help='Begin date in MJD')
    parser.add_argument('--days',type=int,required=True,
                        help='Number of following days from MJD (included)')
    parser.add_argument('--ObsPath',type=str,required=True,
                        help='Observation path')
    parser.add_argument('--WrkPath',type=str,required=True,
                        help='Work path')
    parser.add_argument('--OutPath',type=str,help='Output path')

    # For debug
    args=parser.parse_args()
    #args=parser.parse_args(['--mjd','58583','--days','1',
    #                        '--ObsPath','E:/PPP/OBS',
    #                        '--WrkPath','D:/Code/PROJECT/WORK'])
    #parser.print_help()

    # Do some checks
    if not os.path.isdir(args.ObsPath):
        sys.exit('Path '+args.ObsPath+' does not exist!')
    elif not os.path.isdir(args.WrkPath):
        sys.exit('Path '+args.WrkPath+' does not exist!')
    else:
        # Change to work path
        os.chdir(args.WrkPath)
        if not os.path.isfile('file_table'):
            sys.exit('File file_table does not exist!')

    mjd=args.mjd
    while mjd<=(args.mjd+args.days-1):
        year,doy=GNSSTime.mjd2doy(mjd)
        YYYY=str(year)
        YR=str(GNSSTime.year2yr(year)).zfill(2)
        DDD=str(doy).zfill(3)
        ObsPath=os.path.join(args.ObsPath,YYYY,DDD)
        if not os.path.isdir(ObsPath):
            print('Path '+ObsPath+' does not exist!')
            continue
        if args.OutPath != None and os.path.isdir(args.OutPath):
            OutPath=args.OutPath
        else:
            OutPath=ObsPath
        OutFile1=os.path.join(OutPath,YYYY+DDD+'.scan')
        if os.path.isfile(OutFile1):
            os.remove(OutFile1)
        OutFile2=os.path.join(OutPath,YYYY+DDD+'.sta')
        if os.path.isfile(OutFile2):
            os.remove(OutFile2)
        ObsFile=os.listdir(ObsPath)
        for f0 in ObsFile:
            if not fnmatch.fnmatchcase(f0,'????'+DDD+'0.'+YR+'o'):
                continue
            f=os.path.join(ObsPath,f0)
            cmd=['ScanRnxObs','-fRnxO',f,'-Sys','C','-fScan',OutFile1,
                 '-fSta',OutFile2]
            cPro=subprocess.run(cmd)
            if cPro.returncode != 0:
                print('Failed to scan '+f)
        # Extrac from scan file
        if os.path.isfile(OutFile1):
            OutFile3=os.path.join(OutPath,YYYY+DDD+'.scan1')
            if os.path.isfile(OutFile3):
                os.remove(OutFile3)
            cmd=['extscan','-fScan',OutFile1,'-fOut',OutFile3]
            cPro=subprocess.run(cmd)
            if cPro.returncode != 0:
                print('Failed to extscan on '+OutFile1)
        # Next day
        mjd=mjd+1