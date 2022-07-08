#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Download GNSS observation files
'''
__author__ = 'hanbing'

# Standard library imports
import os
import sys
import subprocess
import fnmatch
import gzip
# Related third party imports

# Local application/library specific imports
from PySRC.miscellaneous import GNSSTime

def curl1File(DesPath: str,fURL: list,f: str,fNetrc: str) -> int :
    '''Download a file specified by a URL via curl.
    DesPath: Local directory for storing the downloaded file
    fURL   : Remote directory to download the file
    f      : File to be downloaded from remote directory
    fNetrc : netrc file for authentication
    '''
    strTmp=os.path.join(DesPath,f)
    print('Try '+fURL[0])
    cmd=['curl','-c','curl_cookies','--netrc-file',fNetrc,'--progress-bar',
         '-o',strTmp,'-L','-f','-m','300',fURL[0]+f]
    cPro=subprocess.run(cmd)
    if cPro.returncode!=0:
        if os.path.isfile(strTmp):
            os.remove(strTmp)
        # Try 2nd source
        print('Try '+fURL[1])
        cmd=['curl','--netrc-optional','--progress-bar','-o',strTmp,'-L',
             '-m','300',fURL[1]+f]
        cPro=subprocess.run(cmd)
        if cPro.returncode!=0:
            if os.path.isfile(strTmp):
                os.remove(strTmp)
            # Try 3th source
            print('Try '+fURL[2])
            cmd=['curl','--netrc-optional','--progress-bar','-o',strTmp,'-L',
                 '-m','300',fURL[2]+f]
            cPro=subprocess.run(cmd)

    return cPro.returncode

def downloadRnxO1Day1Sta(YYYY,DOY,cSta):
    '''
    '''
    pass

def downloadRnxO1DayNSta(year,doy,fSta,fNetrc,DesPath) -> int:
    '''Download 1 day RINEX observation files for specified stations (
    via station list file, fSta) to local destination directory DesPath.
    Current directory is the working directory where stores the cookies file,
    curl_cookies and file list of the target remote direcotry, curl_list.
    '''

    if not os.path.isfile(fSta):
        sys.exit('File '+fSta+' does not exist!')
    elif not os.path.isfile(fNetrc):
        sys.exit('File '+fNetrc+' does not exist!')
    elif not os.path.isdir(DesPath):
        os.makedirs(DesPath)

    YYYY=str(year)
    YR=str(GNSSTime.year2yr(year)).zfill(2)
    DDD=str(doy).zfill(3)
    # Remote directory
    Source=[]
    Source1='https://cddis.nasa.gov/archive/gps/data/daily/'+YYYY+'/'+DDD+'/'+\
        YR+'d/'
    Source.append(Source1)
    Source2='ftp://cddis.nasa.gov/gps/data/daily/'+YYYY+'/'+DDD+'/'+YR+'d/'
    Source.append(Source2)
    Source3='ftp://cddis.gsfc.nasa.gov/pub/gps/data/daily/'+YYYY+'/'+DDD+'/'+\
        YR+'d/'
    Source.append(Source3)

    # Get the file list from remote directory
    cmd=['curl','-c','curl_cookies','--netrc-file',fNetrc,'--progress-bar',
         '-o','curl_list','-L',Source1+'*?list']
    cPro=subprocess.run(cmd)

    if cPro.returncode != 0:
        sys.exit('Failed to get the file list from '+Source1)
    elif not os.path.isfile('curl_list'):
        sys.exit('File curl_list does not exist!')

    strTmp='FailedSta_'+YYYY+DDD
    with open(fSta,'rt') as fOb1, open(strTmp,'wt') as fOb3:
        for cLine in fOb1:
            if cLine[0]!=' ' or len(cLine)<5:
                continue
            # Deal with every station
            s=cLine[1:5].lower()
            # Pattern for short observation file name
            fShrt=s+DDD+'0.'+YR+'d.Z'
            S=cLine[1:5].upper()
            # Pattern for long observation file name
            fLong=S+'[0-9][0-9]???_[RSU]_'+YYYY+DDD+\
                '[0-2][0-9][0-5][0-9]_[0-9][0-9][YDHMU]_'+\
                '[0-9][0-9][CZSMHDU]_[CM][O].[rc][nr]x.gz'

            fList=[]
            nShrt=0
            iShrt=[]
            nLong=0
            iLong=[]
            with open('curl_list','rt') as fOb2:
                for cLine in fOb2:
                    if cLine[0]=='#' or len(cLine)<14:
                        continue
                    words=cLine.split()
                    if fnmatch.fnmatchcase(words[0],fLong):
                        fList.append(words[0])
                        nLong=nLong+1
                        iLong.append(nLong+nShrt-1)
                    elif fnmatch.fnmatch(words[0],fShrt):
                        fList.append(words[0])
                        nShrt=nShrt+1
                        iShrt.append(nLong+nShrt-1)
            if len(fList)==0:
                print('Observation file for '+S+' does not exist!')
                fOb3.write('?'+S+'\n')
                continue

            if nLong>1:
                f=fList[iLong[0]]
                print('Multiple longname file exist, '+
                      'try only the first found one!')
            elif nLong==1:
                f=fList[iLong[0]]
            elif nShrt>1:
                f=fList[iShrt[0]]
                print('Multiple shortname file exist, '+
                      'try only the first found one!')
            elif nShrt==1:
                f=fList[iShrt[0]]
            # Remove the old one if exist
            f0=os.path.join(DesPath,f)
            if os.path.isfile(f0):
                os.remove(f0)
            # Download the file
            print('Downloading '+f)
            iReturn=curl1File(DesPath,Source,f,fNetrc)
            if iReturn!=0:
                print('Failed to download '+f)
                fOb3.write(' '+S+'\n')
                if os.path.isfile(f0):
                    os.remove(f0)
            else:
                # Decompress
                if '.gz' in f0:
                    # gzip file
                    f=f0.replace('.gz','')
                    try:
                        gf0=gzip.open(f0,mode='rt')
                        fOb=open(f,'wt')
                        fOb.write(gf0.read())
                        gf0.close()
                        fOb.close()
                    except OSError:
                        print('Failed to gzip '+f0)
                        if os.path.isfile(f):
                            os.remove(f)
                        continue
                else:
                    print('Not able to decompress '+f0)
                    continue
                # Remove the gzip file
                os.remove(f0)

                f0=os.path.join(DesPath,s+DDD+'0.'+YR+'o')
                if '.rnx' in f:
                    os.rename(f,f0)
                else:
                    # Decompress the crx file
                    cmd=['crx2rnx',f,'-']
                    with open(f0,'wt') as fOb:
                        cPro=subprocess.run(cmd,text=True,stdout=fOb)
                    if cPro.returncode==0:
                        os.remove(f)
                    else:
                        print('Failed to crx2rnx '+f)
                        if os.path.isfile(f0):
                            os.remove(f0)

    # Remove the list file
    os.remove('curl_list')

    return 0

def downloadRnxONDayNSta(mjd0,days,fSta,fNetrc,DesPath) -> int:
    '''Download N day RINEX observation files for specified stations (
    via station list file, fSta) to local destination directory DesPath.
    Current directory is the working directory where stores the cookies file,
    curl_cookies and file list of the target remote direcotry, curl_list.
    '''
    mjd=mjd0
    while mjd<=(mjd0+days-1):
        # Convert the date in different formats
        year,doy=GNSSTime.mjd2doy(mjd)
        YYYY=str(year)
        DDD=str(doy).zfill(3)
        # Destination directory for this day
        strTmp=os.path.join(DesPath,YYYY,DDD)
        # Print the day in processing
        print('Downloading for '+YYYY+' '+DDD)
        iReturn=downloadRnxO1DayNSta(year,doy,fSta,fNetrc,strTmp)
        if iReturn!=0:
            print('Failed to download for '+YYYY+' '+DDD)
        # Next day
        mjd=mjd+1

    return 0

if __name__ == '__main__':
    import argparse

    parser=argparse.ArgumentParser()

    parser.add_argument('--mjd',nargs='?',const=58582,default=58582,type=int,
                        help='Begin date in MJD')
    parser.add_argument('--days',nargs='?',const=1,default=1,type=int,
                        help='Number of following days from MJD (included)')
    parser.add_argument('--fStaList',type=str,required=True,
                        help='A text file containing the list of stations '
                        'whose observation files would be downloaded')
    parser.add_argument('--fNetrc',type=str,required=True,
                        help='netrc file for authentication')
    parser.add_argument('--DesPath',type=str,required=True,
                        help='Detination path to store the downloaded files')
    parser.add_argument('--WrkPath',type=str,required=True,
                        help='Work path where stores the cookies file')

    # For debug
    args=parser.parse_args()
    #args=parser.parse_args(['--fStaList','D:/Code/PROJECT/WORK/2019034.StaBDS3',
    #                        '--fNetrc','D:/Code/PROJECT/WORK/netrc',
    #                        '--DesPath','D:/Code/PROJECT/WORK/temp',
    #                        '--WrkPath','D:/Code/PROJECT/WORK'])
    #parser.print_help()

    # Do some checks
    if not os.path.isfile(args.fStaList):
        sys.exit('File '+args.fStaList+' does not exist!')
    elif not os.path.isfile(args.fNetrc):
        sys.exit('File '+args.fNetrc+' does not exist!')
    else:
        if not os.path.isdir(args.DesPath):
            os.makedirs(args.DesPath)
        if not os.path.isdir(args.WrkPath):
            os.makedirs(args.WrkPath)
        # Change to work path
        os.chdir(args.WrkPath)

    iReturn=downloadRnxONDayNSta(args.mjd,args.days,args.fStaList,args.fNetrc,
                                 args.DesPath)
