#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Do GNSS data processing using PANDA software
'''
__author__ = 'hanbing'

# Standard library imports
import os
import os.path
import sys
import re
import shutil
import fnmatch
import subprocess

# Local application/library specific imports
from PySRC.miscellaneous import GNSSTime

def creatProject(SysPath,ProPath,Year,DOY,CF):
    '''Creat the project directory and copy system table files there
    Return the workpath
    '''
    YYYY=str(Year)
    DDD=str(DOY).zfill(3)

    if not os.path.isdir(SysPath):
        print('ERROR: PANDA system table file directory does not exist!')
        return 1
    if not os.path.isdir(ProPath):
        os.makedirs(ProPath)
    WrkPath=os.path.join(ProPath,'WORK'+YYYY+DDD)
    if os.path.isdir(WrkPath):
        shutil.rmtree(WrkPath)
    os.makedirs(WrkPath)

    cf0=os.path.join(SysPath,CF)
    if not os.path.isfile(cf0):
        print('ERROR: File '+cf0+' does not exist!')
        return 1
    else:
        shutil.copyfile(cf0,os.path.join(WrkPath,'cf_0'))

    strTmp=os.path.join(SysPath,'EGM')
    if not os.path.isfile(strTmp):
        print('ERROR: File '+strTmp+' does not exist!')
        return 1
    else:
        shutil.copy(strTmp,WrkPath)
    strTmp=os.path.join(SysPath,'EGM2008_to2190_TideFree')
    if not os.path.isfile(strTmp):
        print('ERROR: File '+strTmp+' does not exist!')
        return 1
    else:
        shutil.copy(strTmp,WrkPath)
    strTmp=os.path.join(SysPath,'file_table')
    if not os.path.isfile(strTmp):
        print('ERROR: File '+strTmp+' does not exist!')
        return 1
    else:
        shutil.copy(strTmp,WrkPath)
    # Copy igs14_????.atx and igs_metadata_*.snx
    for f in os.listdir(SysPath):
        if fnmatch.fnmatch(f,'igs14_????.atx'):
            strTmp=os.path.join(SysPath,f)
            shutil.copy(strTmp,WrkPath)
        elif fnmatch.fnmatch(f,'igs_metadata_*.snx'):
            strTmp=os.path.join(SysPath,f)
            shutil.copy(strTmp,WrkPath)
    #jpleph_de405, special for Windows
    #strTmp=os.path.join(SysPath,'jpleph_de405')
    strTmp='D:/Code/PROJECT/WORK/jpleph_de405'
    if not os.path.isfile(strTmp):
        print('ERROR: File '+strTmp+' does not exist!')
        return 1
    else:
        shutil.copy(strTmp,WrkPath)

    strTmp=os.path.join(SysPath,'leap_seconds')
    if not os.path.isfile(strTmp):
        print('ERROR: File '+strTmp+' does not exist!')
        return 1
    else:
        shutil.copy(strTmp,WrkPath)
    strTmp=os.path.join(SysPath,'ocean_tide')
    if not os.path.isfile(strTmp):
        print('ERROR: File '+strTmp+' does not exist!')
        return 1
    else:
        shutil.copy(strTmp,WrkPath)
    strTmp=os.path.join(SysPath,'fes2004_Cnm-Snm.dat')
    if not os.path.isfile(strTmp):
        print('ERROR: File '+strTmp+' does not exist!')
        return 1
    else:
        shutil.copy(strTmp,WrkPath)
    strTmp=os.path.join(SysPath,'oceanload')
    if not os.path.isfile(strTmp):
        print('ERROR: File '+strTmp+' does not exist!')
        return 1
    else:
        shutil.copy(strTmp,WrkPath)
    strTmp=os.path.join(SysPath,'poleut1_'+YYYY)
    if not os.path.isfile(strTmp):
        print('ERROR: File '+strTmp+' does not exist!')
        return 1
    else:
        shutil.copy(strTmp,os.path.join(WrkPath,'poleut1'))
    strTmp=os.path.join(SysPath,'RECEIVER')
    if not os.path.isfile(strTmp):
        print('ERROR: File '+strTmp+' does not exist!')
        return 1
    else:
        shutil.copy(strTmp,WrkPath)
    strTmp=os.path.join(SysPath,'sat_parameters_new')
    if not os.path.isfile(strTmp):
        print('ERROR: File '+strTmp+' does not exist!')
        return 1
    else:
        shutil.copy(strTmp,WrkPath)
    strTmp=os.path.join(SysPath,'SatellitePannels')
    if not os.path.isfile(strTmp):
        print('ERROR: File '+strTmp+' does not exist!')
        return 1
    else:
        shutil.copy(strTmp,WrkPath)

    ObsPath=os.path.join(ProPath,'OBS',YYYY,DDD)
    if os.path.isdir(ObsPath):
        shutil.rmtree(ObsPath)
    os.makedirs(ObsPath)

    return WrkPath

def pppOneSession(SysPath,ProPath,RstPath,year,doy,dIntv,fSta,cf0,tSys,cSys,
                  SysObs,lFix,OrbPath,AC,fObsSelection):
    '''Do PPP processing for one session, i.e. one day for this moment.
    '''

    # Creat the project
    WrkPath=creatProject(SysPath,ProPath,year,doy,cf0)
    if WrkPath==1:
        print('ERROR: Failed to creat work path!')
        return 1
    else:
        os.chdir(WrkPath)

    YYYY=str(year)
    YR=str(GNSSTime.year2yr(year)).zfill(2)
    DDD=str(doy).zfill(3)
    Week,WeekDay=GNSSTime.doy2wkd(year,doy)
    WK=str(Week).zfill(4)
    WKD=str(WeekDay)
    Month,Day=GNSSTime.doy2dom(year,doy)
    MO=str(Month).zfill(2)
    DD=str(Day).zfill(2)
    HH='00'
    MM='00'
    SS='00'
    SesLength=86400

    # Download  RINEX observation files
    strTmp=os.path.join('D:/Code/PROJECT/WORK/temp',YYYY,DDD)
    if not os.path.isfile(fSta):
        print('ERROR: File '+fSta+' does not exist!')
        return 1
    with open(fSta,'rt') as fOb:
        for cLine in fOb:
            if cLine[0]!=' ' or len(cLine)<5:
                continue
            s=cLine[1:5].lower()
            fRnxO=os.path.join(strTmp,s+DDD+'0.'+YR+'o')
            if os.path.isfile(fRnxO):
                shutil.copy(fRnxO,os.path.join('..','OBS',YYYY,DDD))
                fRnxO=os.path.join('..','OBS',YYYY,DDD,s+DDD+'0.'+YR+'o')
                cmd=['SelectRnxObs','-fInpRnxO',fRnxO,'-fSet',fObsSelection]
                cPro=subprocess.run(cmd)
                if(cPro.returncode!=0):
                    print('ERROR: Failed to SelectRnxObs for '+fRnxO)
                    return 1
                else:
                    shutil.move(fRnxO,fRnxO+'_old')
                    shutil.move(fRnxO+'_new',fRnxO)
                    os.remove(fRnxO+'_old')
            else:
                print('Failed to download '+fRnxO)
    fOb.close()

    # Download RINEX navigation file(s)
    strTmp='E:/PPP/NAV'
    fRnxN='brdm'+DDD+'0.'+YR+'p'
    if os.path.isfile(os.path.join(strTmp,'brdm'+DDD+'0.'+YR+'p')):
        shutil.copy(os.path.join(strTmp,'brdm'+DDD+'0.'+YR+'p'),WrkPath)
    else:
        print('ERROR: Failed to download '+fRnxN)
        return 1

    # Download ORB&&CLK product files
    if not os.path.isdir(OrbPath):
        print('ERROR: Path '+OrbPath+' does not exist!')
        return 1
    if AC=='phb':
        fClk='clk_'+str(YYYY)+DDD
        fSP3='phb'+WK+WKD+'.sp3'
    else:
        fClk=AC+WK+WKD+'.clk'
        fSP3=AC+WK+WKD+'.sp3'
    if os.path.isfile(os.path.join(OrbPath,fClk)):
        shutil.copy(os.path.join(OrbPath,fClk),
                    os.path.join(WrkPath,'clk_'+str(YYYY)+DDD))
    else:
        print('ERROR: Failed to download '+fClk)
        return 1
    if os.path.isfile(os.path.join(OrbPath,fSP3)):
        shutil.copy(os.path.join(OrbPath,fSP3),WrkPath)
    else:
        print('ERROR: Failed to download '+fSP3)
        return 1

    # Download DCB files

    # Merge navigation and observation files

    # Do some basic modification to current ctrl-file (cf_0) to 
    # generate new ctrl-file (cf_1)
    with open('cf_0','r') as fOb1,open('cf_1','w') as fOb2:
        for cLine1 in fOb1:
            if 'Start time&session length =' == cLine1[0:27]:
                cLine2='Start time&session length = '+YYYY+' '+MO+' '+DD+\
                ' '+HH+' '+MM+' '+SS+' '+str(SesLength)+' '+'['+tSys+']\n'
            elif 'Observation interval      =' in cLine1:
                cLine2='Observation interval      = '+str(dIntv)+'\n'
            elif 'Estimate satellite orbit  =' in cLine1:
                cLine2='Estimate satellite orbit  = NO\n'
            elif 'Receiver ISB/IFB          =' in cLine1:
                #cLine2='Receiver ISB/IFB          = NONE\n'
                cLine2='Receiver ISB/IFB          = AUTO+CON\n'
            elif 'Output OMC                =' in cLine1:
                cLine2='Output OMC                = YES\n'
            elif 'Estimate ERP              =' in cLine1[0:27]:
                cLine2='Estimate ERP              = NONE\n'
            else:
                cLine2=cLine1
            fOb2.write(cLine2)

    # Do some checks
    CF='cf_1'
    # Check the existence of observation directory
    strTmp=os.path.join('..','OBS',str(YYYY),DDD)
    if(not os.path.isdir(strTmp)):
        print('ERROR: Directory '+strTmp+' does not exist!')
        return 1
    # Check the existence of ctrl-file
    if(not os.path.isfile(CF)):
        print('ERROR: '+CF+' does not exist!')
        return 1

    # Check the existence of navigation file
    if(not os.path.isfile(fRnxN)):
        print('ERROR: File '+fRnxN+' does not exist!')
        return 1
    if(not os.path.isdir('logs')):
        os.mkdir('logs')
    # Check the existence of orbit file
    if(not os.path.isfile(fSP3)):
        print('ERROR: File '+fSP3+' does not exist!')
        return 1
    else:
        fLog=os.path.join('logs','log_sp3orb')
        try:
            fOb=open(fLog,mode='wt')
        except OSError:
            print('ERROR: Failed to open '+fLog)
            return 1
        cmd=['sp3orb','-sp3',fSP3,'-extend','n','-orb','orb','-ctrl',CF]
        cPro=subprocess.run(cmd,text=True,stdout=fOb,stderr=subprocess.STDOUT)
        fOb.close()
        if cPro.returncode != 0:
            print('ERROR: Filed to convert '+fSP3+' into orb!')
            return 1
        else:
            # Clean those ics_ files
            for f in fnmatch.filter(os.listdir(),'ics_???????_???'):
                os.remove(f)

    # Do some complex modification to the ctrl file (using the FORTRAN
    # program, named the new ctrl file as cf_2)
    fLog=os.path.join('logs','log_UpdateCtrlFile_0')
    try:
        fOb=open(fLog,mode='wt')
    except OSError:
        print('ERROR: Failed to open '+fLog)
        return 1
    cmd=['UpdateCtrlFile','-fCtrl',CF,'-FixSatClk','y','-fStaList',fSta,
         '-SourceOfUpdateCRD','RXOH+BSPP','-SourceOfUpdateANT','RXOH',
         '-SourceOfUpdateREC','RXOH','-fRnxN',fRnxN,'-cSys',cSys,
         '-SysObs',cSys,SysObs]
    cPro=subprocess.run(cmd,text=True,stdout=fOb,stderr=subprocess.STDOUT)
    fOb.close()
    if(cPro.returncode==0):
        shutil.move(CF+'_new','cf_2')
    else:
        print('ERROR: Failed to execute UpdateCtrlFile')
        return 1

    # Do the data processing
    CF='cf_2'
    # Data pre-processing (Parallelization should be applied)
    strTmp=os.path.join('..','OBS',str(YYYY),DDD)
    shutil.copyfile(CF,'cf_lsq')
    with open(CF,'r') as fOb:
        nLine=0
        i1=0
        i2=0
        for cLine in fOb:
            nLine=nLine+1
            if re.match(r'\+Station used',cLine):
                i1=nLine
            elif re.match(r'\-Station used',cLine):
                i2=nLine
            if i1==0:
                continue
            if i2!=0:
                break
            if cLine[0]!=' ' or len(cLine)<5:
                continue
            s=cLine[1:5].lower()
            rnxOF=os.path.join(strTmp,s+DDD+'0.'+YR+'o')
            if not os.path.isfile(rnxOF):
                print(rnxOF+' does not exist!')
                continue
            fLog=rnxOF+'_log'
            try:
                fLogOb=open(fLog,mode='wt')
            except OSError:
                print('ERROR: Failed to open '+fLog)
                return 1
            cmd=['turboedit','-rnxn',fRnxN,'-StaName',s,'-fCtrl',
                 'cf_lsq','-PDifLimit','3000']
            cPro=subprocess.run(cmd,text=True,stdout=fLogOb,
                                stderr=subprocess.STDOUT)
            fLogOb.close()
            if(cPro.returncode!=0):
                print('ERROR: Failed to turboedit for '+s)
                return 1
    os.remove('cf_2')

    # Iterative least squares
    iIter=0
    CF='cf_lsq1'
    os.rename('cf_lsq',CF)
    while iIter<=10:
        iIter=iIter+1
        # lsq
        fLog=os.path.join('logs','log_lsq_'+str(iIter))
        try:
            fOb=open(fLog,mode='wt')
        except OSError:
            print('ERROR: Failed to open '+fLog)
            return 1
        cmd=['lsq','-cf',CF,'-ambcon','no','-PDifLimit','3000']
        cPro=subprocess.run(cmd,text=True,stdout=fOb,stderr=subprocess.STDOUT)
        fOb.close()
        if(cPro.returncode!=0):
            print('ERROR: Failed to lsq at iteration '+str(iIter))
            return 1
        # extclk for AR
        fLog=os.path.join('logs','log_extclk_AR_'+str(iIter))
        try:
            fOb=open(fLog,mode='wt')
        except OSError:
            print('ERROR: Failed to open '+fLog)
            return 1
        cmd=['extclk','-fPar','par_'+str(YYYY)+DDD,'-fClk','rec_'+str(YYYY)+DDD,
             '-ClkType','AR']
        cPro=subprocess.run(cmd,text=True,stdout=fOb,stderr=subprocess.STDOUT)
        fOb.close()
        if(cPro.returncode!=0):
            print('ERROR: Failed to extclk for AR at iteration '+str(iIter))
            return 1
        # edtres
        fLog=os.path.join('logs','log_edtres_'+str(iIter))
        try:
            fOb=open(fLog,mode='wt')
        except OSError:
            print('ERROR: Failed to open '+fLog)
            return 1
        cmd=['edtres','-ResFile','res_'+str(YYYY)+DDD,'-SumFile',
             'sum_'+str(YYYY)+DDD+'_float','-Edit','P','-fDelPoint',
             'node_'+str(YYYY)+DDD+'.del_'+str(iIter),'-ExcludeSat1','n']
        cPro=subprocess.run(cmd,text=True,stdout=fOb,stderr=subprocess.STDOUT)
        fOb.close()
        iEdtres=cPro.returncode
        if(iEdtres!=0 and iEdtres!=2):
            print('ERROR: Failed to edtres at iteration '+str(iIter)+
                     ': '+str(iEdtres))
            return 1
        else:
            print(str(iIter)+' '+str(iEdtres))
        # UpdateCtrlFile
        if os.path.getsize('node_'+str(YYYY)+DDD+'.del_'+str(iIter))>0:
            fLog=os.path.join('logs','log_UpdateCtrlFile_'+str(iIter))
            try:
                fOb=open(fLog,mode='wt')
            except OSError:
                print('ERROR: Failed to open '+fLog)
                return 1
            cmd=['UpdateCtrlFile','-fCtrl',CF,'-fSatDel',
                 'node_'+str(YYYY)+DDD+'.del_'+str(iIter),'-UpdateWhateverANT',
                 'n','-UpdateWhateverREC','n']
            cPro=subprocess.run(cmd,text=True,stdout=fOb,stderr=subprocess.STDOUT)
            fOb.close()
            if(cPro.returncode==0):
                shutil.move(CF+'_new','cf_lsq'+str(iIter+1))
            else:
                print('ERROR: Failed to UpdateCtrlFile at iteration '+str(iIter))
                return 1
        else:
            shutil.copyfile(CF,'cf_lsq'+str(iIter+1))
        # ctrl-file for the next use
        CF='cf_lsq'+str(iIter+1)
        if iEdtres==0:
            break

    
    cmd=['extpos','-fPar','par_'+str(YYYY)+DDD,'-fOut',
         'pos_'+str(YYYY)+DDD+'_float']
    cPro=subprocess.run(cmd)
    if(cPro.returncode!=0):
        print('ERROR: Failed to extpos from '+'par_'+str(YYYY)+DDD)
        return 1
    os.rename('par_'+str(YYYY)+DDD,'par_'+str(YYYY)+DDD+'_float')
    os.rename('res_'+str(YYYY)+DDD,'res_'+str(YYYY)+DDD+'_float')
    # Ambguity fixing
    if lFix:
        # ambfix 1
        fLog=os.path.join('logs','log_ambfix_1')
        try:
            fOb=open(fLog,mode='wt')
        except OSError:
            print('ERROR: Failed to open '+fLog)
            return 1
        cmd=['ambfix','-fCtrl',CF,'-PDifLimit','3000']
        cPro=subprocess.run(cmd,text=True,stdout=fOb,stderr=subprocess.STDOUT)
        fOb.close()
        if(cPro.returncode!=0):
            print('ERROR: Failed to ambfix at first time!')
            return 1
        # lsq
        fLog=os.path.join('logs','log_lsq_ambfix_1')
        try:
            fOb=open(fLog,mode='wt')
        except OSError:
            print('ERROR: Failed to open '+fLog)
            return 1
        cmd=['lsq','-cf',CF,'-ambcon','yes','-PDifLimit','3000']
        cPro=subprocess.run(cmd,text=True,stdout=fOb,stderr=subprocess.STDOUT)
        fOb.close()
        if(cPro.returncode!=0):
            print('ERROR: Failed to lsq after first ambfix')
            return 1
        else:
            os.rename('res_'+str(YYYY)+DDD,'res_'+str(YYYY)+DDD+'_fix_1')
        # extclk for AR
        fLog=os.path.join('logs','log_extclk_AR_ambfix_1')
        try:
            fOb=open(fLog,mode='wt')
        except OSError:
            print('ERROR: Failed to open '+fLog)
            return 1
        cmd=['extclk','-fPar','par_'+str(YYYY)+DDD,'-fClk','rec_'+str(YYYY)+DDD,
             '-ClkType','AR']
        cPro=subprocess.run(cmd,text=True,stdout=fOb,stderr=subprocess.STDOUT)
        fOb.close()
        if(cPro.returncode!=0):
            print('ERROR: Failed to extclk for AR after first ambfix')
            return 1
        # Just for generating summary file
        fLog=os.path.join('logs','log_edtres_fix_1')
        try:
            fOb=open(fLog,mode='wt')
        except OSError:
            print('ERROR: Failed to open '+fLog)
            return 1
        cmd=['edtres','-ResFile','res_'+str(YYYY)+DDD+'_fix_1','-SumFile',
             'sum_'+str(YYYY)+DDD+'_fix_1','-Edit','no','-fDelPoint',
             'node_'+str(YYYY)+DDD+'.del_fix','-ExcludeSat1','n']
        cPro=subprocess.run(cmd,text=True,stdout=fOb,stderr=subprocess.STDOUT)
        fOb.close()
        if(cPro.returncode!=0):
            print('ERROR: Failed to edtres for fix 1')
            return 1

        cmd=['extpos','-fPar','par_'+str(YYYY)+DDD,'-fOut',
             'pos_'+str(YYYY)+DDD+'_fix_1']
        cPro=subprocess.run(cmd)
        if(cPro.returncode!=0):
            print('ERROR: Failed to extpos from '+'par_'+str(YYYY)+DDD)
            return 1
        os.rename('par_'+str(YYYY)+DDD,'par_'+str(YYYY)+DDD+'_fix_1')

        # ambfix 2
        fLog=os.path.join('logs','log_ambfix_2')
        try:
            fOb=open(fLog,mode='wt')
        except OSError:
            print('ERROR: Failed to open '+fLog)
            return 1
        cmd=['ambfix','-fCtrl',CF,'-PDifLimit','3000']
        cPro=subprocess.run(cmd,text=True,stdout=fOb,stderr=subprocess.STDOUT)
        fOb.close()
        if(cPro.returncode!=0):
            print('ERROR: Failed to ambfix at second time!')
            return 1
        # lsq
        fLog=os.path.join('logs','log_lsq_ambfix_2')
        try:
            fOb=open(fLog,mode='wt')
        except OSError:
            print('ERROR: Failed to open '+fLog)
            return 1
        cmd=['lsq','-cf',CF,'-ambcon','yes','-PDifLimit','3000']
        cPro=subprocess.run(cmd,text=True,stdout=fOb,stderr=subprocess.STDOUT)
        fOb.close()
        if(cPro.returncode!=0):
            print('ERROR: Failed to lsq after second ambfix')
            return 1
        else:
            os.rename('res_'+str(YYYY)+DDD,'res_'+str(YYYY)+DDD+'_fix_2')
        # extclk for AR
        fLog=os.path.join('logs','log_extclk_AR_ambfix_2')
        try:
            fOb=open(fLog,mode='wt')
        except OSError:
            print('ERROR: Failed to open '+fLog)
            return 1
        cmd=['extclk','-fPar','par_'+str(YYYY)+DDD,'-fClk','rec_'+str(YYYY)+DDD,
             '-ClkType','AR']
        cPro=subprocess.run(cmd,text=True,stdout=fOb,stderr=subprocess.STDOUT)
        fOb.close()
        if(cPro.returncode!=0):
            print('ERROR: Failed to extclk for AR after first ambfix')
            return 1
        # Just for generating summary file
        fLog=os.path.join('logs','log_edtres_fix_2')
        try:
            fOb=open(fLog,mode='wt')
        except OSError:
            print('ERROR: Failed to open '+fLog)
            return 1
        cmd=['edtres','-ResFile','res_'+str(YYYY)+DDD+'_fix_2','-SumFile',
             'sum_'+str(YYYY)+DDD+'_fix_2','-Edit','no','-fDelPoint',
             'node_'+str(YYYY)+DDD+'.del_fix','-ExcludeSat1','n']
        cPro=subprocess.run(cmd,text=True,stdout=fOb,stderr=subprocess.STDOUT)
        fOb.close()
        if(cPro.returncode!=0):
            print('ERROR: Failed to edtres for fix 2')
            return 1

        cmd=['extpos','-fPar','par_'+str(YYYY)+DDD,'-fOut',
             'pos_'+str(YYYY)+DDD+'_fix_2']
        cPro=subprocess.run(cmd)
        if(cPro.returncode!=0):
            print('ERROR: Failed to extpos from '+'par_'+str(YYYY)+DDD)
            return 1
        os.rename('par_'+str(YYYY)+DDD,'par_'+str(YYYY)+DDD+'_fix_2')

    print('Work end!')

    # Archive the results
    strTmp=os.path.join(RstPath,str(YYYY))
    if(not os.path.isdir(strTmp)):
        os.makedirs(strTmp)

    strTmp=os.path.join(RstPath,str(YYYY),'PAR_PPP')
    if(not os.path.isdir(strTmp)):
        os.makedirs(strTmp)
    for f in os.listdir('.'):
        if fnmatch.fnmatch(f,'par_'+str(YYYY)+DDD+'_*'):
            shutil.copy(f,strTmp)

    strTmp=os.path.join(RstPath,str(YYYY),'POS_PPP')
    if(not os.path.isdir(strTmp)):
        os.makedirs(strTmp)
    for f in os.listdir('.'):
        if fnmatch.fnmatch(f,'pos_'+str(YYYY)+DDD+'_*'):
            shutil.copy(f,strTmp)

    strTmp=os.path.join(RstPath,str(YYYY),'RECCLK_PPP')
    if(not os.path.isdir(strTmp)):
        os.makedirs(strTmp)
    shutil.copy('rec_'+str(YYYY)+DDD,strTmp)

    strTmp=os.path.join(RstPath,str(YYYY),'RES_PPP')
    if(not os.path.isdir(strTmp)):
        os.makedirs(strTmp)
    for f in os.listdir('.'):
        if fnmatch.fnmatch(f,'res_'+str(YYYY)+DDD+'_*'):
            shutil.copy(f,strTmp)
    
    strTmp=os.path.join(RstPath,str(YYYY),'SUM_PPP')
    if(not os.path.isdir(strTmp)):
        os.makedirs(strTmp)
    for f in os.listdir('.'):
        if fnmatch.fnmatch(f,'sum_'+str(YYYY)+DDD+'_*'):
            shutil.copy(f,strTmp)

    return 0

if __name__ == '__main__':
    import argparse

    parser=argparse.ArgumentParser()

    parser.add_argument('--ProPath',type=str,required=True,
                        help='Directory of the project')
    parser.add_argument('--RstPath',type=str,required=True,
                        help='Directory of the result')
    parser.add_argument('--OrbPath',type=str,required=True,
                        help='Directory of ORB && CLK')
    parser.add_argument('--AC',type=str,required=True,choices=['phb','wum'],
                        help='Prefix of ORB && CLK to be used')
    parser.add_argument('--year',type=int,required=True,help='4-digit year')
    parser.add_argument('--doy',type=int,required=True,help='Day of Year')
    parser.add_argument('--days',type=int,required=True,help='Number of days')
    parser.add_argument('--dIntv',type=int,required=True,
                        help='Sampling  interval')
    parser.add_argument('--tSys',nargs='?',const='GPS',default='GPS',type=str,
                        choices=['GPS','BDS'],help='time system used')
    parser.add_argument('--cf',nargs='?',const='cf_ppp',default='cf_ppp',
                        type=str,help='ctrl file')
    parser.add_argument('--cSys',type=str,required=True,
                        help='GNSS system ID(s)')
    parser.add_argument('--SysObs',type=str,required=True,
                        help='Observation types used')
    parser.add_argument('--Fix',action='store_true',
                        help='Fix ambiguities')
    parser.add_argument('--fSta',type=str,required=True,
                        help='List of stations used')
    parser.add_argument('--fObsSelection',type=str,required=True,
                        help='File for observation selection')

    # For debug
    args=parser.parse_args(['--ProPath','E:/PPP/S6_GC_B1I+B2I+B1I+B3I_Bias',
                            '--RstPath','E:/PPP/S6_GC_B1I+B2I+B1I+B3I_Bias/RESULT',
                            '--OrbPath','E:/PPP/PRODUCT/S6_GC','--AC','phb',
                            '--year','2019','--doy','90','--days','11',
                            '--dIntv','300','--cf','cf_PPP_C',
                            '--cSys','C','--SysObs','LC12+PC12+LC13+PC13','--Fix',
                            '--fSta','E:/PPP/StaList',
                            '--fObsSelection','E:/PPP/ObsSelection_B1IB2IB3I'])
    #parser.print_help()

    # Path of system table files
    SysPath=r"E:/FromGFZServer/sys_data"

    mjd0=GNSSTime.doy2mjd(args.year,args.doy)
    mjd=mjd0
    while mjd<=(mjd0+args.days-1):
        year,doy=GNSSTime.mjd2doy(mjd)
        YYYY=str(year)
        DDD=str(doy).zfill(3)
        print('Start PPP for '+YYYY+' '+DDD)
        iReturn=pppOneSession(SysPath,args.ProPath,args.RstPath,year,doy,
                              args.dIntv,args.fSta,args.cf,args.tSys,args.cSys,
                              args.SysObs,args.Fix,args.OrbPath,args.AC,
                              args.fObsSelection)
        if iReturn!=0:
            print('Failed to PPP for '+YYYY+' '+DDD+': '+str(iReturn))
        else:
            print('Finish PPP for '+YYYY+' '+DDD)
        # Next day
        mjd=mjd+1