#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Functions for transformation between 4 different time formats related to
GNSS data processing, i.e. 1) Modified Julian Day (mjd); 2) year, month and
Day Of Month (dom); 3) year, Day Of Year (doy); 4) GPS week, Day of Week (wkd)
'''

__author__ = 'hanbing'


def IsLeapYear(year):
	"""Determine whether the input year is a leap one."""
	if year%4!=0:
		return False
	elif year%100!=0:
		return True
	elif year%400==0:
		return True
	else:
		return False

def UTC2TAI(rMJD):
	'''
	Return the difference between TAI and UTC
	'''

	if rMJD < 41317:
		LeapSec=9
	elif rMJD < 41499:
		LeapSec=10
	elif rMJD < 41683:
		LeapSec=11
	elif rMJD < 42048:
		LeapSec=12
	elif rMJD < 42413:
		LeapSec=13
	elif rMJD < 42778:
		LeapSec=14
	elif rMJD < 43114:
		LeapSec=15
	elif rMJD < 43509:
		LeapSec=16
	elif rMJD < 43874:
		LeapSec=17
	elif rMJD < 44239:
		LeapSec=18
	elif rMJD < 44786:
		LeapSec=19
	elif rMJD < 45151:
		LeapSec=20
	elif rMJD < 45516:
		LeapSec=21
	elif rMJD < 46247:
		LeapSec=22
	elif rMJD < 47161:
		LeapSec=23
	elif rMJD < 47892:
		LeapSec=24
	elif rMJD < 48257:
		LeapSec=25
	elif rMJD < 48804:
		LeapSec=26
	elif rMJD < 49169:
		LeapSec=27
	elif rMJD < 49534:
		LeapSec=28
	elif rMJD < 50083:
		LeapSec=29
	elif rMJD < 50630:
		LeapSec=30
	elif rMJD < 51179:
		LeapSec=31
	elif rMJD < 53736:
		LeapSec=32
	elif rMJD < 54832:
		LeapSec=33
	elif rMJD < 56109:
		LeapSec=34
	elif rMJD < 57204:
		LeapSec=35
	elif rMJD < 57754:
		LeapSec=36
	else:
		LeapSec=37

	return LeapSec


def year2yr(year):
	"""Convert 4-digit year to 2-digit year"""
	if year>=1951 and year<=2050:
		if(year>=2000):
			yr=year-2000
		else:
			yr=year-1900

		return yr
	else:
		print('ERROR: Could not convert for '+str(year))
		exit(1)

def yr2year(yr):
	"""
	Convert 2-digit year to 4-digit year
	"""
	if yr<=99:
		if yr<=50:
			year=yr+2000
		else:
			year=yr+1900

		return year
	else:
		print('ERROR: Could not convert for '+str(yr))
		exit(1)

def mjd2doy(MJD):
	"""Convert Modified Julian Day to year, day of year. Given MJD
	should be later than 34012 (i.e. 1952.01.01)

	"""
	if MJD<34012:
		print('ERROR: Could not convert for '+str(MJD))
		exit(1)
	else:
		YYYY=1952
		DAYS=366
		DOY=MJD-34011
		while DOY>DAYS:
			DOY=DOY-DAYS
			YYYY=YYYY+1
			if IsLeapYear(YYYY):
				DAYS=366
			else:
				DAYS=365

		return YYYY,DOY

def mjd2wkd(MJD):
	"""Convert Modified Julian Day to GPS week and week day. Given MJD
	should be later than 44244 (i.e. 1980.01.06, the start of GPS time)

    """
	# Begin from 1980.01.06
	if MJD<44244:
		print('ERROR: Could not convert for '+str(MJD))
		exit(1)
	else:
		WK=(MJD-44244)//7
		WKD=(MJD-44244)%7
		return WK,WKD

def doy2mjd(YYYY,DOY):
	"""Convert year, day of year to Modified Julian Day. Given (year,
	day of year) should be later than (1952, 001).

	"""
	# Begin from 1952.01.01
	if YYYY<1952 or DOY<1:
		print('ERROR: Could not convert for '+str(YYYY)+' '+str(DOY))
		exit(1)
	else:
		YYYY0=1952
		DOY0=1
		MJD=34012
		while YYYY0<YYYY or DOY0<DOY:
			MJD=MJD+1
			DOY0=DOY0+1
			if IsLeapYear(YYYY0):
				if DOY0>366:
					DOY0=1
					YYYY0=YYYY0+1
			else:
				if DOY0>365:
					DOY0=1
					YYYY0=YYYY0+1

		return MJD

def doy2wkd(YYYY,DOY):
	"""Convert year, day of year to GPS week, day of week.

	"""
	MJD=doy2mjd(YYYY,DOY)
	WK,WKD=mjd2wkd(MJD)
	return WK,WKD

def doy2dom(YYYY,DOY):
	"""Convert year, day of year to month, day of month.

	"""
	DaysInMonth=[31,28,31,30,31,30,31,31,30,31,30,31]
	if IsLeapYear(YYYY):
		DaysInMonth[1]=29

	DD=DOY
	MO=1
	while DD>0:
		DD=DD-DaysInMonth[MO-1]
		MO=MO+1

	MO=MO-1
	DD=DD+DaysInMonth[MO-1]

	return MO,DD

def mjd2dom(MJD):
	"""Convert Modified Julian Day to year, month and day of month.

	"""
	YYYY,DOY=mjd2doy(MJD)
	MO,DD=doy2dom(YYYY,DOY)
	return YYYY,MO,DD

def dom2mjd(YYYY,MO,DD):
	"""Convert year, month and day of month to Modified Julian Day.

	"""
	DOYInMonth=[0,31,59,90,120,151,181,212,243,273,304,334]
	if MO<=2:
		iYYYY=YYYY-1
	else:
		iYYYY=YYYY
	MJD=365*YYYY-678941+iYYYY//4-iYYYY//100+iYYYY//400+DD
	MJD=MJD+DOYInMonth[MO-1]

	return MJD

def dom2wkd(YYYY,MO,DD):
	"""Convert year, month and day of month to GPS week, day of week.

	"""
	MJD=dom2mjd(YYYY,MO,DD)
	WK,WKD=mjd2wkd(MJD)
	return WK,WKD

def dom2doy(YYYY,MO,DD):
	"""Convert year, month and day of month to year, day of year.

	"""
	MJD=dom2mjd(YYYY,MO,DD)
	YYYY,DOY=mjd2doy(MJD)
	return YYYY,DOY

def wkd2mjd(WK,WKD):
	"""Convert GPS week and day of week to Modified Julian Day.

	"""
	if WK<0 or WKD<0:
		print('ERROR: Could not convert for '+str(WK)+' '+str(WKD))
		exit(1)
	else:
		MJD=WK*7+WKD+44244
		return MJD

def wkd2doy(WK,WKD):
	"""Convert GPS week and day of week to year, day of year.

	"""
	MJD=wkd2mjd(WK,WKD)
	YYYY,DOY=mjd2doy(MJD)
	return YYYY,DOY

def wkd2dom(WK,WKD):
	"""Convert GPS week and day of week to year, month and day of month.

	"""
	MJD=wkd2mjd(WK,WKD)
	YYYY,MO,DD=mjd2dom(MJD)
	return YYYY,MO,DD

def snx2mjd(cSNX):
	'''
	yyddd:sssss
	yyyyddd:sssss
	yy:ddd:sssss
	yyyy:ddd:sssss
	'''

	cWord=cSNX.split(sep=':')
	nWord=len(cWord)
	if nWord == 2:
		if len(cWord[0]) == 5 and len(cWord[1]) == 5:
			#yyddd:sssss
			YYYY=yr2year(int(cWord[0][0:2]))
			DOY=int(cWord[0][2:5])
			iSOD=int(cWord[1][0:5])
		elif len(cWord[0]) == 7 and len(cWord[1]) == 5:
			#yyyyddd:sssss
			YYYY=int(cWord[0][0:4])
			DOY=int(cWord[0][4:7])
			iSOD=int(cWord[1][0:5])
		else:
			sys.exit('Unknown SINEX time format '+cSNX)
	elif nWord == 3:
		if len(cWord[0]) == 2 and len(cWord[1]) == 3 and len(cWord[2]) == 5:
			#yy:ddd:sssss
			YYYY=yr2year(int(cWord[0]))
			DOY=int(cWord[1])
			iSOD=int(cWord[2])
		elif len(cWord[0]) == 4 and len(cWord[1]) == 3 and len(cWord[2]) == 5:
			#yyyy:ddd:sssss
			YYYY=int(cWord[0])
			DOY=int(cWord[1])
			iSOD=int(cWord[2])
		else:
			sys.exit('Unknown SINEX time format '+cSNX)
	else:
		sys.exit('Unknown SINEX time format '+cSNX)

	iMJD=doy2mjd(YYYY,DOY)
	rMJD=iMJD+iSOD/86400.0
	return rMJD
