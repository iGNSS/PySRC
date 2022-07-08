#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Functions for SISRE computation
'''

__author__ = 'hanbing'

# Standard library imports
import os
import sys
import os.path
import glob
import math

# Related third party imports
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import griddata

from astropy.timeseries import LombScargle

mpl.rcParams['mathtext.fontset'] = 'stix'


def Lat2Ele(Rs, Re, Lat, lForward):
    '''
    Calculate the elevation angle of the LOS to a satellite given the
    latitude of a site or vice versa.

          Rs --- the geocenter distance of the satellite
          Re --- the geocenter distance of the site
         Lat --- the latitude of the site or elevation angle of the
                 LOS, in deg
    lForward --- whether calculate the elevation angle given the
                 latitude, otherwise vice versa.
    '''

    if lForward:
        # From Lat to Ele
        Ele = (Rs*np.sin(np.deg2rad(Lat))-Re)/(Rs*np.cos(np.deg2rad(Lat)))
        Ele = np.rad2deg(np.arctan(Ele))
    else:
        # From Ele to Lat
        y = np.tan(np.deg2rad(Lat))
        Ele = (2*Re+np.sqrt(4*Re*Re-4*(y*y+1)*(Re*Re-Rs*Rs*y*y))) / \
              (2*Rs*(y*y+1))
        Ele = np.rad2deg(np.arcsin(Ele))
    return Ele


def CalWeightWR1(Rs, Re, Lat, Lat0):
    '''
    The indefinite integral to calculate the wR weight for SISRE computation.

          Rs --- the geocenter distance of the satellite
          Re --- the geocenter distance of the site
         Lat --- "Latitude" above the AC/TN plane, in degree
        Lat0 --- the minimum "latitude" above the AC/TN plane, in degree
    '''

    y = Rs/Re
    theta = np.deg2rad(Lat)
    theta0 = np.deg2rad(Lat0)

    wR = np.sqrt(1 + y**2 - 2*y*np.sin(theta))*(-2*y**2 + y *
                                                np.sin(theta) + 1)/(3*y**2*(1 - np.sin(theta0)))
    return wR


def CalWeightWR2(Rs, Re, Lat, Lat0):
    '''
    The indefinite integral to calculate the wR2*wR2 weight for SISRE computation.

          Rs --- the geocenter distance of the satellite
          Re --- the geocenter distance of the site
         Lat --- "Latitude" above the AC/TN plane, in degree
        Lat0 --- the minimum "latitude" above the AC/TN plane, in degree
    '''

    y = Rs/Re
    theta = np.deg2rad(Lat)
    theta0 = np.deg2rad(Lat0)
    # the indefinite integral of wR2*wR2
    wR2 = (2*y*np.sin(theta)*(3*y**2-y*np.sin(theta)-1) - (y**2-1)**2 *
           np.log(y**2-2*y*np.sin(theta)+1))/(8*y**3*(1-np.sin(theta0)))
    return wR2


def CalWeightWA2(Rs, Re, Lat, Lat0):
    '''
    The indefinite integral to calculate the wA2*wA2 (or wC2*wC2) weight for SISRE computation.

          Rs --- the geocenter distance of the satellite
          Re --- the geocenter distance of the site
         Lat --- "Latitude" above the AC/TN plane, in degree
        Lat0 --- the minimum "latitude" above the AC/TN plane, in degree
    '''

    y = Rs/Re
    theta = np.deg2rad(Lat)
    theta0 = np.deg2rad(Lat0)
    # the indefinite integral of wA2*wA2 (or wC2*wC2)
    wA2 = (2*y*np.sin(theta)*(y**2+y*np.sin(theta)+1) + (y**2-1)**2 *
           np.log(y**2-2*y*np.sin(theta)+1))/(16*y**3*(1-np.sin(theta0)))
    return wA2


def CalWeight(Rs, Re, Lat0):
    '''
    Calculate the RTN/RAC weights for SISRE computation.

          Rs --- the geocenter distance of the satellite
          Re --- the geocenter distance of the site
        Lat0 --- The minimum "latitude" above the AC/TN plane, in degree
    '''

    wR1 = CalWeightWR1(Rs, Re, 90.0, Lat0)-CalWeightWR1(Rs, Re, Lat0, Lat0)
    wR2 = CalWeightWR2(Rs, Re, 90.0, Lat0)-CalWeightWR2(Rs, Re, Lat0, Lat0)
    wA2 = CalWeightWA2(Rs, Re, 90.0, Lat0)-CalWeightWA2(Rs, Re, Lat0, Lat0)

    return wR1, np.sqrt(wR2), np.sqrt(wA2)


def PlotWeight(Rs, Re, Ele0, OutFilePrefix, OutFileSuffix):
    '''
    Plot the weight functions for a specified minimum "latitude"

      Rs --- list of geocenter distance of the satellite
      Re --- list of geocenter distance of the site
    Ele0 --- the minimum elevation
    '''

    y = []
    wR1 = []
    wR2 = []
    wA2 = []
    for i in range(len(Rs)):
        # The minimum "latitude" corresponds to the minimum elevation
        Lat0 = Lat2Ele(Rs[i], Re[i], Ele0, False)
        # The ratio of Rs to Re
        y.append(Rs[i]/Re[i])
        wR1.append(CalWeightWR1(Rs[i], Re[i], 90.0, Lat0) -
                   CalWeightWR1(Rs[i], Re[i], Lat0, Lat0))
        wR2.append(np.sqrt(CalWeightWR2(Rs[i], Re[i], 90.0, Lat0) -
                   CalWeightWR2(Rs[i], Re[i], Lat0, Lat0)))
        wA2.append(np.sqrt(CalWeightWA2(Rs[i], Re[i], 90.0, Lat0) -
                   CalWeightWA2(Rs[i], Re[i], Lat0, Lat0)))

    fig, axs = plt.subplots(1, 1, squeeze=False, figsize=(6, 3))
    axs[0, 0].text(0.02, 0.98, r'$E_{\mathrm{min}}=$'+'{:>4.1f}'.format(Ele0)+r'$^\circ$',
                   transform=axs[0, 0].transAxes, ha='left', va='top',
                   fontdict={'fontsize': 14, 'fontname': 'Arial'})
    axs[0, 0].plot(y, wR1, '.r', label=r'$\omega_\mathrm{R}$', ms=4)
    axs[0, 0].plot(y, wR2, 'vg', label=r'$\omega_{\mathrm{R}^2}$', ms=2)
    axs[0, 0].plot(y, wA2, 'sb', label=r'$\omega_{\mathrm{A}^2}$', ms=2)
    axs[0, 0].legend(ncol=1, loc='upper right', bbox_to_anchor=(1.0, 1.0),
                     framealpha=0.3, prop={'family': 'Arial', 'size': 14})
    axs[0, 0].grid(which='major', axis='y', c='darkgray', ls='--', lw=0.4)
    axs[0, 0].set_axisbelow(True)

    axs[0, 0].set_ylabel('SISRE Weights', fontname='Arial', fontsize=16)
    for tl in axs[0, 0].get_yticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    axs[0, 0].set_xlabel(r'$r^s/r$', fontname='Arial', fontsize=16)
    # axs[0, 0].xaxis.set_major_formatter('{x:7.1f}')
    for tl in axs[0, 0].get_xticklabels():
        tl.set_fontname('Arial')
        tl.set_fontsize(14)

    strTmp = OutFilePrefix+OutFileSuffix+'.png'
    fig.savefig(strTmp, dpi=900, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.svg'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    strTmp = OutFilePrefix+OutFileSuffix+'.pdf'
    fig.savefig(strTmp, transparent=True, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    import argparse

    # cWhere='Local'
    # cWhere='GFZ'
    cWhere = 'HWC'
    if cWhere == 'Local':
        # Local mappings
        cWrkPre0 = r'Y:/'
        cDskPre0 = r'Z:/'
    elif cWhere == 'GFZ':
        # GFZ section cluster
        cWrkPre0 = r'/wrk/hanbing/'
        cDskPre0 = r'/dsk/hanbing/'
    elif cWhere == 'HWC':
        # Huawei cloud server
        cWrkPre0 = r'/home/hanbing/phb111/wrk/'
        cDskPre0 = r'/home/hanbing/phb111/dsk/'
    else:
        sys.exit('Unknow environment: '+cWhere)
    print('Run On '+cWhere)

    LatList = [14.5, 19.8, 19.8,
               13.9, 19.1, 24.3,
               13.2, 18.4, 23.5,
               12.4, 17.6, 22.7,
               8.7, 13.8, 18.8,
               9.4, 8.1]
    EleList = [0.0,  5.0, 10.0,
               0.0,  5.0, 10.0,
               0.0,  5.0, 10.0,
               0.0,  5.0, 10.0,
               0.0,  5.0, 10.0,
               0.0,  0.0]
    # list of geocenter distance of the satellite
    RadList = [25440, 25440, 25440,
               26560, 26560, 26560,
               27900, 27900, 27900,
               29600, 29600, 29600,
               42164, 42164, 42164,
               39002, 45326]
    # list of geocenter distance of the site
    ReaList = [6371, 6371, 6371,
               6371, 6371, 6371,
               6371, 6371, 6371,
               6371, 6371, 6371,
               6371, 6371, 6371,
               6371, 6371]

    # for i in range(len(LatList)):
    #     Ele = Lat2Ele(RadList[i], 6371, LatList[i], True)
    #     Lat = Lat2Ele(RadList[i], 6371, EleList[i], False)
    #     wR1, wR2, wA2 = CalWeight(RadList[i], ReaList[i], LatList[i])
    #     print('{: >8.2f} {: >8.2f} {: >6.4f} {: >6.4f} {: >6.4f}'.format(
    #         Ele, Lat, wR1, wR2, wA2))

    # print(Lat2Ele(25440, 6371, 19.8, True))
    # print(Lat2Ele(25440, 6371, 5.0, False))

    RadList = np.linspace(6371+300, 45326, 100, dtype=np.float64)
    # Receiver on the Earth ground
    ReaList = np.full(100, 6371, dtype=np.float64)
    OutFilePrefix = os.path.join(cDskPre0, r'SISRE_Test/')
    OutFileSuffix = 'Weight_S_0'
    PlotWeight(RadList, ReaList, 0.0, OutFilePrefix, OutFileSuffix)

    # RadList = np.linspace(6371+500, 45326, 100, dtype=np.float64)
    RadList = np.full(100, 6371+21528, dtype=np.float64)
    # ReaList = np.full(100, 6371, dtype=np.float64)
    ReaList = np.linspace(6371+500, 6371+2000, 100, dtype=np.float64)
    OutFilePrefix = os.path.join(cDskPre0, r'SISRE_Test/')
    OutFileSuffix = 'Weight_R_BDS_MEO'
    PlotWeight(RadList, ReaList, 0.0, OutFilePrefix, OutFileSuffix)
