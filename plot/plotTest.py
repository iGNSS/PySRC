#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''Only for temporary test
'''
__author__ = 'hanbing'

# Standard library imports
import os
import sys
import os.path
import glob

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.interpolate import griddata

from astropy.timeseries import LombScargle
import astropy.units as u

# Local application/library specific imports
from PySRC.miscellaneous import GNSSTime


if __name__ == '__main__':
    import argparse

    print('OK')
