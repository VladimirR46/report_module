import numpy as np
import pyxdf
import logging
import json
import argparse
from pathlib import Path
from scipy.signal import firwin, butter, filtfilt, sosfiltfilt, welch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s.%(msecs)04d %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

FRONTAL_CHANNELS      = np.array(['Fp1', 'Fp2', 'F7',  'Fz', 'F8'])
CENTRAL_L_CHANNELS    = np.array(['T3', 'C3', 'P3', 'F3'])
CENTRAL_R_CHANNELS    = np.array(['T4', 'C4', 'P4', 'F4'])
OCCIPITAL_CHANNELS    = np.array(['O1', 'O2','Pz','T6','T5'])

REGIONS = {
    'frontal': FRONTAL_CHANNELS,
    'central_l': CENTRAL_L_CHANNELS,
    'central_r': CENTRAL_R_CHANNELS,
    'occipital': OCCIPITAL_CHANNELS,
}

