import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime as dt
from uszipcode import SearchEngine
from math import *
from pprint import pprint
from seirsplus.models import *
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import imageio
import copy
import numba
import os


def haversine(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 3956 # Radius of earth: 6371 kilometer, use 3956 for miles
    return c * r
