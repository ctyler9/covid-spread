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
import overpass
from collections import Counter

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

def county_search(county):
    search = SearchEngine(simple_zipcode=False)
    county = search.by_city(county)
    return county

def county_bounds(county):
    api = overpass.API()
    resp = api.get("""[out:json];relation[name="{}"];out bb;""".format(county), build=False, responseformat="json")
    return resp["elements"][0]["bounds"]

def county_demographics(county):
    temp = county_search(county)

    population = list()
    population_density = list()
    median_household_income = list()
    for instance in temp:
        population.append(instance.population)
        population_density.append(instance.population_density)
        median_household_income.append(instance.median_household_income)
    pop = np.sum(population)
    pop_d = np.average(population_density)
    mhi = np.average(median_household_income)

    return {"population": pop, "population_density": pop_d, "median_household_income": mhi}

def sers(graph):
    G_normal = custom_exponential_graph(baseGraph, scale=100)
    # Social distancing interactions:
    G_distancing = custom_exponential_graph(baseGraph, scale=10)
    # Quarantine interactions:
    G_quarantine = custom_exponential_graph(baseGraph, scale=5)

    model = SEIRSNetworkModel(G=G_normal, beta=0.155, sigma=1/5.2, gamma=1/12.39, mu_I=0.0004, p=0.5,
                              Q=G_quarantine, beta_D=0.155, sigma_D=1/5.2, gamma_D=1/12.39, mu_D=0.0004,
                              theta_E=0.02, theta_I=0.02, phi_E=0.2, phi_I=0.2, psi_E=1.0, psi_I=1.0, q=0.5,
                              initI=10)

    checkpoints = {'t': [20, 100], 'G': [G_distancing, G_normal], 'p': [0.1, 0.5], 'theta_E': [0.02, 0.02], 'theta_I': [0.02, 0.02], 'phi_E':   [0.2, 0.2], 'phi_I':   [0.2, 0.2]}

    model.run(T=500, checkpoints=checkpoints)

    model.figure_infections()

def nodes_connected(graph, u, v):
    return nx.node_connected_component(graph, u) == nx.node_connected_component(graph, v)


if __name__ == '__main__':
    print(county_demographics("Appling, GA"))
