import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from uszipcode import SearchEngine


import datetime as dt
from math import *
import os
from collections import Counter
import csv
import ssl
import sys


ssl._create_default_https_context = ssl._create_unverified_context


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

#from seirsplus.models import *
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


all_counties = [13001.0, 13003.0, 13005.0, 13007.0, 13009.0, 13011.0, 13013.0, 13015.0, 13017.0, 13019.0, 13021.0, 13023.0, 13025.0, 13027.0, 13029.0, 13031.0, 13033.0, 13035.0, 13037.0, 13039.0, 13043.0, 13045.0, 13047.0, 13049.0, 13051.0, 13053.0, 13055.0, 13057.0, 13059.0, 13061.0, 13063.0, 13065.0, 13067.0, 13069.0, 13071.0, 13073.0, 13075.0, 13077.0, 13079.0, 13081.0, 13083.0, 13085.0, 13089.0, 13087.0, 13091.0, 13093.0, 13095.0, 13097.0, 13099.0, 13101.0, 13103.0, 13105.0, 13107.0, 13109.0, 13111.0, 13113.0, 13115.0, 13117.0, 13119.0, 13121.0, 13123.0, 13127.0, 13129.0, 13131.0, 13133.0, 13135.0, 13137.0, 13139.0, 13141.0, 13143.0, 13145.0, 13147.0, 13149.0, 13151.0, 13153.0, 13155.0, 13157.0, 13159.0, 13161.0, 13163.0, 13165.0, 13167.0, 13169.0, 13171.0, 13173.0, 13175.0, 13177.0, 13179.0, 13181.0, 13183.0, 13185.0, 13187.0, 13193.0, 13195.0, 13197.0, 13189.0, 13191.0, 13199.0, 13201.0, 13205.0, 13207.0, 13209.0, 13211.0, 13213.0, 13215.0, 13217.0, 13219.0, 13221.0, 80013.0, 13223.0, 13225.0, 13227.0, 13229.0, 13231.0, 13233.0, 13235.0, 13237.0, 13239.0, 13241.0, 13243.0, 13245.0, 13247.0, 13249.0, 13251.0, 13253.0, 13255.0, 13257.0, 13259.0, 13261.0, 13263.0, 13267.0, 13269.0, 13271.0, 13273.0, 13275.0, 13277.0, 13279.0, 13281.0, 13283.0, 13285.0, 13287.0, 13289.0, 90013.0, 13291.0, 13293.0, 13295.0, 13297.0, 13299.0, 13301.0, 13303.0, 13305.0, 13307.0, 13309.0, 13311.0, 13313.0, 13315.0, 13317.0, 13319.0, 13321.0]


#import numba

# @numba.jit(nopython=True)
def gillespie_():
    #initial
    n_S = self.S0
    n_I = self.I0
    n_R = self.R0
    list_S, list_I, list_R = [self.S0], [self.I0], [self.R0]

    #create status of each agent
    #0 denotes susceptible
    #1 denotes infected
    #-1 denotes recovered
    status = np.array([0]*self.S0+[1]*self.I0+[-1]*self.R0)
    np.random.shuffle(status)
    list_t = [0]

    #de facto deep copy
    list_status = [i for i in status]

    #unless every patient is recovered
    #or t has reach the maximum elapsed time
    while self.t < self.tmax:
        if n_I == 0:
            break

        #compute propensity
        propensity1 = self.infection_rate * n_S * n_I
        propensity2 = self.recovery_rate * n_I
        propensity_all = propensity1 + propensity2


        #tau leaping
        tau =- np.log(np.random.rand())/propensity_all
        self.t = self.t + tau

        currently_infected = np.array([i for i in range(self.adjmatrix.shape[0]) if status[i] == 1])

        if np.random.rand() < propensity1/propensity_all:

            #if people around susceptible cannot spread disease
            #we have to terminate the infection
            #otherwise we're stuck in infinitive loops
            currently_susceptible = [i for i in range(self.adjmatrix.shape[0]) if status[i]==0]

            #this part looks confusing
            #it is equivalent to [status[j] for i in currently_susceptible for j in graph.neighbors(i)]
            neighbor_status = [status[j] for i in currently_susceptible for j in [ii for ii in range(self.adjmatrix.shape[0]) if self.adjmatrix[ii][i]==1]]
            if (neighbor_status).count(1) == 0:
                continue

            #randomly select an infected patient
            #randomly select one of her/his susceptible connections
            #infect that poor soul
            stop = False
            while not stop:
                selected=np.random.choice(currently_infected,1)[0]

                #this part looks confusing
                #it is equivalent to [i for i in graph.neighbors(selected) if status[i]==0]
                connections = np.array([i for i in [ii for ii in range(self.adjmatrix.shape[0]) if self.adjmatrix[ii][selected]==1] if status[i]==0])
                if connections.shape != (0,):
                    new_infected = np.random.choice(connections,1)[0]
                    stop = True

            #update data
            status[new_infected] = 1
            n_S = n_S - 1
            n_I = n_I + 1

        else:

            #cure a random infected patient
            selected=np.random.choice(currently_infected,1)[0]

            #update data
            status[selected] = -1
            n_I = n_I - 1
            n_R = n_R + 1

        #update data
        list_S.append(n_S)
        list_I.append(n_I)
        list_R.append(n_R)
        list_t.append(self.t)

        #doesnt support nested list
        #thus, we need to improvise
        list_status += [i for i in status]

    self.list_t = list_t
    self.list_status = list_status
    self.matrix_status= [list_status[i:i + self.N] for i in range(0,len(list_status), self.N)]


    return list_t, list_status, list_S, list_I, list_R


if __name__ == '__main__':
    print(county_demographics("Appling, GA"))
