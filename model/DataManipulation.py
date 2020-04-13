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


class DataManipulation():
    def __init__(self):
        self.case_df = pd.read_csv("../data/c_data.csv")
        self.graph = None

    def city_search(self, city):
        search = SearchEngine(simple_zipcode=False)
        city = search.by_city(city)
        return city

    def haversine(self, lon1, lat1, lon2, lat2):
        # convert decimal degrees to radians
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a))
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles
        return c * r

    def state_dict(self):
        df_us = self.case_df.loc[self.case_df["Country_Region"] == "US"]

        state_dict = dict()
        for state in df_us["Province_State"].unique():
            G = nx.Graph()
            temp = df_us.loc[df_us["Province_State"] == state]
            lat = temp["Lat"].tolist()
            lon = temp["Long_"].tolist()
            city = temp["Combined_Key"].tolist()


            for lat, lon, city_ in zip(lat, lon, city):
                G.add_node(city_, pos=(lat, lon))
            state_dict[state] = G

        self.graph = state_dict

        return state_dict

    def country_dict(self):
        df = self.case_df

        country_dict = dict()
        for country in df["Country_Region"].unique():
            G = nx.Graph()
            temp = df.loc[df["Country_Region"] == country]
            lat = temp["Lat"].tolist()
            lon = temp["Long_"].tolist()
            city = temp["Combined_Key"].tolist()

            for lat, lon, city_ in zip(lat, lon, city):
                G.add_node(city_, pos=(lat, lon))


            country_dict[country] = G

        return country_dict

    def graph_state(self, state):
        graph = self.add_edges_state(state)

        pos = nx.get_node_attributes(graph, 'pos')
        nx.draw(graph, pos, node_size=10)
        plt.show()

        return graph

    def graph_country(self, country):
        country_graph = self.country_dict()
        c = country_graph[country]

        pos = nx.get_node_attributes(c, 'pos')
        nx.draw(c, pos, node_size=10)
        plt.show()

    def add_edges_state(self, state):
        graph = self.state_dict()
        s = graph[state]


        cities = nx.get_node_attributes(s, 'pos')
        for city1, coordinates1 in cities.items():
            for city2, coordinates2 in cities.items():
                dist = self.haversine(coordinates1[0], coordinates1[1], coordinates2[0], coordinates2[1])
                if dist < 30 and dist != 0:
                    s.add_edge(city1, city2)

        return s

    def city_demographics(self, state):
        graph = self.state_dict()
        s = graph[state]

        cities = nx.get_node_attributes(s, 'pos')
        city_dict = dict()
        for city in tqdm(list(cities.keys())[0:3]):
            temp = self.city_search(city)

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

            city_dict[city] = {"population": pop, "population_density": pop_d, "median_household_income": mhi}


        return city_dict

    def sers(self):
        baseGraph = self.graph_state("Georgia")
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


if __name__ == '__main__':
    avar = DataManipulation()
    print(avar.graph_state("Georgia"))




