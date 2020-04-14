from util import *


class UnitedStatesMap():
    def __init__(self):
        self.case_df = pd.read_csv("../data/c_data.csv")
        self.states_graph = None
        self.state_list = self.case_df["Province_State"].unique()
        self.population_dict = dict()
        self.infected_dict = dict()
        self.recovered_dict = dict()
        self.death_dict = dict()

        # population counts
        self.number_susceptible = 0
        self.number_infected = 0
        self.number_recovered = 0

    def state_df(self, state):
        # Filter df by State
        df_us = self.case_df.loc[self.case_df["Country_Region"] == "US"]
        state_df = df_us.loc[df_us["Province_State"] == state]

        return state_df


    def county_df(self, state):
        # Get list of cities for that state
        temp = self.state_df(state)
        lat = temp["Lat"].tolist()
        lon = temp["Long_"].tolist()
        county = temp["Combined_Key"].tolist()

        infected = temp["Confirmed"].tolist()
        deaths = temp["Deaths"].tolist()
        recovered = temp["Recovered"].tolist()

        county = [county.split(',')[0] for county in county]

        self.county_list = county
        self.infected_dict = dict(zip(county, infected))
        self.recovered_dict = dict(zip(county, recovered))
        self.death_dict = dict(zip(county, deaths))

        return dict(zip(county, zip(lon, lat)))

    def county_plot(self, county):
        G = nx.Graph()
        county_dict = self.county_df("Georgia")
        lon, lat = county_dict[county]

        ## Exceptions
        try:
            temp = county_demographics(county)
        except:
            print('error')
            pass
        if lat == None:
            pass


        self.population_dict[county] = temp['population']
        print(self.population_dict)
        # number of nodes per thousand to represent the population
        population = temp['population'] / 100
        # 1 degree of coordinates = 69 miles

        # base square block off of population
        square_block = 25 # in miles
        edge_block = square_block**(1/2)
        degree_conversion = edge_block/69

        # Degree boundaries
        degree_west = lat - degree_conversion
        degree_east = lat + degree_conversion
        degree_north = lat + degree_conversion
        degree_south = lat - degree_conversion


        #G.add_node(county, pos=(lat, lon))
        for i in range(int(population)):
            lat_r = np.random.uniform(degree_west, degree_east)
            lon_r = np.random.uniform(degree_south, degree_north)
            G.add_node(county + str(i), pos=(lat_r, lon_r))

        return G


    def add_edges_county(self, county, radius=0.7):
        graph = self.county_plot(county)

        people = nx.get_node_attributes(graph, 'pos')
        for person1, coordinates1 in people.items():
            for person2, coordinates2 in people.items():
                dist = haversine(coordinates1[0], coordinates1[1], coordinates2[0], coordinates2[1])
                if dist < radius and dist != 0:
                    graph.add_edge(person1, person2)

        return graph

    def county_dict(self, county_list=None):
        if county_list == None:
            self.county_df()
            county_list = self.county_list

        county_dict = dict()
        for county in county_list:
            county_dict[county] = self.add_edges_county(county)

        return county_dict

    def connect_counties(self, county_list):
        cd = self.county_dict(county_list)
        count = 0
        for key in cd.keys():
            if count == 0:
                combined_graph = cd[key]
            else:
                combined_graph = nx.disjoint_union(combined_graph, cd[key])
            count += 1

        print(self.population_dict)

        return combined_graph


    def graph(self, G):
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, node_size=10)
        plt.show()

        return G

    def index_dict(self):
        index_dict = dict()
        for county in self.population_dict.keys():
            self.number_infected += self.infected_dict[county]
            self.number_recovered += self.recovered_dict[county] + self.death_dict[county]
            self.number_susceptible += (self.population_dict[county] - self.number_infected - self.number_recovered)

            index_dict[county] = (self.population_dict[county] - self.infected_dict[county],
                self.infected_dict[county])

        return index_dict

    def SIR(self):
        return int(self.number_susceptible/100), int(self.number_infected/100), int(self.number_recovered/100)


if __name__ == '__main__':
    avar = UnitedStatesMap()
    atl = avar.connect_counties(["Fulton"])
    print(avar.index_dict())
    print(avar.infected)
    #print(avar.graph(atl))




