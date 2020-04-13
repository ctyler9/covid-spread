from util import *


class UnitedStatesMap():
    def __init__(self):
        self.case_df = pd.read_csv("../data/c_data.csv")
        self.states_graph = None
        self.state_list = self.case_df["Province_State"].unique()


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

        county = [county.split(',')[0] for county in county]

        self.county_list = county

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


    def add_edges_county(self, county, radius):
        graph = self.county_plot(county)

        people = nx.get_node_attributes(graph, 'pos')
        for person1, coordinates1 in people.items():
            for person2, coordinates2 in people.items():
                dist = haversine(coordinates1[0], coordinates1[1], coordinates2[0], coordinates2[1])
                if dist < radius and dist != 0:
                    graph.add_edge(person1, person2)

        return graph

    def connect_counties(self, county_list=None):
        if county_list == None:
            self.county_df()
            county_list = self.county_list

        count = 0
        for county in county_list:
            if count == 0:
                G_nodes = self.add_edges_county(county, 0.7)
            else:
                temp = self.add_edges_county(county, 0.7)
                comb = nx.disjoint_union(G_nodes, temp)
            count += 1

        return comb


    def graph(self, G):
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, node_size=10)
        plt.show()

        return G


if __name__ == '__main__':
    avar = UnitedStatesMap()
    atl = avar.connect_counties(["Gwinnett", "Fulton"])
    print(avar.graph(atl))




