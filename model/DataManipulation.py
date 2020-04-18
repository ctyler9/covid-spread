from util import *


class UnitedStatesMap():
    def __init__(self):
        self.case_df = pd.read_csv("../data/c_data.csv")
        self.states_graph = None
        self.state_list = self.case_df["Province_State"].unique()


        # total information
        self.population_dict = dict()
        self.infected_dict = dict()
        self.recovered_dict = dict()
        self.death_dict = dict()

        # total counts
        self.number_susceptible = 0
        self.number_infected = 0
        self.number_recovered = 0
        self.idx_dict = dict()


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
        # number of nodes per thousand to represent the population
        population = temp['population'] / 100
        g_ratio = 15/446.47

        # 1 degree of coordinates = 69 miles
        # base square block off of population
        square_block = g_ratio * population # in miles (15 old)
        edge_block = square_block**(1/2)
        degree_conversion = edge_block/69

        # Degree boundaries
        degree_west = lat - degree_conversion
        degree_east = lat + degree_conversion
        degree_north = lat + degree_conversion
        degree_south = lat - degree_conversion


        #G.add_node(county, pos=(lat, lon))
        count = 0
        for i in range(int(population)):
            lat_r = np.random.uniform(degree_west, degree_east)
            lon_r = np.random.uniform(degree_south, degree_north)
            G.add_node(county + "_" + str(i), pos=(lat_r, lon_r))
            count += 1

        self.idx_dict[county] = count

        return G

    def add_edges_county(self, county, radius=0.5):
        graph = self.county_plot(county)
        all_nodes = nx.get_node_attributes(graph, 'pos')
        node_key = list(all_nodes.keys())

        for person1, coordinates1 in all_nodes.items():
            for person2, coordinates2 in all_nodes.items():
                dist = haversine(coordinates1[0], coordinates1[1], coordinates2[0], coordinates2[1])
                if dist < radius and dist != 0:
                   graph.add_edge(person1, person2)

        while nx.number_connected_components(graph) != 1:
            node1_k = np.random.choice(node_key)
            node2_k = np.random.choice(node_key)

            if nodes_connected(graph, node1_k, node2_k) == False:
                graph.add_edge(node1_k, node2_k)

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
                combined_graph = nx.compose(combined_graph, cd[key])
            count += 1

        self.G = combined_graph
        return combined_graph

    def combine_connected_graphs(self, num, alist):
        combined_graph = self.connect_counties(alist)
        all_nodes = nx.get_node_attributes(combined_graph, 'pos')
        node_key = list(all_nodes.keys())

        count = 0
        while count <= num:
            node1_k = np.random.choice(node_key)
            node2_k = np.random.choice(node_key)

            name1 = node1_k.split('_')[0]
            name2 = node2_k.split('_')[0]
            if name1 != name2:
                node1 = all_nodes[node1_k]
                node2 = all_nodes[node2_k]
                if haversine(node1[0], node1[1], node2[0], node2[1]) < 100:
                    combined_graph.add_edge(node1_k, node2_k)
                    count += 1

        self.G = combined_graph
        return combined_graph

    def graph(self, G):
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, node_size=10)
        plt.show()

        return G

    def index_dict(self):
        index_dict = dict()

        for county in self.population_dict.keys():
            self.number_infected += self.infected_dict[county]/100
            self.number_recovered += (self.recovered_dict[county]/100 + self.death_dict[county]/100)
            self.number_susceptible += (self.population_dict[county]/100 - self.number_infected/100 - self.number_recovered/100)


        p_dict = {key: ceil(value/100) for key,value in self.population_dict.items()}
        i_dict = {key: ceil(value/100) for key,value in self.infected_dict.items() if key in self.population_dict.keys()}
        r_dict = {key: ceil(value/100) for key,value in self.recovered_dict.items() if key in self.population_dict.keys()}
        d_dict = {key: ceil(value/100) for key,value in self.death_dict.items() if key in self.population_dict.keys()}

        s_dict = dict(Counter(p_dict) - Counter(i_dict) - Counter(r_dict))
        r_dict = dict(Counter(r_dict) + Counter(d_dict))

        # Scale the values down by the constant
        # s_dict = {key: ceil(value) for key,value in s_dict.items()}
        # i_dict = {key: ceil(value) for key,value in i_dict.items()}
        # r_dict = {key: ceil(value) for key,value in r_dict.items()}

        return s_dict, i_dict, r_dict

    def SIR(self):
        self.index_dict()
        len_G = len(self.G)
        ns = ceil(self.number_susceptible)
        ni = ceil(self.number_infected)
        nr = ceil(self.number_recovered)

        total = ns + ni + nr

        if total < len_G:
            ns += (len_G - total)
        elif total > len_G:
            ns -= (total - len_G)
        else:
            pass

        total = ns + ni + nr

        return ns, ni, nr


def main():
    avar = UnitedStatesMap()
    #las = avar.add_edges_county("Fulton")
    #print(avar.graph(las))

    sdl = avar.combine_connected_graphs(20, ["Fulton", "Henry", "Cobb", "DeKalb", "Bacon", "Clayton"])
    #sdl = avar.county_dict(10)
    print(avar.graph(sdl))
    # print(avar.index_dict())
    # print(avar.SIR())
    #print(avar.graph(atl))


if __name__ == '__main__':
    main()



