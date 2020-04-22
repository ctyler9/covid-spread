from util import *


class UnitedStatesMap():
    def __init__(self):
        global division_num
        global max_node_degree
        global max_num_components

        division_num = 100
        max_node_degree = 2
        max_num_components = 1

        # Get current data url
        yesterday = dt.date.today() - dt.timedelta(days=1)
        str_time = yesterday.strftime("%m-%d-%Y")
        url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_daily_reports/{}.csv".format(str_time)

        self.case_df = pd.read_csv(url)
        self.case_df['Lat'].replace('', np.nan, inplace=True)
        self.case_df.dropna(subset=['Lat'], inplace=True)
        self.states_graph = None
        self.state_list = self.case_df["Province_State"].unique()

        self.population_df = pd.read_csv("../data/census_data.csv", encoding="ISO-8859-1")

        # total information
        self.population_dict = dict()
        self.infected_dict = dict()
        self.recovered_dict = dict()
        self.death_dict = dict()
        self.county_output_info = dict()
        self.county_list = dict()

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

    def county_population(self, state):
        # wont overlap
        df = self.population_df.loc[self.population_df["STNAME"] == state]

        county_dict = df[["CTYNAME", "POPESTIMATE2019"]]
        county_dict = county_dict.copy()
        county_dict["CTYNAME"] = county_dict["CTYNAME"].str.split(" Count", expand=True)[0]
        county_dict = county_dict.set_index("CTYNAME").to_dict()
        county_dict = county_dict["POPESTIMATE2019"]

        return county_dict

    def county_df(self, state):
        # Get list of cities for that state
        temp = self.state_df(state)
        if (state == 'Massachusetts'):
            temp = temp.fillna(value=25007.0)
        lat = temp["Lat"].tolist()
        lon = temp["Long_"].tolist()
        county_ = temp["Combined_Key"].tolist()

        infected = temp["Confirmed"].tolist()
        deaths = temp["Deaths"].tolist()
        recovered = temp["Recovered"].tolist()
        fips = temp["FIPS"].tolist()

        county = [county.split(',')[0] for county in county_ if county != state]
        state_labels = [county.split(',')[1] for county in county_]
        counties = [county.split(',')[0] for county in county_]

        self.county_list = fips
        self.county_labels = dict(zip(fips, counties))
        self.county_output_info = dict(zip(fips, zip(counties, state_labels)))
        self.infected_dict = dict(zip(fips, infected))
        self.recovered_dict = dict(zip(fips, recovered))
        self.death_dict = dict(zip(fips, deaths))


        return dict(zip(fips, zip(lon, lat)))


    def county_plot(self, county, state):
        G = nx.Graph()
        county_dict = self.county_df(state)
        #print(county_dict)
        lon, lat = county_dict[county]
        county_name = self.county_labels[county]
        county_pop_info = self.county_population(state)
        if (county_name == 'Dukes and Nantucket'):
            county_name = 'Dukes'

        ## Exceptions
        try:
            population = county_pop_info[county_name]
        except:
            print(str(county_name) + " not found")
            print("Generating random population for " + str(county_name))
            population = np.random.uniform(10000, 100000)
        if lat == None:
            pass

        # Print invidiual county names to see progress
        print(county_name)

        # divide by two because herd immunity makes assumption
        # that once half the population is infected, the virus
        # cannot really spread more
        self.population_dict[county] = population/2
        # number of nodes per thousand to represent the population
        population_scaled = population / division_num
        g_ratio = 15/1063.937

        # 1 degree of coordinates = 69 miles
        # base square block off of population
        square_block = g_ratio * population_scaled # in miles (15 old)
        edge_block = square_block**(1/2)
        degree_conversion = edge_block/69

        # Degree boundaries
        degree_west = lat - degree_conversion
        degree_east = lat + degree_conversion
        degree_north = lon + degree_conversion
        degree_south = lon - degree_conversion


        #G.add_node(county, pos=(lat, lon))
        count = 0
        for i in range(ceil(population_scaled)):
            lat_r = np.random.uniform(degree_west, degree_east)
            lon_r = np.random.uniform(degree_south, degree_north)
            G.add_node(str(county) + "_" + str(i), pos=(lon_r, lat_r))
            count += 1

        self.idx_dict[county] = count

        return G

    def add_edges_county(self, county, state, radius=0.5):
        graph = self.county_plot(county, state)
        all_nodes = nx.get_node_attributes(graph, 'pos')
        node_key = list(all_nodes.keys())

        for person1, coordinates1 in all_nodes.items():
            for person2, coordinates2 in all_nodes.items():
                dist = haversine(coordinates1[0], coordinates1[1], coordinates2[0], coordinates2[1])
                if dist < radius and dist != 0:
                    if len(graph.edges(person1)) <= max_node_degree-1 and len(graph.edges(person2)) <= max_node_degree-1:
                        graph.add_edge(person1, person2)

        while nx.number_connected_components(graph) > max_num_components:
            node1_k = np.random.choice(node_key)
            node2_k = np.random.choice(node_key)

            if nodes_connected(graph, node1_k, node2_k) == False:
                graph.add_edge(node1_k, node2_k)

        return graph

    def county_dict(self, state, county_list=None):
        if county_list == None:
            self.county_df(state)
            county_list = self.county_list

        county_dict = dict()
        for county in county_list:
            county_dict[county] = self.add_edges_county(county, state)

        return county_dict

    def connect_counties(self, state, county_list):
        cd = self.county_dict(state, county_list)
        count = 0
        for key in cd.keys():
            if count == 0:
                combined_graph = cd[key]
            else:
                combined_graph = nx.compose(combined_graph, cd[key])
            count += 1

        self.G = combined_graph
        return combined_graph

    def combine_connected_graphs(self, state, alist=None):
        if alist != None:
            combined_graph = self.connect_counties(state, alist)
        else:
            combined_graph = self.connect_counties(state, self.county_list)
        all_nodes = nx.get_node_attributes(combined_graph, 'pos')
        node_key = list(all_nodes.keys())
        node_key_truncated = set([county.split("_")[0] for county in node_key])

        count = 0
        used_names = list()
        while nx.number_connected_components(combined_graph) != 1:
            unused_names = list(node_key_truncated - set(used_names))
            node1_k = np.random.choice(node_key)
            node2_k = np.random.choice(node_key)

            name1 = node1_k.split('_')[0]
            name2 = node2_k.split('_')[0]
            if name1 != name2:
                node1 = all_nodes[node1_k]
                node2 = all_nodes[node2_k]
                if count <= int(len(self.county_list) * 1.25):
                    if haversine(node1[0], node1[1], node2[0], node2[1]) < 350:
                            combined_graph.add_edge(node1_k, node2_k)
                            used_names.append(name1)
                            used_names.append(name2)
                            count += 1
                else:
                    if len(unused_names) == 0:
                        break
                    used_names_edit = [name+"_"+str(0) for name in used_names]
                    unused_names_edit = [name+"_"+str(0) for name in unused_names]

                    node1_k = np.random.choice(used_names_edit)
                    node2_k = np.random.choice(unused_names_edit)

                    name1 = node1_k.split('_')[0]
                    name2 = node2_k.split('_')[0]

                    combined_graph.add_edge(node1_k, node2_k)
                    used_names.append(name2)
                    count += 1

            if count >= 1000:
                print(nx.number_connected_components(combined_graph))
                print("too many iterations, a county is out of reach, adjust distance constraint?")
                break

        self.G = combined_graph
        return combined_graph

    def make_state(self, state):
        self.county_df(state)
        return self.combine_connected_graphs(state)

    def graph(self, G):
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, node_size=10)
        plt.show()

        return G

    def index_dict(self):
        index_dict = dict()

        for county in self.population_dict.keys():
            self.number_infected += self.infected_dict[county]/division_num
            self.number_recovered += (self.recovered_dict[county]/division_num + self.death_dict[county]/division_num)
            self.number_susceptible += (self.population_dict[county]/division_num - self.number_infected/division_num - self.number_recovered/division_num)


        p_dict = {key: ceil(value/division_num) for key,value in self.population_dict.items()}
        i_dict = {key: ceil(value/division_num) for key,value in self.infected_dict.items() if key in self.population_dict.keys()}
        r_dict = {key: ceil(value/division_num) for key,value in self.recovered_dict.items() if key in self.population_dict.keys()}
        d_dict = {key: ceil(value/division_num) for key,value in self.death_dict.items() if key in self.population_dict.keys()}

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

    sdl = avar.make_state("Georgia")
    print(avar.graph(sdl))


if __name__ == '__main__':
    main()
