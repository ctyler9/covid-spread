from DataManipulation import *
import datetime

## MODEL
class gispPrediction():
    def __init__(self, tmax, t, infection_rate, recovery_rate, S0, I0, R0, graph, index):
        self.tmax = tmax
        self.t = 0
        self.infection_rate = infection_rate
        self.recovery_rate = recovery_rate
        self.S0 = S0
        self.I0 = I0
        self.R0 = R0
        self.N = S0 + I0 + R0

        self.graph = graph
        self.adjmatrix = nx.to_numpy_array(graph)
        self.colorlist = ['#99B898','#FECEAB','#FF847C']
        self.index = index

    def gillespie(self):
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

        # # prototype for individual region starting
        # for i in self.index.values():
        #     sus = np.zeros(self.S0)
        #     inf = np.ones(self.I0)
        #     rec = -1 * np.ones(self.R0)

        #     status = np.concatenate([sus, inf, rec])

        #status = da


        #unless every patient is recovered
        #or t has reach the maximum elapsed time
        #nothing is gonna stop us now
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

            #numba doesnt support nested list
            #thus, we need to improvise
            list_status += [i for i in status]

        self.list_t = list_t
        self.list_status = list_status
        self.matrix_status= [list_status[i:i + self.N] for i in range(0,len(list_status), self.N)]


        return list_t, list_status, list_S, list_I, list_R

    def create_data(self, dataMatcher, iRate, rRate):
        #unpack
        recovered_color,susceptible_color,infected_color = self.colorlist

        #create dataframe
        data = pd.DataFrame(index = self.graph.nodes, columns = self.list_t)
        for i in range(len(data.columns)):
            data[data.columns[i]] = self.matrix_status[i]

        #cleanup
        df = data.T
        df.reset_index(inplace=True)
        dt = '2020-04-18'
        c_dt = datetime.datetime.strptime('2020-04-18', "%Y-%m-%d")
        #discrete time rounding
        df['time'] = df['index'].apply(int).diff(-1)
        df.at[0, 'time'] = -1
        df.at[len(df)-1, 'time'] = -1
        df = df.loc[df[df['time'] != 0].index]
        df['real time'] = df['index'].apply(lambda x : int(x) + 1)
        df.at[0,'real time'] = 0
        df['real time'] = df['real time'].apply(lambda x: c_dt + pd.to_timedelta(x,unit='d'))
        df['real time'] = df['real time'].apply(lambda x: x.strftime("%Y-%m-%d"))
        df['date'] = df['real time']
        #cleanup
        del df['index']
        del df['time']
        del df['real time']
        df.set_index('date',inplace=True)
        df = df.T

        df = df.replace(-1, 0)
        df = df.reset_index()
        print(df)
        # cols = list(df.columns)
        # cols.remove("date")
        # cols = [col.split(".")[0] for col in cols]
        # print(cols)
        df['county'] = df['index'].str.split('.', expand=True)[0]
        print(df)
        df = df.drop(columns=['index'])
        print(df)
        df = df.groupby(['county'], as_index=False).sum()
        print(df)

        self.df = df
        df.to_csv("output_{0}_{1}.csv".format(iRate, rRate))

        return df

    def plot_graph(self, save_gif=False):
        #unpack
        recovered_color,susceptible_color,infected_color = self.colorlist

        #fix positions
        pos = nx.get_node_attributes(self.graph, 'pos')

        #everytime there is a status change
        #we visualize it
        for i in self.df.columns:
            plt.title(f't={i}')

            #replace value with color
            nx.draw(self.graph,pos,node_color=self.df[i].replace({-1:recovered_color,
                                                       0:susceptible_color,
                                                       1:infected_color}), node_size=10)

            #create legends
            S = mlines.Line2D([],[],color=susceptible_color,
                                      marker='o',markersize=10,
                                      linewidth=0,label='Susceptible')
            I = mlines.Line2D([],[],color=infected_color,
                                      marker='o',markersize=10,
                                      linewidth=0,label='Infected')
            R = mlines.Line2D([],[],color=recovered_color,
                                      marker='o',markersize=10,
                                      linewidth=0,label='Recovered')
            plt.legend(handles=[S,I,R],loc=0)

            if save_gif:
                plt.savefig(f't{i}.png')
            plt.show()

        #create gif
        if save_gif:
            filenames=["t%d.png" % (ii) for ii in self.df.columns]
            images=list(map(lambda filename:imageio.imread(filename),
                            filenames))
            imageio.mimsave('movie.gif',images,duration=0.5)



def main():
    ## VARIABLES
    #maximum elapsed time
    tmax = 2

    #beginning time
    t = 0

    # import graph from John Hopkin's Data
    avar = UnitedStatesMap()
    all_counties = [13001.0, 13003.0, 13005.0, 13007.0, 13009.0, 13011.0, 13013.0, 13015.0, 13017.0, 13019.0, 13021.0, 13023.0, 13025.0, 13027.0, 13029.0, 13031.0, 13033.0, 13035.0, 13037.0, 13039.0, 13043.0, 13045.0, 13047.0, 13049.0, 13051.0, 13053.0, 13055.0, 13057.0, 13059.0, 13061.0, 13063.0, 13065.0, 13067.0, 13069.0, 13071.0, 13073.0, 13075.0, 13077.0, 13079.0, 13081.0, 13083.0, 13085.0, 13089.0, 13087.0, 13091.0, 13093.0, 13095.0, 13097.0, 13099.0, 13101.0, 13103.0, 13105.0, 13107.0, 13109.0, 13111.0, 13113.0, 13115.0, 13117.0, 13119.0, 13121.0, 13123.0, 13127.0, 13129.0, 13131.0, 13133.0, 13135.0, 13137.0, 13139.0, 13141.0, 13143.0, 13145.0, 13147.0, 13149.0, 13151.0, 13153.0, 13155.0, 13157.0, 13159.0, 13161.0, 13163.0, 13165.0, 13167.0, 13169.0, 13171.0, 13173.0, 13175.0, 13177.0, 13179.0, 13181.0, 13183.0, 13185.0, 13187.0, 13193.0, 13195.0, 13197.0, 13189.0, 13191.0, 13199.0, 13201.0, 13205.0, 13207.0, 13209.0, 13211.0, 13213.0, 13215.0, 13217.0, 13219.0, 13221.0, 80013.0, 13223.0, 13225.0, 13227.0, 13229.0, 13231.0, 13233.0, 13235.0, 13237.0, 13239.0, 13241.0, 13243.0, 13245.0, 13247.0, 13249.0, 13251.0, 13253.0, 13255.0, 13257.0, 13259.0, 13261.0, 13263.0, 13267.0, 13269.0, 13271.0, 13273.0, 13275.0, 13277.0, 13279.0, 13281.0, 13283.0, 13285.0, 13287.0, 13289.0, 90013.0, 13291.0, 13293.0, 13295.0, 13297.0, 13299.0, 13301.0, 13303.0, 13305.0, 13307.0, 13309.0, 13311.0, 13313.0, 13315.0, 13317.0, 13319.0, 13321.0]
    graph = avar.make_state("Georgia")
    index = avar.index_dict()
    output_lst = avar.county_output_info
    #initial parameters
    infection_rate = 0.001
    recovery_rate = 0.06

    S0, I0, R0 = avar.SIR()

    avar = gispPrediction(tmax, t, infection_rate, recovery_rate, S0, I0, R0, graph, index)
    avar.gillespie()

    avar.create_data(output_lst, infection_rate, recovery_rate)
    #avar.plot_graph()



if __name__ == '__main__':
    main()
