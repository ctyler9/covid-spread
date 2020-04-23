from DataManipulation import *

## MODEL
class gispPrediction():
    def __init__(self, tmax, t, infection_rate, recovery_rate, S0, I0, R0, graph, index):
        div_num = 1000

        self.tmax = tmax
        self.t = 0
        self.infection_rate = infection_rate / div_num
        self.recovery_rate = recovery_rate / div_num
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

    def create_data(self, state):
        #unpack
        recovered_color,susceptible_color,infected_color = self.colorlist

        #create dataframe
        data = pd.DataFrame(index = self.graph.nodes, columns = self.list_t)
        for i in range(len(data.columns)):
            data[data.columns[i]] = self.matrix_status[i]

        #cleanup
        df = data.T
        df.reset_index(inplace=True)
        adate = dt.date.today().strftime("%Y-%m-%d")
        c_dt = dt.datetime.strptime(adate, "%Y-%m-%d")


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

        df['county'] = df['index'].str.split('.', expand=True)[0]
        df = df.drop(columns=['index'])
        df = df.groupby(['county'], as_index=False).sum()

        df.to_csv("../data/output_{0}_{1}_{2}.csv".format(state, str(self.infection_rate), str(self.recovery_rate)))

        return df


def main(state_list, i_list, r_list, tmax, t):
    ## create CSVs for all states and infection rates
    for state in state_list:
        for infection_rate, recovery_rate in zip(i_list, r_list):
            us = UnitedStatesMap()
            graph = us.make_state(state)
            index = us.index_dict()
            infection_rate = 0.01
            recovery_rate = 0.06
            S0, I0, R0 = us.SIR()

            state_prediction = gispPrediction(tmax, t0, infection_rate, recovery_rate, S0, I0, R0, graph, index)
            state_prediction.gillespie()
            state_prediction.create_data(state)

            print("\n")


if __name__ == '__main__':
    ## VARIABLES
    state_list = ["New Hampshire", "Vermont", "Connecticut", "Maine"]
    i_list = [0.01, 0.05, 0.1]
    r_list = [0.01, 0.05, 0.1]

    #maximum elapsed time
    tmax = 4
    #beginning time
    t0 = 0


    main(state_list, i_list, r_list, tmax, t0)
