import numpy as np
import networkx as nx

# Theoretical Results - Complete Graph ----------------------------------------------------------------------------------
#Expected Probability of reaching Consensus on best option
def success(values,n,k):
    a = values[0]
    b = values[1]
    if a == b:
        value = (k/n)
    else :
        numerator = np.power((a / b), k) - 1
        denominator = np.power((a / b), n) - 1
        value = numerator / denominator
    return value

#Expected time  for all values of k
def calculating_times(values,n):
    a = values[0]
    b = values[1]
    k = np.arange(1,n)
    rate_forward = (1/a) * (k*(n-k))/((k)+(n-k))
    rate_backward = (1/b) * (k*(n-k))/((k)+(n-k))
    mean_time = 1/(rate_forward + rate_backward)
    prob_forward = b/(b+a)
    prob_backward = a/(a+b)

    rates = np.zeros(((n-1),(n-1)),dtype = float)
    index = np.arange(0,(n-1))
    rates[index[1:(n-1)],index[0:(n-2)]]=prob_backward
    rates[index[0:(n-2)],index[1:(n-1)]]=prob_forward
    rates = rates - np.identity(n-1)
    final = np.linalg.solve(rates,-mean_time)
    return(final)

# Simulations Noise-Free -----------------------------------------------------------------------------------------------

# Simulation - Complete Graph
def trial_complete_clean(values, n, k, trials):
    #avg_time, counts_1 = 0, 0
    count_array = np.zeros(trials)
    time_array = np.zeros(trials)
    values = 1 / values
    for i in range(0, trials):
        agents_1 = k
        time = 0
        end = agents_1 == n or agents_1 == 0
        while end == False:
            parameter_0 = (n - agents_1)*values[0]
            parameter_1 = agents_1*values[1]
            parameter_sum = parameter_0 + parameter_1
            agent_increase = parameter_0*agents_1/(n*parameter_sum)
            agent_decrease = parameter_1*(n-agents_1)/(n*parameter_sum)
            occurence_prob = agent_decrease+agent_increase
            p = (agent_increase/occurence_prob)
            agents_1 = agents_1 + (np.random.binomial(1,p)*2)-1
            time = time + np.random.exponential(1 /(occurence_prob*parameter_sum))
            end = agents_1 == n or agents_1 == 0
        count_array[i] = np.int(agents_1==n)
        time_array[i] = time
    print([np.sum(count_array), np.mean(time_array)])
    return count_array, time_array


# Simulation - Complete Graph with lag
def trial_complete_clean_lag(values, lag_time, n, k, trials):
    value_rates = 1 / values
    lag_rate = 1/lag_time #so lag_time is 0.1 if on average the lag is 1/10 as long as the signals for the best option
    counts_array = np.zeros(trials)
    times_array = np.zeros(trials)
    jumps = np.zeros([6,4], int)
    jumps[0,:] = [-1,0,1,0] #agent 0 choosing 0 (entering lag state)
    jumps[1,:] = [-1,0,0,1] #agent 0 choosing 1 (entering lag state)
    jumps[2,:] = [0,-1,1,0] #agent 1 choosing 0 (entering lag state)
    jumps[3,:] = [0,-1,0,1] #agent 1 choosing 1 (entering lag state)
    jumps[4,:] = [1,0,-1,0] #lag 0 finishing
    jumps[5,:] = [0,1,0,-1] #lag 1 finishing
    for i in range(0, trials):
        # agents 0, agents 1, agents lag 0, agents lag 1
        agents = np.array([n-k,k,0,0])
        time = 0
        agents_1 = agents[1] + agents[3]
        end = agents_1== n or agents_1 == 0
        while end == False:
            # the rates at which any of the jumps are made
            parameters = np.zeros(6)
            parameters[0] = agents[0] * value_rates[0] * (agents[0]/n)
            parameters[1] = agents[0] * value_rates[0] * (agents[1]/n)
            parameters[2] = agents[1] * value_rates[1] * (agents[0]/n)
            parameters[3] = agents[1] * value_rates[1] * (agents[1]/n)
            parameters[4] = agents[2]*lag_rate
            parameters[5] = agents[3]*lag_rate
            parameters_prob = parameters/np.sum(parameters)
            jump_type = np.random.choice(6,p = parameters_prob)
            agents = agents + jumps[jump_type,:]
            time = time + np.random.exponential(1 /np.sum(parameters))
            agents_1 = agents[1] + agents[3]
            end = agents_1 == n or agents_1 == 0
        #print(agents_1==n)
        counts_array[i] = np.int(agents_1==n)
        times_array[i] = time
    print([(np.sum(counts_array)/trials), np.mean(times_array)])
    return counts_array, times_array


# Simulation - Complete Graph with Many Options (best-of-n with n>2)
def trial_complete_clean_qualities(values, n, k_array, trials):
    #avg_time = 0
    #count_array = np.zeros_like(values)
    time_array = np.zeros(trials)
    count_array = np.zeros(trials)
    value_rates = 1 / values
    opinion_number = np.size(values)
    transition_rates = np.zeros((opinion_number,opinion_number))
    for i in range(0, trials):
        agents = np.copy(k_array)
        time = 0
        end = np.any(agents ==n)
        while end == False:
            # fill out transition rate matrix
            for j in range(0,opinion_number):
                for k in range(0,opinion_number):
                    transition_rates[j,k] = agents[j]*value_rates[j]*agents[k]/n
            transition_probs = transition_rates/np.sum(transition_rates)
            jump_type = np.random.choice((opinion_number*opinion_number),p = transition_probs.flatten())
            jump_from = np.floor(jump_type/opinion_number).astype(int)
            jump_to = (jump_type - (jump_from*opinion_number)).astype(int)
            agents[jump_from] = agents[jump_from] -1
            agents[jump_to] = agents[jump_to] + 1
            time = time + np.random.exponential(1 /np.sum(transition_rates))
            end = np.any(agents ==n)
        poss_opinions = np.arange(opinion_number)
        count_array[i] = int(poss_opinions[agents==n])
        time_array[i] = time
    return count_array, time_array


# Simulation - Any graph with static adjacency matrix
# (Used for Regular Expander graph, Random ER Graphs, and Random Regular Graphs)
# One graph used for all trials

#Connected Adjacency Matrices for Random Graphs (ER and geometric)
def random_geometric_adj_mat(current_n,current_r, current_seed):
    connected = False
    while connected == False:
        random_graph = nx.random_geometric_graph(n = current_n, radius = current_r, seed= current_seed)
        connected = nx.is_connected(random_graph)
    random_adj = nx.to_numpy_matrix(random_graph,dtype = int)
    return random_adj

def random_ER_adj_mat(current_n,current_p, seed_number, current_seed):
    connected = False
    while connected == False:
        random_graph = nx.erdos_renyi_graph(n = current_n, p = current_p, seed= current_seed)
        connected = nx.is_connected(random_graph)
    random_adj = nx.to_numpy_matrix(random_graph,dtype = int)
    return random_adj


def trial_adj_clean(values, n, k, trials, adj_mat):
    #avg_time, counts_1 = 0, 0
    count_array = np.zeros(trials)
    time_array = np.zeros(trials)
    values = 1 / values
    index = (np.linspace(0, n - 1, n, dtype=int))
    for i in range(0, trials):
        agent_preferences = np.zeros((n), dtype=int)
        agent_preferences[np.random.choice(n, k, replace=False)] = 1
        agent_parameters = values[agent_preferences]
        time = 0
        end = False
        while end == False:
            parameter_sum = np.sum(agent_parameters)
            time = time + np.random.exponential(1 / parameter_sum)
            node_probabilities = agent_parameters / parameter_sum
            node_increment = int(np.random.choice(index, 1, p=node_probabilities))
            node_chosen = int(np.random.choice(index[np.array(adj_mat[node_increment, :]==1).flatten()], 1, replace=False))
            agent_preferences[node_increment] = agent_preferences[node_chosen]
            agent_parameters[node_increment] = values[agent_preferences[node_increment]]
            end = sum(agent_preferences) == n or sum(agent_preferences) == 0
        count_array[i] = int(agent_preferences[0])
        time_array[i] = time
    return count_array, time_array

# Simulation - Random Regular Graph (new adj mat selected at each trial)
# New graph sampled at each trial

def trial_regular_clean(values, n, d, k, trials):
    #avg_time, counts_1 = 0, 0
    count_array = np.zeros(trials)
    time_array = np.zeros(trials)
    values = 1 / values
    for i in range(0, trials):
        adj_mat = nx.to_numpy_matrix(nx.random_regular_graph(d,n, seed= int(i)),dtype = int)
        index = (np.linspace(0, n - 1, n, dtype=int))
        agent_preferences = np.zeros((n), dtype=int)
        agent_preferences[np.random.choice(n, k, replace=False)] = 1
        agent_parameters = values[agent_preferences]
        time = 0
        end = False
        while end == False:
            parameter_sum = np.sum(agent_parameters)
            time = time + np.random.exponential(1 / parameter_sum)
            node_probabilities = agent_parameters / parameter_sum
            node_increment = int(np.random.choice(index, 1, p=node_probabilities))
            node_chosen = int(np.random.choice(index.reshape((1,np.size(index)))[np.array(adj_mat[node_increment, :],dtype = bool)], 1, replace=False))
            agent_preferences[node_increment] = agent_preferences[node_chosen]
            agent_parameters[node_increment] = values[agent_preferences[node_increment]]
            end = sum(agent_preferences) == n or sum(agent_preferences) == 0
        count_array[i] = int(agent_preferences[0])
        time_array[i] = time
    return count_array, time_array


# Simulation - Resampled Random Regular Graph

# Rewires random regular graph when a node updates (denoted by 'node_moving')
def random_regular_rewire(adj_mat,d,node_moving):
    node_number = np.shape(adj_mat)[0]
    node_list = np.arange(node_number)
    edges_to_rewire = d
    #finding nodes to sever
    nodes_severed = node_list[np.array(adj_mat[node_moving,:]==1).flatten()]
    #severing edges
    adj_mat[nodes_severed,node_moving] =0
    adj_mat[node_moving,nodes_severed] =0
    #new node to find
    viable_node_list = node_list[node_list !=node_moving]
    while edges_to_rewire >0:
        new_node = np.random.choice(viable_node_list)
        # check to see whether it is an old node
        if np.any(nodes_severed==new_node):
            #reduce the list of nodes_severed
            nodes_severed = nodes_severed[nodes_severed!=new_node]
        else:
            #find neighbours
            new_node_neighbours = node_list[np.array(adj_mat[new_node,:]==1).flatten()]
            new_node_neighbours = np.random.permutation(new_node_neighbours)
            i = 0
            pairing_found = False
            #choose between its neighbours
            while pairing_found == False:
                #check if new_node_neighbour is viable
                new_node_neighbour_neighbours = node_list[np.array(adj_mat[new_node_neighbours[i],:]==1).flatten()]
                #check that those doing contain node_severed[0]
                if new_node_neighbours[i]!=nodes_severed[0] and np.all(new_node_neighbour_neighbours!=nodes_severed[0]):
                    pairing_found = True
                    #sever between new node neighbour and new node
                    adj_mat[new_node_neighbours[i],new_node] = 0
                    adj_mat[new_node,new_node_neighbours[i]] = 0
                    #attach between new node neighbour and severed node
                    adj_mat[new_node_neighbours[i],nodes_severed[0]] = 1
                    adj_mat[nodes_severed[0],new_node_neighbours[i]] = 1
                    #print('adjusted')
                i = i +1
            nodes_severed = nodes_severed[nodes_severed!= nodes_severed[0]]
        # attach new edge between new node and moving node
        adj_mat[new_node,node_moving] =1
        adj_mat[node_moving,new_node] =1
        #reduce edges to rewire
        edges_to_rewire = edges_to_rewire -1
        viable_node_list = viable_node_list[viable_node_list!=new_node]
    return adj_mat


#Paired Simulation that calls to have the graph rewired at every update point
def trial_regular_resampling(values, n, d, k, trials):
    avg_time, counts_1 = 0, 0
    values = 1 / values
    for i in range(0, trials):
        adj_mat = nx.to_numpy_matrix(nx.random_regular_graph(d,n, seed= int(i)),dtype = int)
        index = (np.linspace(0, n - 1, n, dtype=int))
        agent_preferences = np.zeros((n), dtype=int)
        agent_preferences[np.random.choice(n, k, replace=False)] = 1
        agent_parameters = values[agent_preferences]
        time = 0
        end = False
        while end == False:
            parameter_sum = np.sum(agent_parameters)
            time = time + np.random.exponential(1 / parameter_sum)
            node_probabilities = agent_parameters / parameter_sum
            node_increment = int(np.random.choice(index, 1, p=node_probabilities))
            node_chosen = int(np.random.choice(index.reshape((1,np.size(index)))[np.array(adj_mat[node_increment, :],dtype = bool)], 1, replace=False))
            agent_preferences[node_increment] = agent_preferences[node_chosen]
            agent_parameters[node_increment] = values[agent_preferences[node_increment]]
            adj_mat = random_regular_rewire(adj_mat,d,node_increment)
            end = sum(agent_preferences) == n or sum(agent_preferences) == 0
        counts_1 = counts_1 + int(agent_preferences[0])
        avg_time = avg_time + (time / trials)
        print(i)
    return counts_1, avg_time

#------------ DEMO COMPLETE GRAPH ------------------------------------------------------------------------------------------------------

# Simulation results for the complete graph
n_array = np.array([20, 40, 60, 80, 100, 200, 400, 600, 800,1000])
k = 1
values = np.array([0.9,1])
trials = 1000

for i in range(0,np.size(n_array)):
    n_value = n_array[i]
    count_array , time_array = trial_complete_clean(values, n_value, k, trials)

# Predicted Results
n_predicted_array = np.linspace(2,1000,500, dtype=int)
counts_predicted = np.zeros(np.size(n_predicted_array))
time_predicted = np.zeros(np.size(n_predicted_array))

for i in (0, np.size(n_predicted_array)):
    n_value = n_predicted_array[i]
    counts_predicted[i] = success(values,n_value,k)
    time_predicted[i] = calculating_times(values,n_value)[k-1]