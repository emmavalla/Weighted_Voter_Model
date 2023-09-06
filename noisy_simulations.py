import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import gamma
import branching_theory_code as branching

#-----------------------------------------------------------------------------------------------------------------------

# Finds Distribution Boundaries (within which most of the probability lies)
def gamma_lower(x, *args):
    quality,current_shape = args
    lower = 0+0.00001
    x_cdf = gamma.cdf(x,current_shape, scale = quality/current_shape)
    return np.abs(x_cdf - lower)
def gamma_upper(x, *args):
    quality,current_shape = args
    upper = 1-0.00001
    x_cdf = gamma.cdf(x,current_shape, scale = quality/current_shape)
    return np.abs(x_cdf - upper)
def find_boundaries(quality,current_shape):
    x0 = np.array([quality])
    res_lower = minimize(gamma_lower, x0, args = (quality,current_shape), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    res_upper = minimize(gamma_upper, x0, args = (quality,current_shape), method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    boundaries = np.array([res_lower.x, res_upper.x])
    return boundaries


# Discretizes distribution
def discretize(quality, current_shape, bounds, size):
    lower_boundary = bounds[0]
    upper_boundary = bounds[1]
    x = np.linspace(1,size,size)-0.5
    unit = (upper_boundary-lower_boundary)/size
    x = lower_boundary + (x*unit)
    gamma_vals = gamma.cdf(x+ (unit/2),current_shape, scale = quality/current_shape) - gamma.cdf(x- (unit/2),current_shape, scale = quality/current_shape)
    return x, gamma_vals


# Finds Invariant distribution, from the discretized distribution
def find_invariant(noisy_vals, noisy_dist):
    size = np.size(noisy_vals)
    noisy_dist = np.reshape(noisy_dist, (-1, 1))
    r_rates = 1 / (noisy_vals)

    Q_mat = np.transpose(noisy_dist * r_rates)
    Q_mat = Q_mat - (np.eye(size) * Q_mat)
    Q_mat = Q_mat - (np.sum(Q_mat, axis=1) * np.eye(size))
    Q_mat[:, np.shape(Q_mat)[0] - 1] = 1
    Q_inv = np.linalg.inv(Q_mat)

    b = np.zeros((size), dtype=float)
    b[size - 1] = 1

    b_trans = np.matmul(b, Q_inv)
    b_trans = np.matmul(Q_mat, b_trans)

    invariant = np.linalg.solve(Q_mat, b_trans)
    return invariant


# Returns discrete distribution, and invariant distribution
def all_gamma_dist(quality, scale, granularity):
    bounds = find_boundaries(quality, scale)
    vals, dist = discretize(quality, scale, bounds, size=granularity)
    invariant = find_invariant(vals, dist)
    return vals, dist, invariant

#-----------------------------------------------------------------------------------------------------------------------

# Add gamma noise
def gamma_noise(value, current_shape):
    gamma_noise = np.random.gamma(shape = current_shape,scale=value/current_shape)
    return gamma_noise

# Initial Distribution as dictated by size biased gamma dist for site B preferring nodes, and gamma for site A preferring nodes
def gamma_initialize(agent_preferences, values, current_shape, invariant_values, invariant_dist):
    initialization = np.zeros((np.size(agent_preferences)),dtype = float)
    initialization[agent_preferences ==0] = np.random.choice(invariant_values, size = np.sum(agent_preferences==0), p=invariant_dist, replace = True)
    initialization[agent_preferences ==1] = np.random.gamma(shape=current_shape,scale=(values[1]/current_shape) , size= np.sum(agent_preferences==1))
    return initialization


#----------SIMULATION-------------------------------------------------------------------------------------------------

def branching_gamma_invariant(values , n,  trials, shape, invariant_values, invariant_dist):
    index = (np.linspace(0, n - 1, n, dtype=int))
    counts_1 = 0

    for i in range(0, trials):
        agent_preferences = np.zeros((n), dtype=int)
        agent_preferences[0] = 1
        agent_parameters = gamma_initialize(agent_preferences, values, shape, invariant_values, invariant_dist)
        end = False

        while end == False:
            current_prop = np.sum(agent_preferences) / n
            agent_rates = (1 / agent_parameters)
            agent_probs = agent_rates / np.sum(agent_rates)

            agent_index = np.random.choice(index, 1, p=agent_probs)
            agent_preferences[agent_index] = np.random.binomial(1, current_prop)
            agent_parameters[agent_index] = gamma_noise(values[agent_preferences[agent_index]],  shape )

            end = sum(agent_preferences) == n or sum(agent_preferences) == 0

        counts_1 = counts_1 + np.int(agent_preferences[0])
    print(counts_1)
    return counts_1



#------------DEMO------------------------------------------------------------------------------------------------------

n = 100
trials = 1000
site_0_values = np.array([0.1,0.5,0.9], dtype = float)
shape_values = np.linspace(0.5, 5, 10)
shape_values_theory = np.linspace(0.01, 5, 30)
survival = np.zeros((np.size(site_0_values),np.size(shape_values)), dtype = float)
survival_theory = np.zeros((np.size(site_0_values),np.size(shape_values_theory)), dtype = float)
granularity = 100

# Simulations
i = 0
for current_scale in shape_values:
    for j in range(0,np.size(site_0_values)):
        # finding the invariant distribution of the worst site
        invariant_values, gamma_dist, invariant_dist  = all_gamma_dist(site_0_values[j], current_scale, granularity)
        # normalizing dist etc
        invariant_dist[invariant_dist<0] = 0
        invariant_dist = (1 / np.sum(invariant_dist))*invariant_dist
        # trials
        values = np.array((site_0_values[j],1.0), dtype = float)
        survival[j,i] = branching_gamma_invariant(values , n,  trials, current_scale, invariant_values, invariant_dist)/trials
    i = i + 1


# Theoretical Results
i = 0
for current_shape in shape_values_theory:
    vals_1, dist_1, invariant_1 = all_gamma_dist(1, current_shape, granularity)
    for j in range(0, np.size(site_0_values)):
        extinction = np.sum(dist_1 *branching.find_extinction(site_0_values[j], vals_1, dist_1))
        survival_theory[j,i] =  1 - extinction
    i = i + 1







