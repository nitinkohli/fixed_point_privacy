# ============================#
# =*= Necessary Packages =*= #
# ============================#

import itertools

import numpy as np
np.set_printoptions(linewidth = 150)

from scipy.stats import geom
from scipy.optimize import linprog
from scipy.linalg import null_space
from scipy.special import binom
import pandas as pd

import pywt

import time
import warnings
warnings.filterwarnings('ignore')
import signal

from collections import Counter

import opendp.prelude as dp

from scipy.stats import norm
from scipy.optimize import bisect

dp.enable_features('contrib')

# #### Generating Histograms

# ## Data 1: Draws from a binomial distribution

def draws_to_hist(values, n):
    return [values.count(i) for i in range(n)] ## note: we can't use the standard collections trick since we need to count 0 occurences

def binomial_hist(n, num_categories = 10000, p = 0.5):
    draws = np.random.binomial(n-1, p, size = num_categories)
    return draws_to_hist(draws.tolist(), n)

def uniform_hist(n, num_categories):
    draws = np.random.choice(n, size = num_categories)
    return draws_to_hist(draws.tolist(), n)

def right_skewed(n, num_categories):
    draws = (np.random.geometric(p=.5/n, size = num_categories)-1)%n
    return draws_to_hist(draws.tolist(), n)

def left_skewed(n, num_categories):
    draws = n - 1 - (np.random.geometric(p=.5/n, size = num_categories)-1)%n
    return draws_to_hist(draws.tolist(), n)

def top_inflated(n, num_categories):
    draws = np.concatenate([np.random.randint(low=0, high = n, size = int(num_categories *.99)) , np.zeros(int(num_categories*.01))+n-1] )
    return draws_to_hist(draws.tolist(), n)

def zero_inflated(n, num_categories):
    draws = np.concatenate([np.random.randint(low=0, high = n, size = int(num_categories *.98)) , np.zeros(int(num_categories*.02))] )
    return draws_to_hist(draws.tolist(), n)

'''
## Just to generate the first time, we now load it from a csv from now on

num_categories = 10000    
binom_n = 21
binom_hist_dataset = binomial_hist(binom_n, num_categories)

binom_df = pd.DataFrame(binom_hist_dataset)
binom_df.to_csv("binom_hist_data.csv")

'''

binom_hist_df = pd.read_csv("simulations_data/binom_hist_data.csv") 
binom_hist_dataset = binom_hist_df['0'].values
binom_n = len(binom_hist_dataset)

# ## Data 2: FBI Homicides

crime = pd.read_csv("simulations_data/crime_data.csv") 

crime_n = 51

def crime_hist(n):
    z_pre = crime.MURDER.apply(lambda x: min(x,n-1)).value_counts()
    new_index = pd.RangeIndex(start=0, stop=n)
    z_pre = z_pre.reindex(new_index, fill_value=0)
    return z_pre.values


# ## Data 3: Michigan Schools

schools = pd.read_csv("simulations_data/us-public-schools.csv", sep = ';') 
schools = schools[schools.FT_TEACHER != -999]

schools_n = 81

def schools_hist(n):
    z_pre = schools.FT_TEACHER.apply(lambda x: min(x,n-1)).value_counts()
    #z_pre = schools.staff.apply(lambda x: min(x,n-1)).value_counts()
    new_index = pd.RangeIndex(start=0, stop=n)
    z_pre = z_pre.reindex(new_index, fill_value=0)
    return z_pre.values

# ## Helper function to convert a histogram of counts to a distribution of counts; dtype = np.longdouble

def hist_to_dist(hist):
    hist = np.asarray(hist, dtype = np.longdouble)
    return hist / np.sum(hist)

# ## Figure 1 Data: Bird Flu Cases by State

bird_flu = pd.read_csv("simulations_data/bird_flu.csv")

# ######## Solvers

# =======================#
# =*= Our Heuristic =*= #
# =======================#

'''
Requires an input called selector, which is a function that maps an array of weighted column sums to a column index
'''

tolerance = 1e-25

def max_selector(col_sum):
    return np.argmax(col_sum)

def min_selector(col_sum):
    temp = col_sum.copy()
    temp[temp < tolerance] = np.inf
    return np.argmin(temp)

def random_selector(col_sum):
    temp = np.where(col_sum > tolerance)[0]
    return np.random.choice(temp)

def sandwich_selector(col_sum):
    n = len(col_sum)
    distance = np.abs(n/2 - np.arange(n))+1
    distance[col_sum < tolerance] = 0
    return np.argmax(distance)


def heuristic_constructor(z, epsilon, selector, tolerance = tolerance, print_progress=0):
    ## ==> set up
    n = len(z)
    row_sum = np.ones(n, dtype = np.longdouble) ##float128
    col_sum = z.copy()
    start_time = time.time()
    A = np.zeros((n, n))
    ## ==> as long as there is mass to allocate, do the following
    row_bindings = np.zeros(n-1)
    while np.sum(col_sum > tolerance) > 0 and np.any(row_sum > tolerance):
        ## ==> select column via selector
        col_num = selector(col_sum) 
        if col_sum[col_num] < tolerance:
            break
        if print_progress:
            print("Row Sum = ") 
            print(row_sum)
            print("Column Sum = ") 
            print(col_sum)
            print("Column number selected = ")
            print(col_num)
        old_ratios = np.zeros(n-1)
        while col_sum[col_num] > tolerance and np.any(row_sum > tolerance):
            ## ==> generate a concentrating pattern for this column
            ratios = [1]*(col_num) + [-1]*(n-col_num-1)
            ## ==> update ratios as needed
            for i in range(0, n-1):
                if ratios[i] == - row_bindings[i]:
                    ratios[i] *= -1
            ## ==> generate scale using the pattern from ratios
            padded_ratios = np.asarray([1]+ratios, dtype = np.longdouble)
            scale = np.exp(np.cumsum(epsilon * padded_ratios))
            scale = scale / np.sum(scale)
            ## ==> compute gamma pointwise
            gamma = col_sum[col_num] / np.dot(z,scale)
            binding_row = -1
            for a in range(0, len(row_sum)-1):
                check_a = scale[a+1] > scale[a]
                if check_a:
                    numerator = np.exp(epsilon)*row_sum[a+1] - row_sum[a]
                    denominator = np.exp(epsilon)*scale[a+1] - scale[a]
                    gamma_a = numerator / denominator
                else:
                    numerator = np.exp(epsilon)*row_sum[a] - row_sum[a+1]
                    denominator = np.exp(epsilon)*scale[a] - scale[a+1]
                    gamma_a = numerator / denominator
                if gamma_a < gamma and row_bindings[a] == 0:
                    binding_row = a
                    gamma = np.min([gamma_a, gamma])
            if binding_row > -1:
                row_bindings[binding_row] = - ratios[binding_row]
            amount = gamma 
            ## ==> insert this amount into A
            A[:,col_num] += amount*scale
            ## ==> update row and column sum respectively 
            col_sum[col_num] -= (amount * np.dot(z,scale))
            row_sum -= amount*scale
            old_ratios = np.array(ratios)
    ## ==> return the output matrix A, with the execution time 
    end_time = time.time()
    execution_time = end_time - start_time
    return({'corner':A, 'execution_time':execution_time})

def min_heuristic_constructor(z, epsilon):
    return heuristic_constructor(z, epsilon, selector = min_selector)

def max_heuristic_constructor(z, epsilon):
    return heuristic_constructor(z, epsilon, selector = max_selector)

def sandwich_heuristic_constructor(z, epsilon):
    return heuristic_constructor(z, epsilon, selector = sandwich_selector)



# #### Scipy Solvers



'''
This function assumes we've already flattened the objective as a vector
'''

def l1_deviation(a,b):
    return(np.abs(a-b))

def squared_deviation(a,b):
    return((a-b)**2)

def count_error_minimizing_fixed_point_constructor(z, epsilon, solve_method):
    z = list(z) ## note: the code is a bit wonky when z is a numpy array due to list concatenation operations below
    n = len(z)
    z_mat = np.tile(z, (n, 1)).T
    measure_mat = np.fromfunction(l1_deviation, (n,n))
    combined_mat = z_mat * measure_mat
    l1_objective = list(combined_mat.flatten())
    ## ==> construct equality constraints
    # first, start with the row sum constraints
    A_equal = []
    for i in range(n):
        A_equal.append([0]*(n*i) + [1]*(n) + [0]*(n**2 - ((i+1)*n)))
    # next, put in the fixed point constraints
    for i in range(n):
        temp = list(np.zeros(n**2))
        for j in range(len(z)):
            temp[(n*j)+i] = z[j]
        A_equal.append(temp)
    # construct the equal values
    b_equal = [1]*n + z
    #print([A_equal, b_equal])
    ## ==> construct inequality constraints
    # insert the differential privacy constraints
    A_upper = []
    for i in range(0,n-1):
        for j in range(0,n):
            temp = np.zeros((n,n))
            temp[i,j] = 1
            temp[i+1,j] = -np.exp(epsilon)
            A_upper.append(list(temp.flatten()))
            temp = np.zeros((n,n))
            temp[i,j] = -np.exp(epsilon)
            temp[i+1,j] = 1
            A_upper.append(list(temp.flatten()))
    # construct the upper bound values
    b_upper = [0]*len(A_upper)
    #print([A_upper, b_upper])
    ## ==> run scipy solver
    start_time = time.time()
    res = linprog(l1_objective, A_ub=A_upper, b_ub=b_upper, A_eq = A_equal, b_eq = b_equal, method = solve_method)
    end_time = time.time()
    execution_time = end_time - start_time
    if res.success == True:
        corner = np.array(res.x).reshape((n,n))
    else:
        corner = None
    return({'corner':corner, 'execution_time':execution_time})

def interior_method_fixed_point_constructor(z, epsilon):
    return count_error_minimizing_fixed_point_constructor(z, epsilon, solve_method = 'highs')

def simplex_fixed_point_constructor(z, epsilon):
    return count_error_minimizing_fixed_point_constructor(z, epsilon, solve_method = 'simplex')

# #### Unconstrained Baseline Constructor


def concentrating_scale(peak, n, epsilon):
    ratios = [1]*(peak) + [-1]*(n-peak-1)
    padded_ratios = np.asarray([1]+ratios)
    scale = np.exp(np.cumsum(epsilon * padded_ratios))
    return scale / np.sum(scale)

def unfixed_baseline_constructor(z, epsilon):
    n = len(z)

    z_mat = np.tile(z, (n, 1)).T
    measure_mat = np.fromfunction(l1_deviation, (n,n))
    combined_mat = z_mat * measure_mat

    start_time = time.time()
    
    concentrating_scale_matrix = np.zeros((n,n))
    for column in range(n):
        concentrating_scale_matrix[:,column] = concentrating_scale(column, n, epsilon)
    amount_of_each_scale = np.linalg.solve(concentrating_scale_matrix, np.ones(n))
    
    
    corner = np.zeros((n,n))
    current_column = 0
    current_scale = 0
    
    while current_scale < n:
        current_error = combined_mat[:,current_column] @ concentrating_scale(current_scale, n, epsilon)
        if current_column == n-1:
            next_error = np.inf
        else:
            next_error = combined_mat[:,current_column+1] @ concentrating_scale(current_scale, n, epsilon)
        if next_error > current_error:
            corner[:,current_column] += amount_of_each_scale[current_scale] * concentrating_scale(current_scale, n, epsilon)
            current_scale += 1
        else:
            current_column +=1

    end_time = time.time()
    execution_time = end_time - start_time

    return({'corner':corner, 'execution_time':execution_time})



# #### Truncated Geometric

def truncated_geom(z, epsilon):
    n = len(z)
    start_time = time.time()
    T = np.zeros((n, n), dtype=float)
    for x in range(n):
        for y in range(n):
            if y == 0:
                T[x, y] = np.exp(-epsilon * x) / (1 - np.exp(-epsilon))
            elif y == n - 1:
                T[x, y] = np.exp(epsilon * (x - n + 1)) / (1 - np.exp(-epsilon))
            else:
                T[x, y] = np.exp(-epsilon * abs(y - x))
    # Normalize row to sum to 1
        row_sum = np.sum(T[x, :])
        if row_sum > 0:
            T[x, :] /= row_sum
    end_time = time.time()
    execution_time = end_time - start_time
    return {'corner': T, 'execution_time': execution_time} 


# #### Staircase Mechanism over R

# # This is Algorithm 1 of Geng et al (pdf available here: https://pramodv.ece.illinois.edu/pubs/GKOV.pdf)

def staircase_mechanism(true_value, epsilon, sensitivity):
    rng = np.random.default_rng()
    gamma = 1.0 / (1.0 + np.exp(epsilon / 2.0)) ## this is the default proposed by Geng et al. in Equation (11)
    sign = -1 if rng.random() < 0.5 else 1
    p = 1 - np.exp(-epsilon)
    geom_rv = rng.geometric(p) - 1
    unif_rv = rng.random()
    binary_prob = gamma / (gamma + (1 - gamma) * np.exp(-epsilon))
    binary_rv = 0 if rng.random() < binary_prob else 1

    if binary_rv == 0:
        noise = geom_rv + gamma * unif_rv
    else:
        noise = geom_rv + gamma + (1 - gamma) * unif_rv

    return true_value + (sign * sensitivity * noise)

def staircase(array, epsilon, sensitivity = 1):
    return [staircase_mechanism(x, epsilon, sensitivity) for x in array]



# #### Discrete Gaussian over R

def find_dg_scale(epsilon, delta, Delta = 1):
    '''
    Solve for sigma in Theorem 7 of the Canonne, Kamath, and Steinke (2024)
    
    Given privacy parameters epsilon, delta, and sensitivity,
    return the minimal sigma such that discrete Gaussian noise
    satisfies (epsilon, delta)-DP.
    '''
    def delta_fn(sigma):
        eps = epsilon
        s = sigma

        t1 = eps * s**2 / Delta - Delta / 2
        t2 = eps * s**2 / Delta + Delta / 2

        P1 = norm.sf(t1, loc=0, scale=s)  # This is Pr[X > t1]
        P2 = norm.sf(t2, loc=0, scale=s)  # Similarly, this is Pr[X > t2]

        delta_val = P1 - np.exp(eps) * P2
        return delta_val

    ## Find sigma such that delta_fn(sigma) = delta using bisection root finding
    sigma_opt = bisect(lambda s: delta_fn(s) - delta, 1e-5, 100, xtol=1e-8)
    return sigma_opt


def discrete_gaussian_mechanism(output, dg_scale):
    input_space = dp.atom_domain(T=int, nan=False), dp.absolute_distance(T=float)
    gaussian = dp.m.make_gaussian(*input_space, scale=dg_scale)
    return gaussian(output)


def discrete_gaussian(array, epsilon, sensitivity = 1):
    delta = 1 / (np.sum(array) + 1)
    sigma = find_dg_scale(epsilon, delta)
    return [discrete_gaussian_mechanism(x, sigma) for x in array]


#### Rule of thumb used to decide on splitting privacy budget when needed in two-stage framework
# Exponential decay model
def f_exp(x, A, B, K):
    return B + (A - B) * np.exp(-K * x)

def rule_of_thumb(epsilon_t):
    '''
    Values are derived from epsilon_split_experiment notebook
    '''
    return f_exp(epsilon_t, 0.6390268955896642, 0.1061220926959348, 2.869164606789786)


# ### Error Measurement Utilities

'''
Given a matrix mat and fixed point z, compute the noise as per the measurement function
'''

def count_error(transition_matrix, z, measure):
    if type(transition_matrix) == type(None):
        return(None)
    else:
        n = transition_matrix.shape[0]
        ## ==> construct z_matrix
        z_mat = np.tile(z, (n, 1)).T
        ## ==> construct measure_mat
        weight_matrix = z_mat * np.fromfunction(measure, (n,n))
        ## ==> return the sum of the Hadamard product of the above matrices
        return(np.sum(weight_matrix * transition_matrix))

def total_deviation(dist1, dist2):
    return 0.5 * np.sum(np.abs(dist1 - dist2))

def KS_distance(dist1, dist2):
    return np.max(np.abs(dist1.cumsum() - dist2.cumsum()))

def wasserstein_1_distance(dist1, dist2):
    return np.sum(np.abs(dist1.cumsum() - dist2.cumsum()))


##### Distribution Privatizers
def laplace_mechanism(histogram, dist_privatizer_epsilon):
    ## Laplace mechanism for a vector of counts, so global sensitivity is 2.
    priv_hist = np.asarray(histogram) + np.random.laplace(loc = 0.0, scale = 2./(dist_privatizer_epsilon), size = len(histogram))
    ## Post-process
    priv_hist[priv_hist <= 0] = 0
    ## Return
    return priv_hist

def cyclic_laplace_mechanism(histogram, dist_privatizer_epsilon):
    ## Drawn n Laplaces with global sensitivity of 1, since this is a vector of counts
    lap_vals = np.random.laplace(loc = 0.0, scale = 1./(dist_privatizer_epsilon), size = len(histogram))
    lap_vals = np.concatenate([lap_vals, [lap_vals[0]]])
    priv_hist = np.asarray([histogram[i] + lap_vals[i] - lap_vals[i+1] for i in range(len(histogram))])
    ## Post-process
    priv_hist[priv_hist <= 0] = 0
    ## Return
    return priv_hist


def pad_to_power_of_two(arr):
    length = len(arr)
    next_pow2 = 1 << (length - 1).bit_length()  # Next power of two
    pad_width = next_pow2 - length
    padded = np.pad(arr, (0, pad_width), mode='constant', constant_values=0)
    return padded

def privelet_haar_mechanism(histogram, dist_privatizer_epsilon):
    ## check if power of 2, and pad otherwise
    pre_n = len(histogram)
    if (pre_n & (pre_n - 1)) != 0:
        histogram = pad_to_power_of_two(histogram)
    ## Now, proceed with privelet
    histogram = np.asarray(histogram, dtype=float)
    n = len(histogram)
    #assert (n & (n - 1)) == 0, "Length must be a power of two."
    # Compute Haar wavelet coefficients
    coeffs = pywt.wavedec(histogram, 'haar')
    # Calculate weight for each coefficient
    weight = [n]
    for i in range(1,len(coeffs)):
        weight.append(n / 2**(i-1))
    # Compute the level-wise scale
    gen_sens = 1 + int(np.log2(n))
    level_lap_scales = [2 * gen_sens / (w*dist_privatizer_epsilon) for w in weight]
    # Add Laplace noise to all coefficients
    #return coeffs, level_lap_scales
    noisy_coeffs = [coeffs[l] + np.random.laplace(0, level_lap_scales[l], size=coeffs[l].shape) for l in range(len(level_lap_scales))]
    # Reconstruct privatized data
    privatized= pywt.waverec(noisy_coeffs, 'haar')
    priv_hist = privatized[:pre_n] ## note that we need to remove some because of the padding
    ## Post-process
    priv_hist[priv_hist <= 0] = 0
    ## Return
    return priv_hist

def qyl_mechanism(histogram, dist_privatizer_epsilon, branching_factor):
    '''
    H_b^c from Qardaji–Yang–Li (VLDB 2013), Section 3.3.
    We represent this as a list of numpy arrays, one per level (index 0 = leaves; last = root).

    The steps below do the following:
      (1) build exact counts per level,
      (2) add Laplace noise (epsilon split equally across all levels, including root),
      (3) bottom-up weighted averaging (z_i[v]),
      (4) top-down mean consistency (bar n_i[v]).
    '''
    hist = np.asarray(histogram, dtype=float)
    n = len(hist)
    b = int(branching_factor)

    ## Paper levels: i=1,...,h, leaves at i=1, root at i=h
    ## Choose h so L=b^(h-1) >= n
    h = int(np.ceil(np.log(n) / np.log(b) - 1e-12)) + 1
    L = b ** (h - 1)

    ## Pad leaves to full b-ary level
    if L > n:
        leaves_true = np.concatenate([hist, np.zeros(L - n, dtype=float)])
    else:
        leaves_true = hist.copy()

    ## (1) True counts per level (list-of-arrays, leaves -> root)
    true_levels = [leaves_true]
    for _ in range(1, h):
        prev = true_levels[-1]
        parents = prev.reshape(-1, b).sum(axis=1)
        true_levels.append(parents)


    ## (2) Add Laplace noise to evert level (even including root); split the epsilon equally
    eps_per_level = dist_privatizer_epsilon / h
    lap_scale = 1.0 / eps_per_level 
    rng = np.random.default_rng()
    noisy_levels = [arr + rng.laplace(0.0, lap_scale, size=arr.shape) for arr in true_levels]


    ## (3) Bottom-up weighted averaging (Section 3.3)
    z_levels = [None] * h
    z_levels[0] = noisy_levels[0].copy()  ## i=1 (leaves)

    for level in range(1, h):
        i = level + 1  # paper's level index
        alpha_i = (b**i - b**(i - 1)) / (b**i - 1.0)
        sum_children = z_levels[level - 1].reshape(-1, b).sum(axis=1)
        z_levels[level] = alpha_i * noisy_levels[level] + (1.0 - alpha_i) * sum_children


    ## (4) Top-down mean consistency (Section 3.3)
    nbar_levels = [None] * h
    nbar_levels[-1] = z_levels[-1].copy()  ## root unchanged

    for level in range(h - 2, -1, -1):
        parent_bar = nbar_levels[level + 1].reshape(-1, 1)  ## (P,1)
        child_z = z_levels[level].reshape(-1, b)            ## (P,b)
        adjust = (parent_bar - child_z.sum(axis=1, keepdims=True)) / float(b)
        nbar_levels[level] = (child_z + adjust).reshape(-1)

    ## Return de-padded leaves within clipping to non-negative values
    est = nbar_levels[0][:n]
    est = np.maximum(0.0, est)
    return est


# #### Two-stage counting Framework Simulation Code

### For those that use a transition matrix with z
def create_simulate_run_from_constructor(constructor):
    def simulate_run(emp_hist, laplace_epsilon, counting_epsilon):
        result = {'method': constructor.__name__}
        emp_z = hist_to_dist(emp_hist)
    
        ## Privatize the distribution (if laplace epsilon is less than infinity)
        if laplace_epsilon < np.inf:
            z = hist_to_dist(cyclic_laplace_mechanism(emp_hist, laplace_epsilon))
        else:
            z = emp_z

        ## Use constructor to build matrix M using z
        M = constructor(z, counting_epsilon)['corner']

        result['L1 noise'] = count_error(M, emp_z, l1_deviation)
        result['MSE'] = count_error(M, emp_z, squared_deviation)

        output_hist = np.zeros(len(emp_hist))
        for i in range(len(emp_hist)):
            if emp_hist[i] > 0:
                output_vals_i = np.random.choice(len(emp_hist), emp_hist[i], p = M[i, :]/np.sum(M[i, :]))
                output_freq_i = Counter(output_vals_i) ## this is a dictionary with keys equal to the unique entries in output_list_i, and values equal to the count of the entries in the output_list_i
                ## now, update the output_hist by putting in the occurences from output_freq_i in the correct place
                for k in output_freq_i:
                    output_hist[k] += output_freq_i[k]
        ## Compute M's empirical output distribution
        output_dist = output_hist / np.sum(output_hist)
        ## return the total deviation between the two distributions
        result['KS distance'] = KS_distance(emp_z, output_dist)
        result['total deviation'] = total_deviation(emp_z, output_dist)
        result['wasserstein'] = wasserstein_1_distance(emp_z, output_dist)
   
        return result
    return simulate_run

### For those that use a transition matrix but not z
def simulate_run_truncated_geom(emp_hist, laplace_epsilon, counting_epsilon):
    result = {'method': 'truncated_geom'}
    emp_dist = hist_to_dist(emp_hist)    
    ## Compute the truncated geometric
    M = truncated_geom(emp_hist, laplace_epsilon + counting_epsilon)['corner']

    result['L1 noise'] = count_error(M, emp_dist, l1_deviation)
    result['MSE'] = count_error(M, emp_dist, squared_deviation)

    
    ## For each point in emp_hist, sample from M based on the point's value and get output histogram
    output_hist = np.zeros(len(emp_hist))
    for i in range(len(emp_hist)):
        if emp_hist[i] > 0:
            output_vals_i = np.random.choice(len(emp_hist), emp_hist[i], p = M[i, :]/np.sum(M[i, :]))
            output_freq_i = Counter(output_vals_i) ## this is a dictionary with keys equal to the unique entries in output_list_i, and values equal to the count of the entries in the output_list_i
            ## now, update the output_hist by putting in the occurences from output_freq_i in the correct place
            for k in output_freq_i:
                output_hist[k] += output_freq_i[k]
    ## Compute M's empirical output distribution
    output_dist = hist_to_dist(output_hist)

    ## return the total deviation between the two distributions
    result['KS distance'] = KS_distance(emp_dist, output_dist)
    result['total deviation'] = total_deviation(emp_dist, output_dist)
    result['wasserstein'] = wasserstein_1_distance(emp_dist, output_dist)

    return result


### For those that use a mechanism but not z
def postprocess_counts(dp_toc, upper_bound):
    new_dp_toc = (np.round(dp_toc)+ 0.00001).astype(int) ## adding a bit of buffer
    new_dp_toc = np.clip(new_dp_toc, a_min=0, a_max=upper_bound) ## this includes the right-endpoint
    return new_dp_toc


# # This function computes the count error and distribution error by comparing the privatized outputs to the true outputs
# # Note that there is no laplace_epsilon in this

def create_simulate_run_from_alg_description(alg_description):
    def simulate_run_alg_description(emp_hist, laplace_epsilon, counting_epsilon):
        result = {'method': alg_description.__name__}
        emp_dist = hist_to_dist(emp_hist)

        ## Using the mechanism, sample noise and add to the table of counts, truncating to 0 and n-1 if necessary
        # (a) First, create the table of counts from the true emp_hist
        toc = np.array([i for i, count in enumerate(emp_hist) for a in range(count)], dtype=np.int32)

        # (b) Produce a table of DP counts, truncating as needed
        dp_toc = alg_description(toc, laplace_epsilon + counting_epsilon)
        dp_toc = postprocess_counts(dp_toc, len(emp_hist) - 1)

        ## Compute the count error empirically
        result['L1 noise'] = np.mean(np.abs(dp_toc - toc))
        result['MSE'] = np.mean((dp_toc - toc)**2)

        ## Next, measure the distributional distance
        # (a) First, get the output distribution based on this privatization strategy
        output_hist = np.bincount(dp_toc, minlength = len(emp_hist))
        output_dist = hist_to_dist(output_hist)

        ## (b) Return various measures of the distribution
        result['KS distance'] = KS_distance(emp_dist, output_dist)
        result['total deviation'] = total_deviation(emp_dist, output_dist)
        result['wasserstein'] = wasserstein_1_distance(emp_dist, output_dist)

        return result
    return simulate_run_alg_description




# # Repetition of simulator

def repeat_simulate_run(replications, simulate_run_function, print_progress = False, **kwargs):
    results = None
            
    if print_progress:
        print("Simulation: ", end = "")
    start_time = time.time()
    for sim_num in range(replications):
        if print_progress:
            print(f'{sim_num+1}', end=" ")
        if results is None:
            results = pd.DataFrame(simulate_run_function(**kwargs), index = [0])
        else:
            results.loc[len(results)] = simulate_run_function(**kwargs)                          
    summary = {"runtime": (time.time()-start_time)/replications}
    for column, column_type in zip(results.columns, results.dtypes):
        if pd.api.types.is_float_dtype(column_type):
            summary['mean '+column] = results[column].mean()
            summary['std '+column] = results[column].std()
        else:
            summary[column] = results[column][0]
    return summary


def accuracy_experiment(replications, emp_hist, simulate_run_functions, total_epsilon_values = 10**np.linspace(-1, np.log10(5), 11), **kwargs):
    results = None
    fraction_epsilon_for_laplace = rule_of_thumb(total_epsilon_values) 
    laplace_epsilon_values = total_epsilon_values*fraction_epsilon_for_laplace
    counting_epsilon_values = total_epsilon_values - laplace_epsilon_values
    
    for laplace_epsilon, counting_epsilon in zip(laplace_epsilon_values, counting_epsilon_values):

        print(f'Laplace Epsilon {laplace_epsilon} Counting Epsilon {counting_epsilon}:', end = " ")
        for simulate_run_function in simulate_run_functions:
            sim_results = repeat_simulate_run(replications = replications,simulate_run_function = simulate_run_function, emp_hist = emp_hist, laplace_epsilon = laplace_epsilon, counting_epsilon = counting_epsilon, **kwargs)
            #sim_results['simulate_run_function_name'] = simulate_run_function.__name__
            sim_results['laplace_epsilon'] = laplace_epsilon
            sim_results['counting_epsilon'] = counting_epsilon
            sim_results['total_epsilon'] = counting_epsilon + laplace_epsilon
            if results is None:
                results = pd.DataFrame(sim_results, index = [0])
            else:
                results.loc[len(results)] = sim_results
        print('')
    return results



