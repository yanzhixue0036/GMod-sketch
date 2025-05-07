from math import exp, pow, log, sqrt
import math

def compute_difference(lst_A, lst_B):
    """
    :param lst_A: raw set A
    :param lst_B: raw set B

    :output the set difference cardinality between set A and set B
    """
    cardinality_union = len(list(set(lst_A).union(set(lst_B))))
    cardinality_intersection = len(list(set(lst_A).intersection(set(lst_B))))

    return cardinality_union, cardinality_intersection


def compute_rse(lst):
    rse = 0
    rse_IVW = 0
    true_cardinality = lst[0][0]
    num_exp = len(lst)

    for exp in lst:
        rse += (exp[1] - true_cardinality) ** 2
        rse_IVW += (exp[2] - true_cardinality) ** 2

    rse = (rse / num_exp) ** 0.5
    rse /= true_cardinality

    rse_IVW = (rse_IVW / num_exp) ** 0.5
    rse_IVW /= true_cardinality

    return rse, rse_IVW


def compute_aare(lst, true_cardi):
    if lst == []: return 1.0
    
    aare = 0
    num_exp = 0

    for exp in lst:
        # if exp > 1.0:
        #     aare += abs(exp-true_cardi) / true_cardi
        #     num_exp += 1
        
        # if exp > 0:
        #     aare += abs(exp-true_cardi) / true_cardi
        # else:
        #     aare += abs(0-true_cardi) / true_cardi
        
        aare += abs(exp-true_cardi) / true_cardi
        num_exp += 1
    if num_exp == 0: return 1.0
    
    aare /= num_exp

    return aare


def compute_rrmse(results, true_cardi):
    sum = 0
    neg_count = 0
    if results == []: return 1.0
    for i in range(len(results)):
        # if results[i] < 0:
        #     neg_count += 1
        #     continue
        sum += pow((results[i] - true_cardi), 2)
    temp = sqrt(sum / (len(results) - neg_count))
    rrmse = temp / true_cardi
    return rrmse


# def compute_stderr(n_hat, m, w, epsilon, lamb):
#     z1 = w
#     z2 = 0.8
#     alpha = 0.26
#     for i in range(w - 1):
#         z1 -= pow((1.0 - 2.0 * alpha), 4) * exp(-4.0 * n_hat * lamb / pow(2, i + 1))
#     z1 -= pow((1.0 - 2.0 * alpha), 4) * exp(-4.0 * n_hat * lamb / pow(2, w - 1))
#
#     for j in range(w - 1):
#         z2 += 2.0 * n_hat * lamb * pow((1.0 - 2.0 * alpha), 2) * exp(-2.0 * n_hat * lamb / pow(2, j + 1)) / pow(2, j + 1)
#     z2 += 2.0 * n_hat * lamb * pow((1.0 - 2.0 * alpha), 2) * exp(-2.0 * n_hat * lamb / pow(2, w - 1)) / pow(2, w - 1)
#
#     error = 1 / sqrt(m) * sqrt(z1) / z2
#
#     return error


# def compute_star_stderr(n, m, w, epsilon, lamb):
#     var = [0] * 32
#     denomi = 0
#     # alpha = exp(epsilon) / (1 + exp(epsilon))
#     alpha = 0.8
#     for j in range(w):
#         if j < w - 1:
#             p = 1.0 / pow(2.0, j + 1)
#         else:
#             p = 1.0 / pow(2.0, j)
#         var[j] = (1.0 - pow((1.0 - 2 * alpha), 4) * exp(-4 * n * lamb * p)) / (
#                 4.0 * m * pow((lamb * p), 2) * pow((1.0 - 2.0 * alpha), 4) * exp(-4 * n * lamb * p))
#         denomi += 1.0 / var[j]
#     var = 0.25 * sqrt(1 / denomi) + 1 / pow(epsilon, 2)
#     stderr = sqrt(var) / n
#     return stderr

def sfm_stderr(n, m, w, epsilon):
    p = exp(epsilon) / (1 + exp(epsilon))
    q = 1 - p
    sum = 0
    for j in range(w):
        rho = 1 / (pow(2, min(j, w - 1)) * m)
        gamma = 1 - rho
        sum += pow(log(gamma), 2) * pow(gamma, n) * (p / (p - (p-q)*pow(gamma, n)) - (1-p) / (1-p+(p-q)*pow(gamma, n)))
    temp = m * (p-q) * sum
    error = pow(temp, -0.5)
    return error / n



if __name__ == '__main__':
    n_hat = [1e3, 1e4, 1e5, 1e6, 1e7, 1e8]
    m = 1024
    w = 32
    epsilon = 2
    # for i in n_hat:
    #     stderr = sfm_stderr(i, m, w, epsilon)
    #     print(stderr)

    E = half_phi_fun(1000, 32, 1/1024)
    print(E * (1024*32))
