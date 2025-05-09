import os
import sys
import mmh3
import math
import random
import pickle
import numpy as np
from scipy import stats
from bitarray import bitarray
from math import exp, log, pow

# # # # from numba import jit
# # from jax import jit as jax_jit

sys.path.append('..')
from utils import *
from loader import Dataloader

class PS:
    def __init__(self, dict_dataset, m, w, epsilon, merge_method, seed):
        self.dict_dataset = dict_dataset
        self.m = m
        self.w = w
        assert merge_method == 'loose' or merge_method == 'tight'
        self.epsilon = log(epsilon/2+1) if merge_method == 'loose' else log(0.5*(exp(epsilon)+1))
        self.merge_method = merge_method
        self.seed = seed

    def build_fm_sketch(self):
        self.dict_fm = dict()

        for user in self.dict_dataset:
            fm_sketch = [[0] * self.w for _ in range(self.m)]
            for item in self.dict_dataset[user]:
                item_trans = mmh3.hash(str(item), signed=False, seed=self.seed)
                alpha, index = self.compute_index_value(item_trans)
                if fm_sketch[alpha][index] == 0:
                    fm_sketch[alpha][index] = 1
            # if self.merge_method == 'loose': # xor
            #     flipped_fm = self.asym_flip(fm_sketch)
            # else:
            #     flipped_fm = self.sym_flip(fm_sketch)
            flipped_fm = self.asym_flip(fm_sketch)
            self.dict_fm[user] = flipped_fm

    def compute_index_value(self, item):
        bucket_index = item % self.m
        temp = math.floor(item / self.m)
        binary_temp = '00' + bin(temp)[2:]
        revers_temp = binary_temp[::-1]
        position = revers_temp.find('1')
        return bucket_index, position

    def asym_flip(self, fm_sketch):
        p = 0.5
        q = 1 / (2 * exp(self.epsilon))
        for i in range(self.m):
            for j in range(self.w):
                temp = random.random()
                if fm_sketch[i][j] == 1:
                    if temp <= p:
                        fm_sketch[i][j] = 1
                    else:
                        fm_sketch[i][j] = 0
                if fm_sketch[i][j] == 0:
                    if temp <= q:
                        fm_sketch[i][j] = 1
                    else:
                        fm_sketch[i][j] = 0

        return fm_sketch

    def sym_flip(self, fm_sketch):
        p = exp(self.epsilon) / (exp(self.epsilon) + 1)
        q = 1 - p
        for i in range(self.m):
            for j in range(self.w):
                temp = random.random()
                if fm_sketch[i][j] == 1:
                    if temp <= p:
                        fm_sketch[i][j] = 1
                    else:
                        fm_sketch[i][j] = 0
                elif fm_sketch[i][j] == 0:
                    if temp <= q:
                        fm_sketch[i][j] = 1
                    else:
                        fm_sketch[i][j] = 0

        return fm_sketch

    def deterministic_merge(self, sketch_A, sketch_B):
        sketch_merge = [[0] * self.w for _ in range(self.m)]

        for i in range(self.m):
            for j in range(self.w):
                sketch_merge[i][j] = sketch_A[i][j] ^ sketch_B[i][j]
        return sketch_merge

    def random_merge(self, sketch_A, sketch_B):

        sketch_merge = [[0] * self.w for _ in range(self.m)]

        epsilon1 = self.epsilon
        epsilon2 = self.epsilon
        epsilon_star = -log(exp(-epsilon1) + exp(-epsilon2) - exp(-(epsilon1 + epsilon2)))
        q1 = 1 / (1 + exp(epsilon1))
        q2 = 1 / (1 + exp(epsilon2))
        q_star = 1 / (1 + exp(epsilon_star))

        k1 = np.array([[1 - q1, q1], [q1, 1 - q1]])
        k2 = np.array([[1 - q2, q2], [q2, 1 - q2]])
        v = np.array([q_star, 1 - q_star, 1 - q_star, 1 - q_star])
        k1_inv = np.linalg.inv(k1)
        k2_inv = np.linalg.inv(k2)

        trans_matrix = np.matmul(np.kron(k1_inv, k2_inv), v)

        for i in range(self.m):
            for j in range(self.w):
                if sketch_A[i][j] == 0 and sketch_B[i][j] == 0:
                    temp = stats.bernoulli.rvs(trans_matrix[0], random_state=None)
                    sketch_merge[i][j] = temp
                elif sketch_A[i][j] == 0 and sketch_B[i][j] == 1:
                    temp = stats.bernoulli.rvs(trans_matrix[1], random_state=None)
                    sketch_merge[i][j] = temp
                elif sketch_A[i][j] == 1 and sketch_B[i][j] == 0:
                    temp = stats.bernoulli.rvs(trans_matrix[2], random_state=None)
                    sketch_merge[i][j] = temp
                elif sketch_A[i][j] == 1 and sketch_B[i][j] == 1:
                    temp = stats.bernoulli.rvs(trans_matrix[3], random_state=None)
                    sketch_merge[i][j] = temp

        return sketch_merge

    def estimation_union_cardinality(self):
        random.seed(self.seed)
        lst_user = list(self.dict_dataset.keys())
        num_user = len(lst_user)

        for i in range(num_user - 1):
            user_A = lst_user[i]
            user_B = lst_user[i+1]

            lst_A = self.dict_dataset[user_A]
            lst_B = self.dict_dataset[user_B]

            sfm_sketch_A = self.dict_fm[user_A]
            sfm_sketch_B = self.dict_fm[user_B]

            # if self.merge_method == 'loose':
            #     merged_sketch = self.deterministic_merge(sfm_sketch_A, sfm_sketch_B)
            # else:
            #     merged_sketch = self.random_merge(sfm_sketch_A, sfm_sketch_B)
            merged_sketch =  self.deterministic_merge(sfm_sketch_A, sfm_sketch_B)
            
            cardinality_union = len(list(set(lst_A).union(set(lst_B))))
            # print("The ground truth union cardinality: {}".format(cardinality_union))

            estimation_union = self.newton_raphson(merged_sketch)
            # print("estimation union cardinality: {}".format(estimation_union))

        return estimation_union

    def add_lap_noise(self, data):
        # lap_noise = np.random.laplace(0, 1, len(data))
        lap_noise = np.random.laplace(0, self.epsilon, 1)
        return lap_noise + data
    
    def estimation_itersection_cardinality(self):
        random.seed(self.seed)
        lst_user = list(self.dict_dataset.keys())
        num_user = len(lst_user)

        for i in range(num_user - 1):
            user_A = lst_user[i]
            user_B = lst_user[i+1]

            # lst_A = self.dict_dataset[user_A]
            # lst_B = self.dict_dataset[user_B]

            sfm_sketch_A = self.dict_fm[user_A]
            sfm_sketch_B = self.dict_fm[user_B]

            # if self.merge_method == 'loose':
            #     merged_sketch = self.deterministic_merge(sfm_sketch_A, sfm_sketch_B)
            # else:
            #     merged_sketch = self.random_merge(sfm_sketch_A, sfm_sketch_B)
            merged_sketch = self.deterministic_merge(sfm_sketch_A, sfm_sketch_B)
            
            # cardinality_union = len(list(set(lst_A).union(set(lst_B))))
            # print("The ground truth union cardinality: {}".format(cardinality_union))

            estimation_union = self.newton_raphson(merged_sketch)
            # print("estimation union cardinality: {}".format(estimation_union))
            
        data_A = len(self.dict_dataset[user_A])
        data_B = len(self.dict_dataset[user_B])
            
        A = self.add_lap_noise(data_A)
        B = self.add_lap_noise(data_B)
        return A + B - estimation_union

    def newton_raphson(self, merged_sketch):
        estiamted_union = 100
        e = 10 ** (-5)
        # if self.merge_method == 'loose':
        #     epsilon_star = -log(exp(-self.epsilon) + exp(-self.epsilon) - exp(-(self.epsilon+self.epsilon)))
        #     p = 0.5
        #     q = 1 / (2 * exp(epsilon_star))

        # else:
        #     epsilon_star = -log(exp(-self.epsilon) + exp(-self.epsilon) - exp(-(self.epsilon + self.epsilon)))
        #     p = exp(epsilon_star) / (exp(epsilon_star) + 1)
        #     q = 1 - p
        epsilon_star = -log(exp(-self.epsilon) + exp(-self.epsilon) - exp(-(self.epsilon+self.epsilon)))
        p = 0.5
        q = 1 / (2 * exp(epsilon_star))
            
        # raw_derivative = self.raw_function(merged_sketch, estiamted_union, p, q)
        first_derivative = self.first_derivative(merged_sketch, estiamted_union, p, q)
        second_derivative = self.second_derivative(merged_sketch, estiamted_union, p, q)

        while abs(first_derivative) > e:
            estiamted_union -= first_derivative / second_derivative
            first_derivative = self.first_derivative(merged_sketch, estiamted_union, p, q)
            second_derivative = self.second_derivative(merged_sketch, estiamted_union, p, q)

        return estiamted_union

    def raw_function(self, merged_sketch, n, p, q):
        sum1 = 0
        sum2 = 0
        for i in range(self.m):
            for j in range(self.w):
                rho = 1 / (pow(2, min(1+j, self.w)) * self.m)
                gamma = 1 - rho
                sum1 += (1 - merged_sketch[i][j])*log(1-p+(p-q)*pow(gamma, n))
                sum2 += merged_sketch[i][j] * log(p - (p-q)*pow(gamma, n))
        l_pq = sum1 + sum2
        return l_pq

    def first_derivative(self, merged_sketch, n, p, q):

        sum1 = 0
        sum2 = 0
        for i in range(self.m):
            for j in range(self.w):
                rho = 1 / (pow(2, min(1+j, self.w)) * self.m)
                gamma = 1 - rho
                sum1 += (1 - merged_sketch[i][j]) * (p-q)*pow(gamma, n) * log(gamma) / (1-p+(p-q)*pow(gamma, n))
                sum2 += merged_sketch[i][j] * (p-q)*pow(gamma, n)*log(gamma) / (p-(p-q)*pow(gamma, n))
        l_pq = sum1 - sum2

        return l_pq

    def second_derivative(self, merged_sketch, n, p, q):
        sum1 = 0
        sum2 = 0
        for i in range(self.m):
            for j in range(self.w):
                rho = 1 / (pow(2, min(1+j, self.w)) * self.m)
                gamma = 1 - rho
                sum1 += (1 - merged_sketch[i][j]) * (1-p) * (p-q) * pow(log(gamma), 2) * pow(gamma, n) / pow((1 - p + (p - q) * pow(gamma, n)), 2)

                sum2 += merged_sketch[i][j] * p * (p - q) * pow(log(gamma), 2) * pow(gamma, n) / pow((p - (p - q) * pow(gamma, n)), 2)
        l_pq = sum1 - sum2

        return l_pq



if __name__ == '__main__':
    epsilon = 1
    result_set = []
    c, d = 100, 10**6
    for i in range(100):
        seed = random.randint(1, 2**32-1)
        dataloader = Dataloader('synthetic', c, d, 0.5, seed)
        dict_dataset = dataloader.load_dataset()

        sfm = PS(dict_dataset, 4096, 32, epsilon, 'tight', seed) # 'loose' or 'tight'
        sfm.build_fm_sketch()
        lst_result = sfm.estimation_union_cardinality()
        
        print(lst_result)
        result_set.append(lst_result)
    print(compute_aare(result_set, c+d))
    