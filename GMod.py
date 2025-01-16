import os
import copy
import math
import mmh3
import pickle
import random
import numpy as np

from utils import *
from math import exp, pow, log


class GMod:
    def __init__(self, dict_dataset, m, w, g, output,seed, epsilon=1, random_response=False, delete_dataset=None):
        self.dict_dataset = dict_dataset
        self.m_size = m
        self.w_size = w
        self.g = g
        self.G = 2**self.g
        self.p = [1 / pow(2.0, j + 1) if not j == self.w_size-1 else 1 / pow(2.0, j) for j in range(self.w_size)]
        self.output = output
        self.seed = seed
        self.random_response = random_response
        self.alpha = min((exp(epsilon)-1) / (self.G - 1), (exp(epsilon)-1) / exp(epsilon)) if random_response else 1
        self.N = 2 if random_response else 0
        if delete_dataset == None:
            self.delete_dataset = {}
            for user in self.dict_dataset:
                self.delete_dataset[user] = {}
                self.delete_dataset[user]['elements']=[]
        else:
            self.delete_dataset = delete_dataset       
        #self.repeatTimes = 1 if delete_dataset==None else 10
        self.repeatTimes = 1
        self.key = dict()
            
    def __arriveElement(self, item, row, repeattimes, GMod_sketch, add = 1):
        for p in range(repeattimes):
            temp_hash = mmh3.hash64(str(item), signed=False, seed=self.seed + 1)
            index_j = self.compute_index_value(temp_hash[0])
            
            # i ← h(e)
            temp_i = bin(temp_hash[1])[2:]
            index_i = int(temp_i, 2) % self.m_size
            
            temp_x_pair = [item, row]   # 修改了这行代码
            temp_x = mmh3.hash(str(temp_x_pair), signed=False, seed=self.seed + 3)
            temp_x = bin(temp_x)[2:]
            x = int(temp_x, 2) % self.G
            
            GMod_sketch[index_i][index_j] += add * x
            GMod_sketch[index_i][index_j] %= self.G
            #if(add == 1): row[0] += 1

    def build_sketch(self):
        self.dict_GMod_sketch = dict()
        row = [0]
        #print(self.dict_dataset)
        for user in self.dict_dataset:
            
            GMod_sketch = [[0] * self.w_size for _ in range(self.m_size)]
            # arrive
            user_dict = self.dict_dataset[user]
            repeattimes = self.repeatTimes
            for i in range(len(user_dict['elements'])):
                item = user_dict['elements'][i]       
                #repeattimes = user_dict['repeattimes'][i]
                row[0] = user_dict['index'][i]    
                self.__arriveElement(item, row, repeattimes, GMod_sketch, add=1)
            
            # delete
            if self.delete_dataset is not None and user in self.delete_dataset:
                delete_dict = self.delete_dataset[user]
                for i in range(len(delete_dict['elements'])):

                    item = delete_dict['elements'][i]
                    #repeattimes = user_dict['repeattimes'][i]  
                    row[0] = delete_dict['index'][i]        
                    self.__arriveElement(item, row, repeattimes, GMod_sketch, add=-1)

            # pertube
            if self.random_response:
                GMod_sketch = self.perturbsketch(GMod_sketch)

            self.dict_GMod_sketch[user] = GMod_sketch

    def poisson(self, u):
        x = 0
        p = exp(-(self.m_size * self.lamb))
        s = p
        for i in range(100):
            if u <= s:
                break
            x += 1
            p = p * (self.m_size * self.lamb) / float(x)
            s += p
        return x

    def compute_index_value(self, binary_item):
        binary_item = bin(binary_item)[2:]

        index_j = 0
        # trailing zeros
        rvs_binary_item = binary_item[::-1]
        for bit in rvs_binary_item:
            if bit == '0':
                index_j += 1
            else:
                break
        return min(index_j, self.w_size-1)
    
    def compute_index_value_integrate(self, binary_item):
        binary_item = bin(binary_item)[2:]
        item = int(binary_item, 2) * self.p[-1]
        for j in range(self.w_size):
            if(item > self.p[j]):
                return min(j, self.w_size-1)
            
    def perturbsketch(self, GMod_sketch,):
        
        for j in range(self.w_size):
            for i in range(self.m_size):
                temp = random.random()
                if temp <= 1 - self.alpha:
                    pertur_y = random.randint(0, self.G-1)
                else:
                    pertur_y = 0

                GMod_sketch[i][j] += pertur_y
                GMod_sketch[i][j] %= self.G

        return GMod_sketch
    
    def phi_func(self, n, w, alpha):
        z = 0.0
        for j in range(w):
            z += (1-1/self.G) * pow(1 - self.p[j]/ self.m_size, n) * pow(self.alpha, self.N)
            z += 1 / self.G
        return z

    def phi_func_derived(self, n, w, alpha):
        z = 0.0
        for j in range(w):
            z += (1-1/self.G) * math.log(1 - self.p[j]/self.m_size) \
                * pow(1 - self.p[j]/self.m_size, n) * pow(self.alpha, self.N)
        return z

    def newton_iteration_estimator(self, merge_sketch, m, w, alpha):
        n = 1
        error = 1e-1
        
        if v == None:
            v = 0.0 
            for j in range(w):
                count = 0
                for i in range(self.m_size):
                    count += (merge_sketch[i][j] == 0)
                v += count / self.m_size

        while (self.phi_func(n, w, alpha) - v) * (self.phi_func(n+1, w, alpha) - v) > error:
            n = n - (self.phi_func(n, w, alpha) - v) / self.phi_func_derived(n, w, alpha)
            
        return n

    def binary_search_estimator(self, merge_sketch, m, w, alpha):
        
        v = 0.0 
        for j in range(w):
            count = 0
            for i in range(self.m_size):
                count += (merge_sketch[i][j] == 0)
            v += count / self.m_size
        
        low = 1 
        for i in range(10):
            if((self.phi_func(pow(10,i), w, alpha) - v) * (self.phi_func(pow(10,i+1), w, alpha) - v) < 0):
                low = pow(10,i)

        high=low*10
        while low <= high:
            mid = (low + high) // 2
            if (self.phi_func(mid, w, alpha) - v) * (self.phi_func(mid+1, w, alpha) - v) < 0:
                return mid 
            elif (self.phi_func(mid, w, alpha) - v) * (self.phi_func(high, w, alpha) - v) < 0:
                low = mid + 1
            else:
                high = mid - 1
            n=mid
        return n
    def estimate_n(self, merge_sketch, m, w, alpha):
        
        v = 0.0 
        for j in range(w):
            count = 0
            for i in range(self.m_size):
                count += (merge_sketch[i][j] == 0)
            v += count / self.m_size
        V = v / self.w_size
        
        G = self.G
        w = self.w_size
        m = self.m_size
        gamma = 0.5772156649
        
        a, b = math.sqrt(2), 0.5 
        f = lambda n: (
            1 / G + (pow(self.alpha, self.N) / (w * np.log(2))) * (1 - 1 / G) * (np.log(a * 2**(w - b)) - a * n / (2 * m))
            if n < 2 * m else
            1 / G - (pow(self.alpha, self.N) / (w * np.log(2))) * (1 - 1 / G) * (np.log((n * 2**-(w - b)) / (2 * m)) + gamma)
        )
        
        if V > f(2 * m):
            # return 2 * np.log(2) * m * w * (1 - V) * (G / (G - 1))
            n =  2 * m / a * ( np.log(a * 2**(w - b)) - (V - 1/G) * w * np.log(2) / (1 - 1 / G) / pow(self.alpha, self.N))
        else:
            n =  m * 2**(w - b + 1) * np.exp( -gamma - (w * np.log(2) / pow(self.alpha, self.N) / (1 - 1 / G)) * (V - 1 / G))
        
        return max(n, 0)
        
        

    def MEC_estimate(self, merge_sketch, m0, w, alpha, esti):
        
        n = [0] * w
        var = [0] * w
        n_f = 0
        denomi = 0
        for j in range(w):
            G = self.G
            m = self.m_size
            
            z = 0
            for i in range(self.m_size):
                z += (merge_sketch[i][j] == 0)
            z /= m
            
            
            if z > 1/G:
                pn = pow((1-self.p[j]/m), esti)
                n[j] = log((G*z-1)/(G-1)/pow(self.alpha, self.N)) / log(1-self.p[j]/m)
                deno = m * (G-1) * pow(self.alpha, 2*self.N) * pow(log(1-self.p[j]/m), 2) * pow(pn, 2)
                if deno == 0: var[j] = float('inf')
                else:
                    var_V = 1 + (G-2) * pow(self.alpha, self.N) * pn - (G-1) * pow(pn, 2) * pow(self.alpha, 2*self.N)
                    var[j] = max(var_V / deno, pow(2,-10))
                    
                denomi += 1.0 / var[j]
            else:
                n[j] = -1.0
                var[j] = -1.0

        for j in range(w):
            # print(m, j, n[j], var[j])
            if n[j] >= 0:
                n_f += (n[j] / var[j]) / denomi

        n = n_f
        
        return n
    
    def estimate_cardinality(self):
        random.seed(self.seed)

        lst_user = list(self.dict_dataset.keys())
        num_user = len(lst_user)
        random.shuffle(lst_user)

        lst_result = list()
        for u in range(num_user - 1):
            user_A = lst_user[u]
            user_B = lst_user[u + 1]

            # lst_A = self.dict_dataset[user_A]
            # lst_B = self.dict_dataset[user_B]

            GMod_sketch_A = self.dict_GMod_sketch['A']
            GMod_sketch_B = self.dict_GMod_sketch['B']

            # estimated_union = self.binary_search_estimator(GMod_sketch_A, self.m_size, self.w_size, self.alpha)
            estimated_union = self.estimate_n(GMod_sketch_A, self.m_size, self.w_size, self.alpha)
            # print("actual_n:{} | binary_estimated_n:{} | estimated_n:{}".format(self.intersection + self.difference, estimated_union, estimated_n))
            
            lst_result.append(estimated_union)
            # actual_union, actual_intersection = compute_difference(lst_A, lst_B)

        # foutput = open(os.path.join(self.output, 'setxor.out'), 'wb')
        # pickle.dump(lst_result, foutput)
        # foutput.close()

        return lst_result
    
    def estimate_cardinality_MEC(self):
        random.seed(self.seed)

        lst_user = list(self.dict_dataset.keys())
        num_user = len(lst_user)
        random.shuffle(lst_user)

        lst_result = list()
        for u in range(num_user - 1):
            user_A = lst_user[u]
            user_B = lst_user[u + 1]

            # lst_A = self.dict_dataset[user_A]
            # lst_B = self.dict_dataset[user_B]

            GMod_sketch_A = self.dict_GMod_sketch['A']
            GMod_sketch_B = self.dict_GMod_sketch['B']
            
            # estimated_union = self.binary_search_estimator(GMod_sketch_A, self.m_size, self.w_size, self.alpha)
            estimated_union = self.estimate_n(GMod_sketch_A, self.m_size, self.w_size, self.alpha)

            estimated_union_MEC = self.MEC_estimate(GMod_sketch_A, self.m_size, self.w_size, self.alpha,
                                                    estimated_union)
            
                
            lst_result.append(estimated_union_MEC)

        return lst_result
    
    def add_lap_noise(self, data):
        lap_noise = np.random.laplace(0, 1, len(data))
        return lap_noise + data
    def estimate_union(self):
        random.seed(self.seed)

        lst_user = list(self.dict_dataset.keys())
        num_user = len(lst_user)
        random.shuffle(lst_user)

        lst_result = list()
        for u in range(num_user - 1):
            user_A = lst_user[u]
            user_B = lst_user[u + 1]

            # lst_A = self.dict_dataset[user_A]
            # lst_B = self.dict_dataset[user_B]

            GMod_sketch_A = self.dict_GMod_sketch[user_A]
            GMod_sketch_B = self.dict_GMod_sketch[user_B]

            GMod_sketch_merge = [[0] * self.w_size for _ in range(self.m_size)]
            for i in range(self.m_size):
                for j in range(self.w_size):
                    GMod_sketch_merge[i][j] = (GMod_sketch_A[i][j] + GMod_sketch_B[i][j]) % self.G
                    
            estimated_union = self.estimate_n(GMod_sketch_merge, self.m_size, self.w_size, self.alpha)
            lst_result.append(estimated_union)
            # print("actual_intersection:{} | estimated_intersection:{}".format(actual_intersection, estimated_intersection))

        return lst_result
    
    def estimate_union_MEC(self):
        random.seed(self.seed)

        lst_user = list(self.dict_dataset.keys())
        num_user = len(lst_user)
        random.shuffle(lst_user)

        lst_result = list()
        for u in range(num_user - 1):
            user_A = lst_user[u]
            user_B = lst_user[u + 1]

            # lst_A = self.dict_dataset[user_A]
            # lst_B = self.dict_dataset[user_B]

            GMod_sketch_A = self.dict_GMod_sketch[user_A]
            GMod_sketch_B = self.dict_GMod_sketch[user_B]

            GMod_sketch_merge = [[0] * self.w_size for _ in range(self.m_size)]
            for i in range(self.m_size):
                for j in range(self.w_size):
                    GMod_sketch_merge[i][j] = (GMod_sketch_A[i][j] + GMod_sketch_B[i][j]) % self.G
            
            # estimated_union = self.binary_search_estimator(GMod_sketch_merge, self.m_size, self.w_size, self.alpha)
            estimated_union = self.estimate_n(GMod_sketch_merge, self.m_size, self.w_size, self.alpha)
            estimated_union_MEC = self.MEC_estimate(GMod_sketch_merge, self.m_size, self.w_size, self.alpha, estimated_union)
                
            lst_result.append(estimated_union_MEC)
        return lst_result