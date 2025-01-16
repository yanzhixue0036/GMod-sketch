import math
import mmh3
import random
from bitarray import bitarray
from scipy import stats
import numpy as np

class BloomFilter:

    def __init__(self, dict_dataset, m, epsilon, seed):
        self.dict_dataset = dict_dataset
        self.m = m
        self.p = 1 / (1 + math.exp(epsilon))
        self.seed = seed
        self.epsilon = epsilon

    def build_filter(self):
        self.dict_bloomfilter = dict()

        for user in self.dict_dataset:
            bloom_filter = bitarray(self.m)
            bloom_filter.setall(0)
            for item in self.dict_dataset[user]:
                item_trans = mmh3.hash(str(item), signed=False, seed=self.seed)
                index = item_trans % self.m
                bloom_filter[index] = 1

            flipped_bf = self.flip(bloom_filter, self.p)

            self.dict_bloomfilter[user] = flipped_bf

    def flip(self, bf, p):
        for i in range(self.m):
            temp = stats.bernoulli.rvs(p, random_state=None)
            bf[i] ^= temp

        return bf

    
    
    def estimate_cardinality(self, AorB):
        lst_user = list(self.dict_dataset.keys())
        user_A = lst_user[0]
        user_B = lst_user[1]

        if AorB: bf1 = self.dict_bloomfilter[user_A]
        else :bf1 = self.dict_bloomfilter[user_B]

        bf1_m0 = 0
        bf1_m1 = 0
        L = len(bf1)

        for i in range(L):
            if bf1[i] == 1:
                bf1_m1 += 1
            else:
                bf1_m0 += 1

        cardi = -L * math.log(((1 - self.p) * bf1_m0 - self.p * bf1_m1) / (L * (1 - 2 * self.p)))
        # print(cardi)
        return cardi

    def estimate_union(self):
        random.seed(self.seed)
        lst_user = list(self.dict_dataset.keys())
        user_A = lst_user[0]
        user_B = lst_user[1]
        

        bf1 = self.dict_bloomfilter[user_A]
        bf2 = self.dict_bloomfilter[user_B]
        m00 = 0
        m01 = 0
        m10 = 0
        m11 = 0
        L = len(bf1)
        q = 1 - self.p
        for i in range(L):
            if bf1[i] == 0 and bf2[i] == 0:
                m00 += 1
            elif bf1[i] == 0 and bf2[i] == 1:
                m01 += 1
            elif bf1[i] == 1 and bf2[i] == 0:
                m10 += 1
            elif bf1[i] == 1 and bf2[i] == 1:
                m11 += 1
        # print(m00, m01, m10, m11)
        # print((math.pow(q, 2) * m00 - self.p * q * m01 - self.p * q * m10 + math.pow(self.p, 2) * m11) / (L * math.pow((q - self.p), 2)))
        estimate_u0 = -L * math.log(
            (math.pow(q, 2) * m00 - self.p * q * m01 - self.p * q * m10 + math.pow(self.p, 2) * m11) / (L * math.pow((q - self.p), 2)))
        # print(estimate_u0)
        

        return estimate_u0

    def add_lap_noise(self, data):
        # lap_noise = np.random.laplace(0, 1, len(data))
        lap_noise = np.random.laplace(0, self.epsilon, 1)
        return lap_noise + data
    
    def estimate_inter(self):
        lst_user = list(self.dict_dataset.keys())
        user_A = lst_user[0]
        user_B = lst_user[1]
        data_A = len(self.dict_dataset[user_A])
        data_B = len(self.dict_dataset[user_B])
        
        A = self.add_lap_noise(data_A)
        B = self.add_lap_noise(data_B)
        estimate_intersection = A + B - self.estimate_union()
        
        return estimate_intersection