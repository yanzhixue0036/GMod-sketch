import mmh3
import math
#import torch
import random
import numpy as np
from scipy import stats

import os, sys
import pickle
from math import exp, pow, log

sys.path.append("./baseline")
sys.path.append("../baseline")
sys.path.append("../../baseline")
import cascading_legions 

class CL:
    def __init__(self, dict_dataset, m, l, epsilon, seed, delete_dataset=None):
        self.dict_dataset = dict_dataset
        self.m = m
        self.l = l
        self.epsilon = epsilon
        self.seed = seed
        self.p = 1 / (1 + exp(self.epsilon))
        
        self.noiser = cascading_legions.Noiser(self.p)
        self.estimator = cascading_legions.Estimator(self.p)
        
        if delete_dataset == None:
            self.delete_dataset = {}
            for user in self.dict_dataset:
                self.delete_dataset[user] = {}
                self.delete_dataset[user]['elements']=[]
        else:
            self.delete_dataset = delete_dataset
        self.repeatTimes = 1 if delete_dataset==None else 10
        
    def build_sketch(self):
        self.dict_cl = dict()

        for user in self.dict_dataset:
            cl_sketch = cascading_legions.CascadingLegions(self.l, self.m, random_seed=self.seed)
            # fm_counter = [[0] * self.l for _ in range(self.m)]
            
            # arrive
            user_dict = self.dict_dataset[user]
            for i in range(len(user_dict['elements'])):
                item = user_dict['elements'][i]
                # row = copy.deepcopy(user_dict['index'][i])
                repeattimes = user_dict['repeattimes'][i]
                
                for p in range(repeattimes):
                    # item_trans = mmh3.hash(str(item), signed=False, seed=self.seed)
                    # alpha, index = self.compute_index_value(item_trans)
                    # fm_counter[alpha][index] += 1
                    cl_sketch.add_id(f'{item}')
                        
            # delete
            user_dict = self.delete_dataset[user]
            for i in range(len(user_dict['elements'])):
                item = user_dict['elements'][i]
                # row = copy.deepcopy(user_dict['index'][i])
                repeattimes = user_dict['repeattimes'][i]
                
                for p in range(repeattimes):
                    # item_trans = mmh3.hash(str(item), signed=False, seed=self.seed)
                    # alpha, index = self.compute_index_value(item_trans)
                    # fm_counter[alpha][index] -= 1
                    cl_sketch.add_id(f'{item}')
            
            self.dict_cl[user] = cl_sketch
            
    
    def estimation_union_car(self):
        random.seed(self.seed)
        lst_user = list(self.dict_dataset.keys())
        num_user = len(lst_user)
        
        for i in range(num_user - 1):
            user_A = lst_user[i]
            user_B = lst_user[i+1]

            lst_A = self.dict_dataset[user_A]
            lst_B = self.dict_dataset[user_B]
            
            cardinality_union = len(list(set(lst_A).union(set(lst_B))))
            
            sketch_list = [self.dict_cl[user_A], self.dict_cl[user_B]]
            noised_sketch_list = list(map(self.noiser, sketch_list))
            
            estimation_union = self.estimator(noised_sketch_list)[0]
        
        return [estimation_union]
    
    def add_lap_noise(self, data):
        lap_noise = np.random.laplace(0, self.epsilon, 1)
        return lap_noise + data
    
    def estimation_intersection_cardinality(self):
        random.seed(self.seed)
        lst_user = list(self.dict_dataset.keys())
        num_user = len(lst_user)

        for i in range(num_user - 1):
            user_A = lst_user[i]
            user_B = lst_user[i+1]

            estimation_union = self.estimation_union_car()
            # print("estimation union cardinality: {}".format(estimation_union))
            
        data_A = len(self.dict_dataset['A']['elements'])
        data_B = len(self.dict_dataset['B']['elements'])
            
        A = self.add_lap_noise(data_A)
        B = self.add_lap_noise(data_B)
        res = A + B - estimation_union
        return res