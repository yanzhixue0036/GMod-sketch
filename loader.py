import logging
import random
import time
import copy
import csv
import json

# # from numba import jit
# # from jax import jit as jax_jit

class Dataloader:
    '''
        :param dataset: dataset path or synthetic dataset
        :param intersection: set intersection cardinality
        :param difference: set difference cardinality
        :param ratio: used to control cardinalities of two sets
        :param seed: random seed used to generate items in each set

        :func generate_synthetic_dataset(): generate synthetic dataset
        :func load_public_dataset(): load public-available dataset

        :output: dict_dataset: dataset represented as a map user->list of items
    '''

    def __init__(self, dataset, intersection, difference, ratio, seed, delete_ratio=0, maxrepeattimes=1, dict_dataset = None,path=''):
        self.dataset = dataset
        self.intersection = intersection
        self.difference = difference
        self.ratio = ratio
        self.seed = seed
        self.maxrepeattimes = maxrepeattimes
        self.path = path
        self.dict_dateset = dict_dataset 
        self.delete_ratio = delete_ratio
    def generate_synthetic_dataset(self):
        
        dict_dataset = dict()
        lst_union = list()
        lst_A = list()
        lst_B = list()
        random.seed(self.seed)

        # lst_union = [i for i in range(self.intersection + self.difference)]
        # fast generate synthetic datasets, but not recommended
        random_num = random.randint(0, 2 ** 10 -1)
        random_step = random.randint(1, 10)
        while len(lst_union) < self.intersection + self.difference:
            lst_union.append(random_num)
            random_num += random_step
        # randomly generate synthetic datasets, recommended
        # while len(lst_union) < self.intersection + self.difference:
        #     random_num = random.randint(0, 2 ** 10 - 1)
        #     if random_num not in lst_union:
        #         lst_union.append(random_num)
        lst_A.extend(lst_union[0:self.intersection])
        lst_A.extend(lst_union[self.intersection:self.intersection + int(self.difference * self.ratio)])
        lst_B.extend(lst_union[0:self.intersection])
        lst_B.extend(lst_union[self.intersection + int(self.difference * self.ratio):])
        dict_dataset['A'] = {}
        dict_dataset['B'] = {}
        dict_A = dict_dataset['A']
        dict_B = dict_dataset['B']
        dict_A['elements'] = lst_A
        dict_B['elements'] = lst_B
        maxrepeattimes = self.maxrepeattimes
        dict_A['repeattimes'] = [0]*len(lst_A)
        dict_A['index'] = [0]*len(lst_A)
        
        dict_B['repeattimes'] = [0]*len(lst_B)
        dict_B['index'] = [0]*len(lst_B)
        
        for i in range(len(dict_A['elements'])):
            dict_A['repeattimes'][i] = random.randint(1, maxrepeattimes)
            if not i == len(lst_A)-1:
                dict_A['index'][i+1] = dict_A['index'][i] + dict_A['repeattimes'][i]
            else:
                dict_B['index'][0] = dict_A['index'][-1] + dict_A['repeattimes'][i]
            
        # print(self.intersection, len(set(lst_A) & set(lst_B)))
        for i in range(len(dict_B['elements'])):
            dict_B['repeattimes'][i] = random.randint(1, maxrepeattimes)
            if not i == len(lst_B)-1:
                dict_B['index'][i+1] = dict_B['index'][i] + dict_B['repeattimes'][i]
        
        return dict_dataset

    def generate_single_dataset(self):
        single_set = list()

        while len(single_set) < self.intersection + self.difference:
            random_num = random.randint(0, 2 ** 32 - 1)
            if random_num not in single_set:
                single_set.append(random_num)

        # random_num = random.randint(0, 2 ** 32 - 1)
        # random_step = random.randint(1, 10)
        # while len(single_set) < self.intersection + self.difference:
        #     single_set.append(random_num)
        #     random_num += random_step

        # single_set = [random.randint(0, 2**32) for _ in range(int(self.intersection))]

        return single_set

    def load_public_dataset(self):
        dict_dataset = dict()

        with open(self.dataset) as freader:
            for line in freader:
                [user, item] = list(map(int, line.strip().split()))
                if user not in dict_dataset:
                    dict_dataset[user] = []
                if item not in dict_dataset[user]:
                    dict_dataset[user].append(item)

        return dict_dataset
    
    import csv

    def sort_csv_by_first_column(input_filename, output_filename):
        with open(input_filename, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            data = list(reader)

        sorted_data = sorted(data, key=lambda x: x[0])

        with open(output_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(sorted_data)

        sort_csv_by_first_column('input.csv', 'sorted_output.csv')


    def load_real_dataset(self):
        
        dict_dataset = dict()
        lst_union = list()
        lst_A = list()
        lst_B = list()
        
        csvfile = open(self.path, 'r', newline='')
        csv_reader = csv.reader(csvfile)
        repeattimes = []
        # print("........")
        row = next(csv_reader)
        last_item = None
        while len(lst_union) < self.intersection + self.difference:
            try:
                row = next(csv_reader)
            except Exception as e:
                print("Error:", e)
            item = row[0]
            if last_item == item: 
                repeattimes[-1] += 1
                pass
            # elif item in lst_union:
            #     print(item)
            #     index = lst_union.index(item)
            #     repeattimes[index] += 1
            else:
                last_item = item
                lst_union.append(item)
                repeattimes.append(1)
                
        lst_A.extend(lst_union[0:self.intersection])
        lst_A.extend(lst_union[self.intersection:self.intersection + int(self.difference * self.ratio)])
        lst_B.extend(lst_union[0:self.intersection])
        lst_B.extend(lst_union[self.intersection + int(self.difference * self.ratio):])
        # print(len(lst_A), self.intersection + int(self.difference * self.ratio))
        # print(len(lst_B), self.intersection + int(self.difference * (1-self.ratio)))
        # print(self.intersection, len(set(lst_A) & set(lst_B)))
        
        dict_dataset['A'] = {}
        dict_dataset['B'] = {}
        dict_A = dict_dataset['A']
        dict_B = dict_dataset['B']
        dict_A['repeattimes'] = []
        dict_B['repeattimes'] = []
        
        dict_A['repeattimes'].extend(repeattimes[0:self.intersection])
        dict_A['repeattimes'].extend(repeattimes[self.intersection:self.intersection + int(self.difference * self.ratio)])
        dict_B['repeattimes'].extend(repeattimes[0:self.intersection])
        dict_B['repeattimes'].extend(repeattimes[self.intersection + int(self.difference * self.ratio):])

        
        dict_A['elements'] = lst_A
        dict_B['elements'] = lst_B
        
        dict_A['index'] = [0]*len(lst_A)
        dict_B['index'] = [0]*len(lst_B)
        
        for i in range(len(dict_A['elements'])):
            if not i == len(lst_A)-1:
                dict_A['index'][i+1] = dict_A['index'][i] + dict_A['repeattimes'][i]
            else:
                dict_B['index'][0] = dict_A['index'][-1] + dict_A['repeattimes'][i]
            
        
        for i in range(len(dict_B['elements'])):
            if not i == len(lst_B)-1:
                dict_B['index'][i+1] = dict_B['index'][i] + dict_B['repeattimes'][i]

        # 关闭文件
        csvfile.close()
        # print("finish...........................................")
        return dict_dataset
        

    def load_dataset(self, dataset):
        if dataset == 'synthetic':
            dict_dataset = self.generate_synthetic_dataset()
            self.dict_dateset = dict_dataset
            with open("./tmp/output2.json", "w") as json_file:
                json.dump(dict_dataset, json_file, indent=4)
            # dict_dataset = self.generate_single_dataset()
        elif dataset == 'deleted':
            dict_dataset = self.generate_deleted_dataset(self.dict_dateset,self.delete_ratio)
        else:
            dict_dataset = self.load_real_dataset()

        return dict_dataset
    
    def generate_deleted_dataset(self, referencedataset, delete_ratio):
        dict_dataset = copy.deepcopy(referencedataset)
        
        deleted_c = int(self.intersection * delete_ratio)
        deleted_d = int(self.difference * delete_ratio)
        deleted_d_A = int(self.difference * self.ratio * delete_ratio)
        deleted_d_B = deleted_d - deleted_d_A
        
        dict_A = dict_dataset['A']
        for item in dict_A:
            dict_A[item] = dict_A[item][0:deleted_c] + dict_A[item][self.intersection: self.intersection+deleted_d_A]
        
        dict_B = dict_dataset['B']
        for item in dict_B:
            dict_B[item] = dict_B[item][0:deleted_c] + dict_B[item][self.intersection: self.intersection+deleted_d_B]
        with open("./tmp/output.json", "w") as json_file:
            json.dump(dict_dataset, json_file, indent=4)
        return dict_dataset