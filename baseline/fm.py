import random
from math import exp, log, pow, sqrt
import mmh3
import math
import numpy as np
from bitarray import bitarray


class FM:
    def __init__(self, dict_dataset, m, w, seed):
        self.dict_dataset = dict_dataset
        self.m = m
        self.w = w
        self.seed = seed

    def build_sketch(self):
        self.dict_fm_sketch = dict()
        for user in self.dict_dataset:
            fm_sketch = [[0] * self.w for _ in range(self.m)]
            for p in range(random.randint(1,10)):
                for item in self.dict_dataset[user]:
                    item_trans = mmh3.hash(str(item), signed=False, seed=self.seed)
                    index, pos = self.compute_index(item_trans)
                    if fm_sketch[index][pos] == 0:
                        fm_sketch[index][pos] = 1

            self.dict_fm_sketch[user] = fm_sketch

    def compute_index(self, item):
        bucket_index = item % self.m
        temp = math.floor(item / self.m)
        binary_temp = '00' + bin(temp)[2:]
        revers_temp = binary_temp[::-1]
        position = revers_temp.find('1')

        return bucket_index, position

    def estimate_intersection(self):
        lst_user = list(self.dict_dataset.keys())
        num_user = len(lst_user)
        random.shuffle(lst_user)
        for i in range(num_user - 1):
            user_A = lst_user[i]
            user_B = lst_user[i + 1]

            lst_A = self.dict_dataset[user_A]
            lst_B = self.dict_dataset[user_B]

            fm_sketch_A = self.dict_fm_sketch[user_A]
            fm_sketch_B = self.dict_fm_sketch[user_B]

            fm_merge_sketch = [[0] * self.w for _ in range(self.m)]
            for i in range(self.m):
                for j in range(self.w):
                    fm_merge_sketch[i][j] = fm_sketch_A[i][j] | fm_sketch_B[i][j]

            cardinality_A = self.estimate_cardi(fm_sketch_A)
            cardinality_B = self.estimate_cardi(fm_sketch_B)
            cardinality_union = self.estimate_cardi(fm_merge_sketch)

            estimated_intersection = cardinality_A + cardinality_B - cardinality_union
            actual_intersection = len(list(set(lst_A).intersection(set(lst_B))))

            return [estimated_intersection]

    def estimate_cardi(self, fm_sketch):
        sum = 0
        phi = 0.77351
        for i in range(self.m):
            R = fm_sketch[i].index(0)
            sum = sum + R

        Average_R = sum / self.m
        cardinality = self.m * pow(2, Average_R) / phi

        return cardinality


if __name__ == '__main__':
    data = [random.randint(0, 2**32) for i in range(int(1e4))]
    fm = FM(data, 2048, 32, seed=1234)
    fm.build_sketch()
    fm.estimate_union()

