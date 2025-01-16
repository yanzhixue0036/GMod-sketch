import mmh3
import math
import random
import os
from math import ceil, log, log2


class HyperLogLog:
    def __init__(self, dict_dataset, size, seed):
        self.dict_dataset = dict_dataset
        self.size = size
        self.seed = seed

    def build_sketch(self):
        self.dict_hll_sketch = dict()

        for user in self.dict_dataset:
            hll_sketch = [0] * self.size
            flag = ceil(log2(self.size))

            for item in self.dict_dataset[user]:
                item_trans = mmh3.hash(str(item), signed=False, seed=self.seed)
                index, value = self.compute_index_value(item_trans, flag)
                if value > hll_sketch[index]:
                    hll_sketch[index] = value

            self.dict_hll_sketch[user] = hll_sketch

    def compute_index_value(self, item, flag):
        binary_item = '0' * (32 - len(bin(item)[2:])) + bin(item)[2:]
        index = int(binary_item[0:flag], 2) % self.size
        value = 0
        for bit in binary_item[flag:]:
            if bit == '0':
                value += 1
            else:
                break
        value += 1

        return index, value

    def estimate_cardi(self, hll_sketch):
        zero_bits = 0
        alpha = 0.7213 / (1 + 1.079 / self.size)
        tmp = 0

        for i in range(self.size):
            tmp += (2 ** (-hll_sketch[i]))
            if hll_sketch[i] == 0:
                zero_bits += 1
        cardinality = alpha * (self.size ** 2) / tmp

        if zero_bits == 0:
            zero_bits = 1

        if cardinality < 2.5 * self.size:
            cardinality = -self.size * log(zero_bits / self.size)

        return cardinality

    def estimate_intersection(self):
        random.seed(self.seed)

        lst_user = list(self.dict_dataset.keys())
        num_user = len(lst_user)
        random.shuffle(lst_user)

        for i in range(num_user - 1):
            user_A = lst_user[i]
            user_B = lst_user[i + 1]

            lst_A = self.dict_dataset[user_A]
            lst_B = self.dict_dataset[user_B]

            hll_sketch_A = self.dict_hll_sketch[user_A]
            hll_sketch_B = self.dict_hll_sketch[user_B]

            hll_sketch_merge = [0] * self.size
            for j in range(self.size):
                hll_sketch_merge[j] = max(hll_sketch_A[j], hll_sketch_B[j])

            cardinality_A = self.estimate_cardi(hll_sketch_A)
            cardinality_B = self.estimate_cardi(hll_sketch_B)
            cardinality_union = self.estimate_cardi(hll_sketch_merge)

            estimated_intersection = cardinality_A + cardinality_B - cardinality_union
            actual_intersection = len(list(set(lst_A).intersection(set(lst_B))))

            return [estimated_intersection]

