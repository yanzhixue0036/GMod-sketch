from __future__ import annotations
import struct, copy
from typing import Callable, Optional
import numpy as np
import warnings
import random
from hashfunc import sha1_hash32, sha1_hash64, mmh3_hash32, mmh3_hash64

class CountingHyperLogLogCounter(object):
    
    def __init__(self, seed, value=0, hashfunc: Callable = mmh3_hash32,):
        self.seed = seed
        self.value = value
        self.hashfunc = hashfunc
        self.p = 32
        
        self.max_value = 255
        self.min_value = 0
        

    def increment(self, b):
        if self.value <= 128:
            self.value += 1
        else:
            hv = self.hashfunc(b, seed=self.seed)
            rand_int = hv & (2**self.p - 1)
            rand_random = rand_int / 2**self.p
            if rand_random <= 1/2**(self.value-128):
                self.value += 1
            
    def decrement(self, b):
        if self.value == 0 :
            pass
        elif self.value <= 128:
            self.value -= 1
        else:
            hv = self.hashfunc(b, seed=self.seed)
            rand_int = hv & (2**self.p - 1)
            rand_random = rand_int / 2**self.p
            if rand_random <= 1/2**(self.value-129):
                self.value -= 1

    def get_value(self):
        return self.value
            




class CHLL(object):
    '''
    Methods:
        
        __init__(...):
            Initializes the object with the given parameters.
        
        insert(e):
            Inserts an element e into the sketch.
        
        delete(e):
            Deletes an element e from the sketch.
        
        count():
            Estimates the cardinality with the counting sketch as same as the HyperLogLog.

    '''
    
    __slots__ = ("dict_dataset", "delete_dataset","repeatTimes",\
        "p", "w" ,"m", "reg", "alpha", "max_rank", "hashfunc", "hash_range_bit", "seed")

    # HACK
    # The range of the hash values used for HyperLogLog
    
    
    def _get_alpha(self, p):
        if not (4 <= p <= 16):
            raise ValueError("p=%d should be in range [4 : 16]" % p)
        if p == 4:
            return 0.673
        if p == 5:
            return 0.697
        if p == 6:
            return 0.709
        return 0.7213 / (1.0 + 1.079 / (1 << p))
    
    def __init__(
        self,
        dict_dataset,
        p: int ,
        w: int ,
        seed: int = random.randint(1, 2**32-1),
        delete_dataset = None,
        hashfunc: Callable = mmh3_hash32,
        hash_range_bit: int = 32 
    ):
        self.p = p
        self.m = 1 << p
        self.w = w
        self.reg = np.zeros((self.m, self.w), dtype=object)
        self.hash_range_bit = hash_range_bit
        for i in range(self.m):
            for j in range(self.w):
                self.reg[i, j] = CountingHyperLogLogCounter(seed-1)
        
        self.seed = seed
        random.seed(self.seed)
        self.dict_dataset = dict_dataset
        if delete_dataset == None:
            self.delete_dataset = {}
            for user in self.dict_dataset:
                self.delete_dataset[user] = {}
                self.delete_dataset[user]['elements']=[]
        else:
            self.delete_dataset = delete_dataset    
        # if delete_dataset == None:
        #     self.delete_dataset = {}
        #     for user in self.dict_dataset:
        #         self.delete_dataset[user] = {}
        #         self.delete_dataset[user]['elements'] = []
        # else:
        #     self.delete_dataset = delete_dataset
        # self.repeatTimes = 1 if delete_dataset==None else 10
        self.repeatTimes = 1
    
        # Check the hash function.
        if not callable(hashfunc):
            raise ValueError("The hashfunc must be a callable.")
        self.hashfunc = hashfunc
        
        # Common settings
        self.alpha = self._get_alpha(self.p)
        # self.max_rank = self.hash_range_bit
        self.max_rank = w


    
    
    def _get_rank(self, bits):
        # Get the number of bits starting from the first non-zero bit to the right
        _bit_length = lambda bits: bits.bit_length()
        # For < Python 2.7
        if not hasattr(int, "bit_length"):
            _bit_length = lambda bits: len(bin(bits)) - 2 if bits > 0 else 0
        
        rank = self.max_rank - _bit_length(bits) + 1
        if rank <= 0:
            raise ValueError(
                "Hash value overflow, maximum size is %d\
                    bits"
                % self.max_rank
            )
        return min(rank, self.w-1)
    
    def insert(self, b) -> None:
        """
        Update the HyperLogLog with a new data value in bytes.
        The value will be hashed using the hash function specified by
        the `hashfunc` argument in the constructor.

        Args:
            b: The value to be hashed using the hash function specified.

        Example:
            To update with a new string value (using the default SHA1 hash
            function, which requires bytes as input):

            .. code-block:: python

                hll = HyperLogLog()
                hll.update("new value".encode('utf-8'))

            We can also use a different hash function, for example, `pyfarmhash`:

            .. code-block:: python

                import farmhash
                def _hash_32(b):
                    return farmhash.hash32(b)
                hll = HyperLogLog(hashfunc=_hash_32)
                hll.update("new value")
        """
        b = str(b).encode('utf-8')
        # Digest the hash object to get the hash value
        hv = self.hashfunc(b, seed=self.seed)
        # Get the index of the register using the first p bits of the hash
        reg_index = hv & (self.m - 1)
        # Get the rest of the hash
        bits = self.hashfunc(b, seed=self.seed+1)
        # Update the register
        self.reg[reg_index, self._get_rank(bits)].increment(b)


    def delete(self, b) -> None:
        b = str(b).encode('utf-8')
        # Digest the hash object to get the hash value
        hv = self.hashfunc(b, seed=self.seed)
        # Get the index of the register using the first p bits of the hash
        reg_index = hv & (self.m - 1)
        # Get the rest of the hash
        bits = self.hashfunc(b, seed=self.seed+1)
        
        # Update the register
        self.reg[reg_index, self._get_rank(bits)].decrement(b)
        
    def build_sketch(self):
        
        for user in self.dict_dataset:   
            # arrive
            repeattimes = 1
            user_dict = self.dict_dataset[user]
            for i in range(len(user_dict['elements'])):
                repeattimes = user_dict['repeattimes'][i]
                item = user_dict['elements'][i]                
                self.insert(item)
            #delete
            user_dict = self.delete_dataset[user]
            for i in range(len(user_dict['elements'])):
                item = user_dict['elements'][i]                
                self.delete(item)
                
    def _linearcounting(self, num_zero):
        return self.m * np.log(self.m / float(num_zero))

    def count(self) -> float:
        """
        Estimate the cardinality of the data values seen so far.

        Returns:
            float: The estimated cardinality.
        """
        first_nonzero_index = np.zeros(self.m)
        for i in range(self.m):
            for j in range(self.w):
                if self.reg[i,j].get_value() != 0:
                    first_nonzero_index[i] = j
                    
        
        # Use HyperLogLog estimation function
        e = self.alpha * float(self.m**2) / np.sum(2.0 ** (-first_nonzero_index))
        # Small range correction
        small_range_threshold = (5.0 / 2.0) * self.m
        if abs(e - small_range_threshold) / small_range_threshold < 0.15:
            warnings.warn(
                (
                    "Warning: estimate is close to error correction threshold. "
                    + "Output may not satisfy HyperLogLog accuracy guarantee."
                )
            )
        if e <= small_range_threshold:
            num_zero = self.m - np.count_nonzero(first_nonzero_index)
            return self._linearcounting(num_zero)
        # Normal range, no correction
        if e <= (1.0 / 30.0) * (1 << 32):
            return e
        # Large range correction
        return self._largerange_correction(e)
