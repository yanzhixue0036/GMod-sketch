from __future__ import annotations
import struct, copy
from typing import Callable, Optional
import numpy as np
import warnings
from hashfunc import sha1_hash32, sha1_hash64, mmh3_hash32, mmh3_hash64
from math import exp,log
import random
import json
class HalfXor(object):
    '''
    Methods:
        
        __init__(lamb,m,w,hashfunc,hash_range_bit) :
            Initializes the object with the given parameters.
        
        insert(e,r):
            Inserts an element e with rid r into the sketch.
        
        delete(e,r):
            Deletes an element e with rid r from the sketch.
        
        count():
            Estimates the cardinality with the sketch B and D.


    '''
    __slots__ = ("dict_dataset", "delete_ratio", "repeatTimes",\
        "m", "w", "lamb", "B", "D", "hashfunc", "hash_range_bit", "seed","delete_dataset")

    def __init__(
        self,
        dict_dataset,
        lamb: float,
        m: int ,
        w: int,
        seed = random.randint(1, 2**32-1),
        delete_dataset=None,
        hashfunc: Callable = mmh3_hash32,
        hash_range_bit  = 32
    ):
        self.dict_dataset = dict_dataset
        self.m = m
        self.w = w
        self.lamb = lamb
        self.B=np.zeros((m,w),dtype=np.bool_)
        self.D=np.zeros(m)
        self.hashfunc = hashfunc
        self.hash_range_bit = hash_range_bit
        
        self.seed = seed
        random.seed(self.seed)
        # if delete_dataset == None:
        #     self.delete_dataset = {}
        #     for user in self.dict_dataset:
        #         self.delete_dataset[user] = {}
        #         self.delete_dataset[user]['elements'] = []
        # else:
        #     self.delete_dataset = delete_dataset
        # self.repeatTimes = 1 if delete_dataset==None else 10
        self.delete_dataset = {}
        if delete_dataset == None:
            self.delete_dataset = {}
            for user in self.dict_dataset:
                self.delete_dataset[user] = {}
                self.delete_dataset[user]['elements']=[]
        else:
            self.delete_dataset = delete_dataset       
        self.repeatTimes = 1
    

    def poisson(self, u):
        m=self.m
        lamb=self.lamb
        x = 0
        p = exp(-(m * lamb))
        s = p
        for i in range(100):
            if u <= s:
                break
            x += 1
            p = p * (m * lamb) / float(x)
            s += p
        
        return x
    
    def f_i(self,i,e): 
        u=self.hashfunc((i,e), seed=self.seed+self.m+i) / 2**self.hash_range_bit
        return self.poisson(u) # possion分布
    
    def h(self,i,r,k): return self.hashfunc((i,r,k), seed=self.seed) % 2  
    def h_i(self,i,e,k):
        return self.hashfunc((i,e,k), seed=self.seed+i) 
    def row(self,h_i):
        w=self.w
        #print(h_i)
        binary_item=bin(h_i)[2:].zfill(32)
        j = 0
        for bit in binary_item:
            if bit == '0' and j < w:
                j += 1
            else:
                break
        
        return  j 
    def h_m(self,e): 
        m = self.m
        return self.hashfunc(e, seed=self.seed - 1) % m 

    def _update_B(self,e,r):
        for i in range(self.m):
            tmp = self.f_i(i,e)
            
            for k in range(tmp):
                
                x_irk = self.h(i,r,k)
                #print("x_irk=",x_irk)
                j=self.row(self.h_i(i,e,k))
                #print("j=",j)
                self.B[i][j]= self.B[i][j] ^ x_irk
                     
    def _update_B_3(self,e,r):
        hv = self.hashfunc(e, seed=self.seed-2)
        i= hv & (self.m - 1)
        bits = hv >> int(log(self.m,2))
        x_ir1=self.h(i,r,1)
        j=self.row(self.h_i(i,e,1))
        self.B[i][j]= self.B[i][j] ^ x_ir1
    

    def insert(self,e,r):
        e = str(e).encode('utf-8')
        r = str(r).encode('utf-8')
        self._update_B_3(e,r)
        # 更新D        
        j_2=self.h_m(e)
        self.D[j_2]=self.D[j_2] + 1
        


    def delete(self,e,r):
        e = str(e).encode('utf-8')
        r = str(r).encode('utf-8')
        self._update_B_3(e,r)
         
        j_2=self.h_m(e)
        self.D[j_2]=self.D[j_2] - 1
        
    
    def build_sketch(self):
        #row = 0
        #with open("output.json", "w", encoding="utf-8") as file:
        #    json.dump(self.dict_dataset, file, indent=4, ensure_ascii=False)
        for user in self.dict_dataset:   
            # arrive
            repeattimes = 1
            user_dict = self.dict_dataset[user]
            for i in range(len(user_dict['elements'])):
                repeattimes = user_dict['repeattimes'][i]
                item = user_dict['elements'][i]                
                self.insert(item, user_dict['index'][i])
                #row+=1
            #delete
            user_dict = self.delete_dataset[user]
            for i in range(len(user_dict['elements'])):
                self.delete(user_dict['elements'][i],user_dict['index'][i])
                
    def p_j(self,j):
        w=self.w
        if j>=0 and j<=w-2:
            return 2**(-j-1)
        elif j==w-1:
            return 2**(-w+1)
    
    def phi_func(self, n):
        lamb=self.lamb
        w=self.w
        
        output=0
        for j in range(w): 
            output += 1 - exp(-n*lamb*self.p_j(j))
        output=output/2/w
        return output
    
    def binary_search_estimate(self):
        
        B=self.B
        m=self.m
        w=self.w
        lamb=self.lamb
        
        def binary_search(f, low, high, tolerance=1e-7):
            while high - low > tolerance:
                mid = (low + high) / 2
                if f(mid) == 0:
                    return mid
                elif f(mid) * f(low) < 0:
                    high = mid
                else:
                    low = mid
            return (low + high) / 2
        
        count = 0
        for i in range(m):
            for j in range(w):
                count += B[i][j]
        v = count /m/w
        
        low = 1 
        for i in range(10):
            if((self.phi_func(pow(10,i)) - v) * (self.phi_func(pow(10,i+1)) - v) < 0):
                low = pow(10,i)
        high=low*10
        
        
        phi_func_fixed = lambda n: self.phi_func(n) - v
        
        output=binary_search(phi_func_fixed,1,1e9,1)
        # print("e1=",output)
        return output

    def IVW_estimate(self):
        n_est=self.binary_search_estimate()
        lamb=self.lamb
        B = self.B
        m=self.m
        w=self.w
        
        v_j=np.zeros(w)
        for j in range(w):
            for i in range(m):
                v_j[j] += B[i][j]
        v_j=v_j/m
        
        def n_j_est(j) :
            tmp=1-2*v_j[j]
            if tmp>0:
                return -log(1-2*v_j[j]) / lamb /self.p_j(j)
            else:
                return 0
            
        def var_n_j_est_rec(j):
        
            return (m*( lamb*self.p_j(j)*exp(-n_est *lamb*self.p_j(j)) )**2) / (1-exp(-2*n_est*lamb*self.p_j(j)))
            # except ZeroDivisionError:
            #     print("m=",m)
            #     print("p_j=",self.p_j(j))
            #     print("exp(-n_est *lamb*self.p_j(j))=",exp(-n_est *lamb*self.p_j(j)))
        
        def w_j(j):
            temp=0
            for l in range(w):
                temp += var_n_j_est_rec(l)
            return var_n_j_est_rec(j) / temp
        
        def n_est_star():
            temp =0
            for j in range(w):
                temp += w_j(j)*n_j_est(j)
            return temp
        
        return n_est_star()

    def D_estimate(self):
        m=self.m
        D=self.D
        # print(D)
        u=0
        for i in range(m):
            if D[i] == 0:
                u+=1
        u=u/m
        if u==0:
            return None
        else:
            n_d_est=-m*log(u)
            return n_d_est

    def count(self):
        m=self.m
        n_b_est=self.binary_search_estimate()
        n_d_est=self.D_estimate()
        
        if n_b_est > 3*m:
            return [n_b_est]
        else:
            return [n_d_est]
    
    def count_IVW(self):
        m=self.m
        n_b_est=self.IVW_estimate()
        n_d_est=self.D_estimate()
        
        if n_b_est > 3*m:
            return [n_b_est]
        else:
            return [n_d_est]
