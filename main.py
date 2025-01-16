import time
import random
import argparse

from utils import *
from loader import Dataloader

from GMod import GMod
from baseline.cl import CL
from baseline.fm import FM
from baseline.ll import LL
from baseline.sfm import SFM
from baseline.hll import HyperLogLog
from baseline.hx import HalfXor
from baseline.chll import CHLL
def get_args():
    parser = argparse.ArgumentParser(description="sketch method for estimating intersection cardinalities")
    parser.add_argument('--method', default='LL', type=str, help='method name: GMOD/GMOD_MEC/SFM/CL/LL/HX/CHLL')
    parser.add_argument('--dataset', default='synthetic', type=str, help='dataset path or synthetic or deteled')
    parser.add_argument('--intersection', default=10000, type=int, help='set intersection cardinality') 
    parser.add_argument('--difference', default=100000, type=int, help='set difference cardinality')
    parser.add_argument('--ratio', default=0.5, type=float, help='skewness ratio used to control cardinalities of two sets')
    parser.add_argument('--delete_ratio', default=0.0, type=float, help='delete ratio of the sththetic dataset')
    parser.add_argument('--exp_rounds', default=10, type=int, help='the number of experimental rounds') 
    parser.add_argument('--noise',default=True, type=bool, help='whether add noise')
    parser.add_argument('--output', default='result/', type=str, help='output directory')
    parser.add_argument('--epsilon', default=1, type=int, help='privacy budget')
    parser.add_argument('--counter', default=32, type=int, help='counter size')

    # SFM param
    parser.add_argument('--sfm_Msize', default=4096, type=int, help='m of SFM sketch')
    parser.add_argument('--sfm_Wsize', default=32, type=int, help='w of SFM sketch')
    parser.add_argument('--merge_method', default='deterministic', type=str, help='the merge method of SFM deterministic/random')

    # GMOD/GMOD_MEC param
    parser.add_argument('--GMod_Msize', default=4096, type=int, help='m of GMOD/GMOD_MEC sketch')
    parser.add_argument('--GMod_Wsize', default=32, type=int, help='w of GMOD/GMOD_MEC sketch')
    parser.add_argument('--GMod_gsize', default=3, type=int, help='g of GMOD/GMOD_MEC sketch')
    parser.add_argument('--random_response',default=True, type=bool, help='whether use random response')
    
    # Cascading_Legions param
    parser.add_argument('--cl_Msize', default=4096, type=int, help='m of Cascading_Legions')
    parser.add_argument('--cl_l', default=32, type=int, help='l of Cascading_Legions')
    
    # Liquid_Legions param
    parser.add_argument('--ll_Msize', default=8192, type=int, help='m of Liquid_Legions')
    parser.add_argument('--ll_a', default=10, type=int, help='a of Liquid_Legions')
    
    # HalfXor param
    parser.add_argument('--Hx_lamb', default=1/4096, type=float, help='lamb of HalfXor')
    parser.add_argument('--Hx_m', default=4096, type=int, help = 'm of HalXor')
    parser.add_argument('--Hx_w', default=32, type=int, help = 'w of HalXor')
    
    #CHLL param
    parser.add_argument('--chll_p', default=8, type=int, help='p of chll')
    parser.add_argument('--chll_w', default=32, type=int, help='w of chll')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    exp_rounds = args.exp_rounds
    lst_all_results = list()
    all_time_list = list()
    for r in range(exp_rounds):
        seed = random.randint(1, 2**32-1)
        dataloader = Dataloader(args.dataset, args.intersection, args.difference, args.ratio, seed, args.delete_ratio, maxrepeattimes=10)
        dict_dataset = dataloader.load_dataset('synthetic')
        delete_dataset = dataloader.load_dataset('deleted')
        print('{}th exp_rounds dataset generation finished!'.format(r))
        start_time = time.time()
        if args.method == 'GMOD' and args.noise == True:
            g = int(math.log(min(pow(2, args.GMod_gsize), math.exp(args.epsilon)+1)))
            gmod = GMod(dict_dataset, args.GMod_Msize, args.GMod_Wsize, g,args.output,seed,args.epsilon, args.random_response,delete_dataset)
            gmod.build_sketch()
            result = gmod.estimate_union()
            lst_all_results.append(result[0])
        
        if args.method == 'GMOD' and args.noise == False:
            gmod = GMod(dict_dataset, args.GMod_Msize, args.GMod_Wsize, args.GMod_gsize,args.output,seed,args.epsilon, False,delete_dataset)
            gmod.build_sketch()
            result = gmod.estimate_union() # 
            lst_all_results.append(result[0])
        
        if args.method == 'GMOD_MEC' and args.noise == True:
            g = int(math.log(min(pow(2, args.GMod_gsize), math.exp(args.epsilon)+1)))
            gmod = GMod(dict_dataset, args.GMod_Msize, args.GMod_Wsize, g,args.output,seed,args.epsilon, args.random_response,delete_dataset)
            gmod.build_sketch()
            result = gmod.estimate_union_IVW()
            lst_all_results.append(result[0])
            
        if args.method == 'GMOD_MEC' and args.noise == False:
            gmod = GMod(dict_dataset, args.GMod_Msize, args.GMod_Wsize, args.GMod_gsize,args.output,seed,args.epsilon, False,delete_dataset)
            gmod.build_sketch()
            result = gmod.estimate_union_IVW()
            lst_all_results.append(result[0])
            
        if args.method == "SFM":
            sfm = SFM(dict_dataset, int(args.sfm_Msize/args.counter), args.sfm_Wsize, args.epsilon, args.merge_method, seed)
            sfm.build_fm_sketch()
            result = sfm.estimation_union_cardinality()
            lst_all_results.append(result[0])
            
        if args.method == "CL":
            cl = CL(dict_dataset, int(args.cl_Msize/args.counter), args.cl_l, args.epsilon, seed)
            cl.build_sketch()
            result = cl.estimation_union_car()
            lst_all_results.append(result[0])
            
        if args.method == "LL":
            ll = LL(dict_dataset, args.ll_a, int(args.ll_Msize/args.counter), args.epsilon, seed)
            ll.build_sketch()
            result = ll.estimation_union_cardinality()
            lst_all_results.append(result[0])
        
        if args.method == "HX":
            hx = HalfXor(dict_dataset, args.Hx_lamb, args.Hx_m, args.Hx_w, seed,delete_dataset)
            hx.build_sketch()
            result = hx.count()
            lst_all_results.append(result[0])
        
        if args.method == "CHLL":
            chll = CHLL(dict_dataset, args.chll_p, args.chll_w, seed,delete_dataset)
            chll.build_sketch()
            result = chll.count()
            lst_all_results.append(result)
        
        
        end_time = time.time()
        each_round_time = end_time - start_time
        all_time_list.append(each_round_time)

    print('time:', sum(all_time_list)/len(all_time_list))
    #print(lst_all_results)
    #print((int)(args.intersection + args.difference)*(1-args.delete_ratio))
    AARE = compute_aare(lst_all_results, (int)(args.intersection + args.difference)*(1-args.delete_ratio*args.noise)) #
    print("The value of AARE: {}%".format(AARE * 100))