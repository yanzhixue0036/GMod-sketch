<meta name="robots" content="noindex">

# GMod: A Fast, Mergable, and LDP Compatible Sketch for Counting the Number of Distinct Values in Fully Dynamic Tables

This repository provides our implementation of GMod, a fully dynamic sketch designed for privacy-preserving NDV estimation. GMod efficiently supports insertions and deletions and also incorporates sketch perturbation under Local Differential Privacy (LDP), achieving high accuracy with minimal overhead. Extensive experiments confirm GMod’s superior performance and memory efficiency compared to existing methods. 

### Datasets

Our experiments encompass both synthetic datasets and real-world datasets. For the synthetic datasets, we start by selecting $𝑛$ unique integers from the 64-bit integer space. Then they are each duplicated  randomly. For the real-world datasets, evaluation uses TPC-DS for single sets and BitcoinHeist for multiple sets. TPC-DS models decision-support tasks, while BitcoinHeist covers ransomware-related transactions.

For the TPC-DS dataset, we build sketches that summarize the relevant decision-support data, which can be obtained through:

```
https://www.tpc.org/default5.asp
```

And for the BitcoinHeist dataset, we construct sketches to record the Bitcoin transaction records, which can be obtained through:

```
https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset
```

### Methods implemented

|  Method  |              Description              | Reference |
| :------: | :-----------------------------------: | :-------: |
|   GMod   |     our novel sketch method GMod      |     [GMod.py](./GMod.py)      |
| GMod-MEC | GMod with multiple estimator combined |    [GMod.py](./GMod.py)       |
|   [SFM](https://proceedings.mlr.press/v202/hehir23a.html)    |         the Sketch-Flip-Merge         |      [sfm.py](./baseline/sfm.py)     |
|    [CL](https://research.google/pubs/privacy-preserving-secure-cardinality-and-frequency-estimation/)    |      the CascadingLegions sketch      |      [cl.py](./baseline/cl.py)     |
|    [LL](https://research.google/pubs/privacy-preserving-secure-cardinality-and-frequency-estimation/)    |       the LiquidLegions sketch        |     [ll.py](./baseline/ll.py)      |
|    [HX](https://ieeexplore.ieee.org/abstract/document/10416381)    |          the HalfXor sketch           |     [hx.py](./baseline/hx.py)      |
|   [CHLL](https://www.cidrdb.org/cidr2019/papers/p23-freitag-cidr19.pdf)   |    the Counting HyperLogLog sketch    |    [chll.py](./baseline/chll.py)       |

We evaluate the performance of the night methods mentioned above in estimating cardinality for fully dynamic scenarios in [`main.py`](./main.py). Among them, HLL and FM do not incorporate privacy protection mechanisms. You can execute [`main.py`](./main.py) with the following parameters:

| Parameters        | Description                                              |
| ----------------- | -------------------------------------------------------- |
| --method          | method name: GMOD/GMOD_MEC/SFM/CL/LL/HX/CHLL      |
| --dataset         | dataset path or synthetic or deleted                     |
| --intersection    | set intersection cardinality                             |
| --difference      | set difference cardinality                               |
| --ratio           | skewness ratio used to control cardinalities of two sets |
| --delete_ratio    | delete ratio of the synthetic dataset                    |
| --exp_rounds      | the number of experimental rounds                        |
| --noise           | whether add noise                                        |
| --output          | output directory                                         |
| --epsilon         | privacy budget                                           |
| --counter         | counter size                                             |
| --sfm_Msize       | m of SFM sketch                                          |
| --sfm_Wsize       | w of SFM sketch                                          |
| --merge_method    | the merge method of SFM deterministic/random             |
| --GMod_Msize      | m of GMOD/GMOD_MEC sketch                                |
| --GMod_Wsize      | w of GMOD/GMOD_MEC sketch                                |
| --GMod_gsize      | g of GMOD/GMOD_MEC sketch                                |
| --cl_Msize        | m of Cascading_Legions                                   |
| --cl_l            | l of Cascading_Legions                                   |
| --ll_Msize        | m of Liquid_Legions                                      |
| --ll_a            | a of Liquid_Legions                                      |
| --Hx_lamb         | lamb of HalfXor                                          |
| --Hx_m            | m of HalXor                                              |
| --Hx_w            | w of HalXor                                              |
| --chll_p          | p of chll                                                |
| --chll_w          | w of chll                                                |

### Quickstart
To quickly set up a virtual environment and install the required dependencies, run:
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Example

Use `main.py` to test a method for the Single Set Case where cardinality is 100000, use GMOD-MEC to estimate the cardinality:
```
python3 main.py --method GMOD_MEC --intersection 100000 --difference 0 --noise False
```


In the Multiple Set Case, when the intersection cardinality is 100000 and the difference cardinality is 10000, use GMOD-MEC to estimate the intersection cardinality:

```
python3 main.py --method GMOD_MEC --intersection 100000 --difference 10000 --noise True
```

When the intersection cardinality is 100000 and the difference cardinality is 10000 , use GMOD_MEC to estimate the intersection cardinality with a privacy budget of 2:

```
python main.py --method GMOD_MEC --intersection 100000 --difference 10000 --noise True --epsilon 2
```

When the intersection cardinality is 100000, the difference cardinality is 10000, and the delete _ratio is 0.2, use HalfXor to estimate the intersection cardinality:

```
python main.py --method HX --intersection 100000 --difference 10000 --delete _ratio 0.2
```

When the intersection cardinality is 100000 and the difference cardinality is 10000 in the streaming case, use SFM to estimate the intersection cardinality:

```
python main.py --method SFM --intersection 100000 --difference 10000 --merge_method deterministic
```

Use CascadingLegions to estimate the intersection cardinality:

```
python main.py --method CL --intersection 1000000 --difference 100000
```
