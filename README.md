<meta name="robots" content="noindex">

# GMod: A Fast, Mergable, and LDP Compatible Sketch for Counting the Number of Distinct Values in Fully Dynamic Tables

This repository provides our implementation of GMod, a fully dynamic sketch designed for privacy-preserving NDV estimation. GMod efficiently supports insertions and deletions and also incorporates sketch perturbation under Local Differential Privacy (LDP), achieving high accuracy with minimal overhead. Extensive experiments confirm GMod’s superior performance and memory efficiency compared to existing methods. 

### Datasets

Our experiments encompass both synthetic datasets and real-world datasets. For the synthetic datasets, we start by selecting $𝑛$ unique integers from the 64-bit integer space. Then they are each duplicated  randomly. For the real-world datasets, evaluation uses TPC-DS for single sets and BitcoinHeist for multiple sets. TPC-DS models decision-support tasks, while BitcoinHeist covers ransomware-related transactions.

For the real-world datasets, we use TPC-DS and BitcoinHeist, with links provided below. TPC-DS models decision-support scenarios, while BitcoinHeist captures ransomware-related Bitcoin transactions.

For the TPC-DS dataset, we perform cardinality estimation on all columns across all 24 tables in the TPC-DS benchmark. Specifically, for each column, we treat the values as input to build a sketch and estimate the number of distinct elements. The dataset and schema are publicly available at:

```
https://www.tpc.org/default5.asp
```

To simulate the multiple-set scenario, we randomly sample records from the BitcoinHeist dataset to form two subsets, which may be disjoint or overlapping. For each subset, we construct a separate sketch, inject noise according to the privacy mechanism, and then merge them to estimate the cardinality of their union. The union size ranges from $2^{15}$ to $2^{20}$, depending on the specific samples drawn. The size of the symmetric difference is not fixed and naturally varies with the degree of overlap introduced during sampling.
The dataset is available at:

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
**Note:** This project requires **Python 3.8**. Please make sure Python 3.8 is installed before proceeding.
To quickly set up a virtual environment and install the required dependencies, run:
```
python3.8 -m venv env
source env/bin/activate
pip install numpy==1.21.6
pip install scipy==1.10.1
pip install -r requirements.txt
```

### Example

Use `main.py` to test a method for the Single Set Case where cardinality is 100000, use GMOD-MEC to estimate the cardinality:
```
python3 main.py --method GMOD_MEC --intersection 100000 --difference 0
```


In the Multiple Set Case, when the intersection cardinality is 10000 and the difference cardinality is 100000, use GMOD-MEC to estimate the intersection cardinality:

```
python3 main.py --method GMOD_MEC --intersection 10000 --difference 100000 --noise
```

When the intersection cardinality is 10000 and the difference cardinality is 100000, use GMOD_MEC to estimate the intersection cardinality with a privacy budget of 2:

```
python main.py --method GMOD_MEC --intersection 10000 --difference 100000 --noise --epsilon 2
```

When the cardinality is 100000, and the delete _ratio is 0.2, use HalfXor to estimate the cardinality:

```
python main.py --method HX --intersection 100000 --difference 0 --delete_ratio 0.2
```

When the intersection cardinality is 100000 and the difference cardinality is 10000 in the streaming case, use SFM to estimate the intersection cardinality:

```
python main.py --method SFM --intersection 10000 --difference 100000 --merge_method deterministic
```

Use CascadingLegions to estimate the intersection cardinality:

```
python main.py --method CL --intersection 10000 --difference 100000
```
