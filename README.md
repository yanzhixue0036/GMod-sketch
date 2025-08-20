<meta name="robots" content="noindex">

# ✨ GMod: A Fast, Mergable, and LDP-Compatible Sketch for Counting Distinct Values in Fully Dynamic Tables  

[![Conference](https://img.shields.io/badge/Accepted-SIGMOD%202026-2ea44f?style=flat&logo=acm&logoColor=white)](https://sigmod.org) 
[![Python](https://img.shields.io/badge/Python-3.8+-blue?style=flat&logo=python)](https://www.python.org/) 
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](./LICENSE)



---

🚀 **GMod** is a fully dynamic sketch designed for **privacy-preserving NDV estimation**.  
It supports **insertions + deletions**, **merging**, and **Local Differential Privacy (LDP)** with:  

- ⚡ **High accuracy**  
- 🏗 **Minimal overhead**  
- 🧩 **Mergable design**  
- 🔒 **Privacy protection (LDP)**  

Extensive experiments show that **GMod outperforms existing methods** in both performance and memory efficiency.  

---

## 📊 Datasets  

We evaluate GMod on **synthetic datasets** and **real-world datasets**:  

-  **Synthetic data**: randomly duplicated integers from the 64-bit space.  
- 📈 **TPC-DS**: decision-support benchmark with 24 tables.  
- 🧮 **BitcoinHeist**: real ransomware-related Bitcoin transactions.  

📂 Links:  
- [TPC-DS](https://www.tpc.org/default5.asp)  
- [BitcoinHeist](https://archive.ics.uci.edu/dataset/526/bitcoinheistransomwareaddressdataset)  

---

## 🛠 Methods Implemented  

|   Method   | Description | File |
| :-----------: | :--------- | :-- |
| **GMod** | Our novel sketch method | [GMod.py](./GMod.py) |
| **GMod-MEC** | GMod with multiple estimators combined | [GMod.py](./GMod.py) |
| [SFM](https://proceedings.mlr.press/v202/hehir23a.html) | Sketch-Flip-Merge | [sfm.py](./baseline/sfm.py) |
| [CL](https://research.google/pubs/privacy-preserving-secure-cardinality-and-frequency-estimation/) | CascadingLegions | [cl.py](./baseline/cl.py) |
| [LL](https://research.google/pubs/privacy-preserving-secure-cardinality-and-frequency-estimation/) | LiquidLegions | [ll.py](./baseline/ll.py) |
| [HX](https://ieeexplore.ieee.org/abstract/document/10416381) | HalfXor sketch | [hx.py](./baseline/hx.py) |
| [CHLL](https://www.cidrdb.org/cidr2019/papers/p23-freitag-cidr19.pdf) | Counting HyperLogLog | [chll.py](./baseline/chll.py) |

👉 We evaluate them all in [`main.py`](./main.py).  

---

## ⚙️ Parameters  

Run experiments with `main.py`:  

| Parameter | Description |
| --------- | ----------- |
| `--method` | GMOD / GMOD_MEC / SFM / CL / LL / HX / CHLL |
| `--dataset` | dataset path or synthetic |
| `--intersection` | intersection cardinality |
| `--difference` | set difference cardinality |
| `--delete_ratio` | delete ratio for synthetic datasets |
| `--noise` | whether add noise |
| `--epsilon` | privacy budget |

---

## 🚀 Quickstart  

⚠️ Requires **Python 3.8**  

```bash
python3.8 -m venv env
source env/bin/activate
pip install -r requirements.txt
````

---

## 🧪 Examples

Single Set Case (NDV = 100000, method = GMOD-MEC):

```bash
python3 main.py --method GMOD_MEC --intersection 100000 --difference 0
```

Multiple Set Case (intersection = 10000, diff = 100000, with LDP):

```bash
python3 main.py --method GMOD_MEC --intersection 10000 --difference 100000 --noise --epsilon 2
```

Streaming Case with HalfXor (delete ratio = 0.2):

```bash
python3 main.py --method HX --intersection 100000 --difference 0 --delete_ratio 0.2
```

<!-- ---

## 🏆 Citation

If you use GMod in your research, please cite our **SIGMOD 2026** paper:

```bibtex
@inproceedings{GMod2026,
  title={GMod: A Fast, Mergable, and LDP Compatible Sketch for Counting the Number of Distinct Values in Fully Dynamic Tables},
  author={...},
  booktitle={Proceedings of the ACM SIGMOD International Conference on Management of Data},
  year={2026}
}
``` -->

