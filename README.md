# CMC-GCN: Consistent Multi-Granularity Cascading Graph Convolution Network for Multi-Behavior Recommendation

This repository contains the official implementation of **CMC-GCN**, a novel graph-based approach for multi-behavior recommendation, as presented in our paper:
**"Consistent Multi-Granularity Cascading Graph Convolution Network for Multi-Behavior Recommendation"**

## Introduction
CMC-GCN is a sophisticated graph convolution network designed for multi-behavior recommendation systems. The model effectively captures user-item interactions across different behavior types (e.g., view, cart, purchase) through a consistent multi-granularity cascading architecture, achieving superior recommendation performance.


## Requirements
The code is implemented in Python 3 with PyTorch. The main dependencies are:
- Python 
- PyTorch


To install all required packages:
```bash
pip install -r requirements.txt
```
##  Dataset
We provide four real-world e-commerce datasets for evaluation:

- Taobao
- Tmall
- Beibei
- QKV
  
Due to GitHub's file size restrictions, the datasets are available on Baidu Netdisk (https://pan.baidu.com/s/1_-jI2orAKEMX4uqgeCZxTQ?pwd=ABCD).




After downloading, unzip the dataset.zip file to the project directory:

```
unzip dataset.zip -d ./data/
```

## Usage
To train and evaluate CMC-GCN on the Tmall dataset:
```
python main.py --data_name tmall
```
Available dataset options: taobao, tmall, beibei, qkv

## Citation
If you use this code or find our work useful, please cite our paper:
```bibtex
```
