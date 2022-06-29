# Reframed GES with a Neural Conditional Dependence Measure

This repository contains the code for the UAI 2022 paper [*Reframed GES with a Neural Conditional Dependence Measure*](https://arxiv.org/abs/2206.08531). 

The code implements the reframed GES algorithm with two nonparametric conditional dependence measures, NCD and RCD, for causal discovery. For practitioners, NCD is neural network based and involves hyperparameter tuning as in most deep learning methods. See Appendix D.2 in the paper for a guideline for hyperparameter tuning. RCD (Azadkia and Chatterjee, 2021) is hyperparameter-free and faster to compute.

## Run
```
python main.py
```

## Contact information
If you meet any problems with the code, please submit an issue or contact Xinwei Shen (`xinwei.shen@connect.ust.hk`).

## Citation
If you would refer to or extend our work, please cite the following paper:
```
@inproceedings{
shen2022reframed,
title={Reframed GES with a Neural Conditional Dependence Measure},
author={Shen, Xinwei and Zhu, Shengyu and Zhang, Jiji and Hu, Shoubo and Chen, Zhitang},
booktitle={The 38th Conference on Uncertainty in Artificial Intelligence},
year={2022},
url={https://openreview.net/forum?id=Sl-eewIi9e5}
}
```

## Acknowledgments
The code is based on the FGES (Ramsey et al., 2017) implementation from [this repository](https://github.com/eberharf/fges-py).
