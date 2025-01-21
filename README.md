# AMP
A Novel Adjacent Matrix-based Probabilistic Selection Mechanism for Differential Evolution

## Abstract
This paper proposes a novel selection mechanism for differential evolution (DE) termed the adjacent matrix-based probabilistic (AMP) selection mechanism. When DE enters the selection phase, the proposed method merges the parent and offspring populations and constructs an adjacency matrix using the Manhattan distance. The concept of competition is then introduced, with the probability of being selected as a competitor determined based on the adjacency matrix. According to the proximate optimality principle (POP), closer individuals have a higher probability of being competitors, as neighbors tend to share similar genome and fitness information. This allows a cluster to be represented by a superior individual, reducing redundant fitness evaluations. For each individual, k competitors are chosen, and fitness comparisons are conducted. The winner counts are ranked to determine which individuals survive. To evaluate the performance of the proposed AMP selection mechanism, we integrated it with eight variants of DE and conducted numerical experiments on CEC2017 and CEC2022 benchmark functions. The experimental results and statistical analysis confirm the effectiveness and robustness of the AMP selection mechanism, demonstrating its great potential for integration with population-based optimization techniques to address various optimization tasks. The source code of this research can be downloaded from https://github.com/RuiZhong961230/AMP.

## Citation
@article{Zhong:25,  
title={A Novel Adjacent Matrix-based Probabilistic Selection Mechanism for Differential Evolution},  
author={Rui Zhong and Shilong Zhang and Yujun Zhang and Jun Yu},  
journal={Cluster Computing},  
pages={1-23},  
year={2025},  
volume={28},  
publisher={Springer},  
doi={https://doi.org/10.1007/s10586-024-04915-4 }  
}  

## Datasets and Libraries
CEC benchmarks are provided by the opfunu library.
