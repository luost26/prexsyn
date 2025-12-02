# Chemical Space Projection

To reproduce the results in Table 1 of [the PrexSyn paper](https://arxiv.org/abs/2512.00384), run the following benchmark script:

```bash
python scripts/benchmarks/projection.py
```

There are two benchmark datasets (Enamine REAL and ChEMBL) and three different sample sizes (64, 128, and 256). For each setting, 5 independent runs are performed, leading to a total of 2x3x5=30 runs.

The output should be similar to the following, which corresponds to the results in Table 1:

```
   dataset  num_samples  similarity_mean  similarity_std  recons_mean  recons_std
0   ChEMBL           64           0.7300          0.0008       0.2540      0.0031
1   ChEMBL          128           0.7429          0.0006       0.2718      0.0033
2   ChEMBL          256           0.7533          0.0011       0.2832      0.0020
3  Enamine           64           0.9819          0.0005       0.9264      0.0030
4  Enamine          128           0.9845          0.0004       0.9360      0.0023
5  Enamine          256           0.9859          0.0007       0.9406      0.0026
   num_samples  time_mean  time_std
0           64     0.1021    0.0312
1          128     0.1490    0.0378
2          256     0.2618    0.0563
```

Please note that sampling time is dependent on your specific hardware and system load. The results shown above and in the paper were obtained using a single NVIDIA 4090 GPU, as detailed in the paper.
