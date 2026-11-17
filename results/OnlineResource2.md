# Online Resource 2

**Ground-truth benchmarking of signed, directed lagged connectivity: TemporalSignedMBB from a simulated gut-brain model to neural data**

Tito Waluyo Purboyo

Department of Biomedical Engineering, School of Electrical Engineering, Telkom University, Bandung 40257, Indonesia; titowaluyo@telkomuniversity.ac.id

Supplementary tables. Every value is read from the result files of Online Resource 1. Means are over independent replications; MCSE, Monte Carlo standard error.


**Table S1** Sensitivity of Layer 1 to the omitted CLR coordinate (10 replications of the main scenario, cluster-robust test). Precision, recall and conditional sign accuracy on the nodes common to all four choices; Jaccard index of the signed edge sets relative to the main analysis (Flavonifractor omitted) and number of shared edges with opposite signs, summed over replications. In the generator, Flavonifractor carries no coupling, whereas the other three taxa carry one or two background edges in most replications

| Omitted coordinate | Network | Precision | Recall | Sign accuracy | Jaccard vs main | Sign changes |
|:------------------------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| Flavonifractor | HC-T1 | 0.922 | 0.910 | 1.000 | -- | -- |
| Flavonifractor | MDD-T1 | 0.873 | 0.879 | 1.000 | -- | -- |
| Flavonifractor | NR-T3 | 0.903 | 0.769 | 1.000 | -- | -- |
| Holdemania | HC-T1 | 0.777 | 0.923 | 1.000 | 0.708 | 0 |
| Holdemania | MDD-T1 | 0.743 | 0.888 | 0.999 | 0.714 | 0 |
| Holdemania | NR-T3 | 0.787 | 0.787 | 1.000 | 0.710 | 0 |
| Turicibacter | HC-T1 | 0.712 | 0.935 | 0.999 | 0.661 | 1 |
| Turicibacter | MDD-T1 | 0.641 | 0.890 | 0.995 | 0.645 | 8 |
| Turicibacter | NR-T3 | 0.766 | 0.797 | 0.997 | 0.692 | 0 |
| Christensenella | HC-T1 | 0.678 | 0.916 | 0.998 | 0.625 | 1 |
| Christensenella | MDD-T1 | 0.623 | 0.885 | 0.998 | 0.597 | 1 |
| Christensenella | NR-T3 | 0.692 | 0.788 | 0.999 | 0.626 | 0 |

**Table S2** Null scenario (50 replications): per-pair rejection rate at α = 0.05 (MCSE) and proportion of networks with any false edge after Benjamini-Hochberg correction at q = 0.05, by network and test

| Network | Naive: rate | Naive: any false | df-corrected: rate | df-corrected: any false | Cluster: rate | Cluster: any false |
|:------------------------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| HC-T1 | 0.0508 (0.0013) | 0.06 | 0.0172 (0.0008) | 0.00 | 0.0508 (0.0013) | 0.06 |
| HC-T2 | 0.0533 (0.0013) | 0.06 | 0.0184 (0.0008) | 0.00 | 0.0530 (0.0013) | 0.06 |
| HC-T3 | 0.0513 (0.0012) | 0.02 | 0.0176 (0.0005) | 0.00 | 0.0513 (0.0011) | 0.04 |
| MDD-T1 | 0.0509 (0.0013) | 0.06 | 0.0170 (0.0007) | 0.00 | 0.0512 (0.0013) | 0.06 |
| R-T1 | 0.0503 (0.0014) | 0.02 | 0.0163 (0.0007) | 0.00 | 0.0512 (0.0014) | 0.02 |
| NR-T1 | 0.0506 (0.0011) | 0.00 | 0.0163 (0.0005) | 0.00 | 0.0516 (0.0011) | 0.04 |
| R-T2 | 0.0508 (0.0014) | 0.06 | 0.0157 (0.0007) | 0.00 | 0.0511 (0.0014) | 0.02 |
| NR-T2 | 0.0520 (0.0013) | 0.04 | 0.0171 (0.0007) | 0.00 | 0.0523 (0.0013) | 0.04 |
| R-T3 | 0.0512 (0.0010) | 0.00 | 0.0160 (0.0005) | 0.00 | 0.0520 (0.0011) | 0.06 |
| NR-T3 | 0.0529 (0.0013) | 0.06 | 0.0173 (0.0007) | 0.00 | 0.0536 (0.0013) | 0.04 |

**Table S3** Observation stages and demeaning (main scenario, 30 replications, cluster-robust test): precision, recall and mean estimated self-coefficient (true value 0.45) by stage, subject-mean treatment and network

| Stage and subject mean | Network | Precision | Recall | Self-coefficient |
|:------------------------|:---------:|:---------:|:---------:|:---------:|
| latent, full | HC-T1 | 0.943 | 1.000 | 0.347 |
| latent, full | MDD-T1 | 0.905 | 0.995 | 0.355 |
| latent, full | R-T3 | 0.931 | 0.998 | 0.356 |
| latent, full | NR-T3 | 0.902 | 0.992 | 0.362 |
| latent, full | R-T2 | 0.922 | 0.991 | 0.346 |
| latent, full | NR-T2 | 0.914 | 0.992 | 0.346 |
| latent, window | HC-T1 | 0.641 | 1.000 | 0.009 |
| latent, window | MDD-T1 | 0.640 | 0.992 | 0.008 |
| latent, window | R-T3 | 0.806 | 0.962 | 0.000 |
| latent, window | NR-T3 | 0.769 | 0.978 | 0.003 |
| latent, window | R-T2 | 0.801 | 0.967 | -0.001 |
| latent, window | NR-T2 | 0.764 | 0.984 | 0.003 |
| latent, true | HC-T1 | 0.954 | 1.000 | 0.443 |
| latent, true | MDD-T1 | 0.949 | 0.995 | 0.443 |
| latent, true | R-T3 | 0.938 | 0.982 | 0.434 |
| latent, true | NR-T3 | 0.941 | 0.992 | 0.438 |
| latent, true | R-T2 | 0.941 | 0.991 | 0.434 |
| latent, true | NR-T2 | 0.946 | 0.992 | 0.438 |
| clr, full | HC-T1 | 0.927 | 1.000 | 0.348 |
| clr, full | MDD-T1 | 0.896 | 0.990 | 0.356 |
| clr, full | R-T3 | 0.914 | 0.965 | 0.357 |
| clr, full | NR-T3 | 0.904 | 0.977 | 0.363 |
| clr, full | R-T2 | 0.917 | 0.950 | 0.346 |
| clr, full | NR-T2 | 0.906 | 0.979 | 0.347 |
| clr, window | HC-T1 | 0.709 | 0.992 | 0.008 |
| clr, window | MDD-T1 | 0.714 | 0.978 | 0.008 |
| clr, window | R-T3 | 0.808 | 0.821 | 0.000 |
| clr, window | NR-T3 | 0.786 | 0.891 | 0.003 |
| clr, window | R-T2 | 0.829 | 0.812 | -0.002 |
| clr, window | NR-T2 | 0.800 | 0.912 | 0.002 |
| clr, true | HC-T1 | 0.937 | 1.000 | 0.444 |
| clr, true | MDD-T1 | 0.942 | 0.991 | 0.444 |
| clr, true | R-T3 | 0.917 | 0.961 | 0.435 |
| clr, true | NR-T3 | 0.934 | 0.982 | 0.440 |
| clr, true | R-T2 | 0.929 | 0.968 | 0.435 |
| clr, true | NR-T2 | 0.938 | 0.988 | 0.439 |
| observed, full | HC-T1 | 0.920 | 0.905 | 0.241 |
| observed, full | MDD-T1 | 0.894 | 0.881 | 0.248 |
| observed, full | R-T3 | 0.943 | 0.702 | 0.248 |
| observed, full | NR-T3 | 0.896 | 0.778 | 0.253 |
| observed, full | R-T2 | 0.909 | 0.674 | 0.237 |
| observed, full | NR-T2 | 0.891 | 0.762 | 0.238 |
| observed, window | HC-T1 | 0.747 | 0.737 | -0.080 |
| observed, window | MDD-T1 | 0.760 | 0.720 | -0.081 |
| observed, window | R-T3 | 0.838 | 0.462 | -0.086 |
| observed, window | NR-T3 | 0.807 | 0.570 | -0.084 |
| observed, window | R-T2 | 0.825 | 0.461 | -0.088 |
| observed, window | NR-T2 | 0.818 | 0.549 | -0.085 |

**Table S4** Precision / recall of the cluster-robust Layer 1 by scenario and network (20 replications per scenario)

| Scenario | HC-T1 | MDD-T1 | R-T2 | NR-T2 | R-T3 | NR-T3 |
|:------------------------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| correct | 0.911 / 0.918 | 0.879 / 0.885 | 0.916 / 0.699 | 0.904 / 0.769 | 0.938 / 0.723 | 0.911 / 0.809 |
| initdiag | 0.936 / 0.914 | 0.915 / 0.884 | 0.932 / 0.713 | 0.908 / 0.777 | 0.930 / 0.708 | 0.918 / 0.800 |
| tanh | 0.945 / 0.660 | 0.947 / 0.608 | 0.960 / 0.260 | 0.948 / 0.372 | 0.949 / 0.295 | 0.943 / 0.412 |
| lag2 | 0.800 / 0.679 | 0.795 / 0.650 | 0.822 / 0.405 | 0.806 / 0.477 | 0.825 / 0.472 | 0.801 / 0.561 |
| glv | 0.885 / 0.898 | 0.786 / 0.861 | 0.902 / 0.664 | 0.825 / 0.720 | 0.897 / 0.686 | 0.829 / 0.763 |
| t3 | 0.923 / 0.913 | 0.862 / 0.877 | 0.902 / 0.663 | 0.905 / 0.763 | 0.914 / 0.701 | 0.886 / 0.783 |
| driver | 0.870 / 0.916 | 0.825 / 0.877 | 0.901 / 0.706 | 0.877 / 0.766 | 0.921 / 0.737 | 0.863 / 0.814 |
| hetero | 0.930 / 0.847 | 0.897 / 0.793 | 0.930 / 0.573 | 0.910 / 0.642 | 0.958 / 0.585 | 0.911 / 0.670 |
| nonadd | 0.895 / 0.847 | 0.892 / 0.803 | 0.918 / 0.593 | 0.890 / 0.658 | 0.916 / 0.629 | 0.901 / 0.717 |
| weak | 0.938 / 0.794 | 0.940 / 0.750 | 0.948 / 0.470 | 0.936 / 0.570 | 0.922 / 0.516 | 0.940 / 0.619 |
| combined | 0.852 / 0.703 | 0.758 / 0.650 | 0.871 / 0.387 | 0.812 / 0.472 | 0.914 / 0.435 | 0.811 / 0.523 |

**Table S5** Perturbation score against the long-run response of the generating VAR: mean Spearman correlation across nodes over 30 replications, for every weighting, Laplacian and pseudoinverse tolerance, with mean rank and condition number of the Laplacian

| Variant (weights, Laplacian, tolerance) | HC-T1 | MDD-T1 | R-T3 | NR-T3 |
|:------------------------|:---------:|:---------:|:---------:|:---------:|
| TE, out, 1e-15 | -0.15 | -0.20 | 0.36 | -0.25 |
| TE, out, 1e-08 | 0.24 | 0.20 | 0.38 | 0.27 |
| TE, out, 0.0001 | 0.24 | 0.20 | 0.38 | 0.27 |
| TE, in, 1e-15 | 0.12 | 0.14 | 0.16 | 0.06 |
| TE, in, 1e-08 | 0.12 | 0.14 | 0.15 | 0.06 |
| TE, in, 0.0001 | 0.12 | 0.14 | 0.15 | 0.06 |
| TE, sym, 1e-15 | -0.19 | -0.18 | -0.00 | -0.16 |
| TE, sym, 1e-08 | -0.19 | -0.18 | -0.00 | -0.16 |
| TE, sym, 0.0001 | -0.19 | -0.18 | -0.00 | -0.16 |
| coef, out, 1e-15 | 0.15 | 0.00 | 0.35 | 0.13 |
| coef, out, 1e-08 | 0.23 | 0.18 | 0.37 | 0.31 |
| coef, out, 0.0001 | 0.23 | 0.18 | 0.37 | 0.31 |
| coef, in, 1e-15 | 0.07 | 0.11 | 0.13 | 0.06 |
| coef, in, 1e-08 | 0.07 | 0.09 | 0.10 | 0.05 |
| coef, in, 0.0001 | 0.07 | 0.09 | 0.10 | 0.05 |
| coef, sym, 1e-15 | -0.24 | -0.25 | -0.03 | -0.23 |
| coef, sym, 1e-08 | -0.24 | -0.24 | -0.03 | -0.19 |
| coef, sym, 0.0001 | -0.24 | -0.24 | -0.03 | -0.19 |
| TE_maxnorm, out, 1e-15 | -0.15 | -0.20 | 0.36 | -0.25 |
| TE_maxnorm, out, 1e-08 | 0.24 | 0.20 | 0.38 | 0.26 |
| TE_maxnorm, out, 0.0001 | 0.24 | 0.20 | 0.38 | 0.26 |
| TE_maxnorm, in, 1e-15 | 0.12 | 0.14 | 0.16 | 0.06 |
| TE_maxnorm, in, 1e-08 | 0.12 | 0.14 | 0.15 | 0.06 |
| TE_maxnorm, in, 0.0001 | 0.12 | 0.14 | 0.15 | 0.06 |
| TE_maxnorm, sym, 1e-15 | -0.19 | -0.18 | 0.00 | -0.16 |
| TE_maxnorm, sym, 1e-08 | -0.19 | -0.18 | 0.00 | -0.16 |
| TE_maxnorm, sym, 0.0001 | -0.19 | -0.18 | 0.00 | -0.16 |
| oracle_B, out, 1e-15 | 0.17 | 0.19 | 0.17 | 0.05 |
| oracle_B, out, 1e-08 | 0.19 | 0.19 | 0.19 | 0.15 |
| oracle_B, out, 0.0001 | 0.19 | 0.19 | 0.19 | 0.15 |
| oracle_B, in, 1e-15 | 0.03 | 0.05 | 0.08 | -0.05 |
| oracle_B, in, 1e-08 | -0.02 | 0.02 | 0.07 | -0.06 |
| oracle_B, in, 0.0001 | -0.02 | 0.02 | 0.07 | -0.06 |
| oracle_B, sym, 1e-15 | -0.35 | -0.34 | -0.37 | -0.42 |
| oracle_B, sym, 1e-08 | -0.33 | -0.34 | -0.34 | -0.42 |
| oracle_B, sym, 0.0001 | -0.33 | -0.34 | -0.34 | -0.42 |
| VAR_estimated | 0.40 | 0.37 | 0.38 | 0.51 |
| Laplacian rank (signed TE, out) | 47.5 | 48.1 | 39.9 | 44.5 |
| Condition number (signed TE, out) | 181 | 241 | 68 | 120 |

**Table S6** Descriptors of the size-matched networks (200 subjects per group), mean (SD) over 30 replications

| Descriptor | HC-T1 | HC-T2 | HC-T3 | MDD-T1 | R-T2 | NR-T2 | R-T3 | NR-T3 |
|:------------------------|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|:---------:|
| Edges | 61.467 (7.619) | 61.833 (9.886) | 61.567 (8.118) | 61.000 (8.582) | 59.967 (10.077) | 59.367 (9.279) | 60.100 (9.546) | 70.100 (8.450) |
| Precision | 0.937 (0.045) | 0.902 (0.070) | 0.941 (0.044) | 0.920 (0.059) | 0.908 (0.081) | 0.915 (0.049) | 0.940 (0.041) | 0.928 (0.053) |
| Recall | 0.580 (0.067) | 0.560 (0.078) | 0.585 (0.076) | 0.554 (0.069) | 0.534 (0.069) | 0.536 (0.078) | 0.557 (0.076) | 0.582 (0.071) |
| Positive (%) | 59.218 (5.759) | 58.008 (6.244) | 58.984 (6.435) | 58.148 (5.725) | 64.348 (6.942) | 65.524 (6.565) | 57.934 (4.941) | 69.140 (5.885) |
| Total H1 | 1.564 (0.605) | 1.546 (0.796) | 1.565 (0.611) | 0.906 (0.406) | 1.024 (0.498) | 0.999 (0.575) | 1.218 (0.602) | 1.832 (0.707) |
| H1 z vs null | 3.963 (1.778) | 3.645 (1.680) | 3.987 (1.711) | 2.645 (1.967) | 3.408 (2.131) | 3.062 (1.898) | 3.725 (1.903) | 3.545 (1.384) |
| H1 (log distance) | 556.243 (227.100) | 538.978 (246.511) | 531.074 (210.893) | 471.440 (206.513) | 505.337 (228.976) | 480.243 (244.538) | 500.511 (248.277) | 683.541 (217.040) |
| W2 to reference | 0.185 (0.081) | 0.161 (0.053) | 0.172 (0.055) | 0.233 (0.087) | 0.180 (0.069) | 0.189 (0.066) | 0.192 (0.063) | 0.221 (0.067) |
| Frustration (exact) | 0.072 (0.025) | 0.074 (0.026) | 0.071 (0.027) | 0.064 (0.022) | 0.072 (0.032) | 0.068 (0.029) | 0.063 (0.022) | 0.087 (0.028) |
| Frustration (multigraph) | 0.080 (0.025) | 0.080 (0.028) | 0.077 (0.026) | 0.074 (0.024) | 0.080 (0.031) | 0.079 (0.031) | 0.070 (0.025) | 0.097 (0.028) |
| Frustration (greedy) | 0.096 (0.032) | 0.097 (0.027) | 0.099 (0.024) | 0.089 (0.026) | 0.096 (0.031) | 0.093 (0.034) | 0.085 (0.029) | 0.112 (0.031) |
| AUC^R | 0.220 (0.050) | 0.264 (0.067) | 0.274 (0.068) | 0.243 (0.083) | 0.252 (0.074) | 0.252 (0.071) | 0.267 (0.070) | 0.302 (0.072) |
| Δfrag | 0.115 (0.147) | -0.074 (0.238) | -0.103 (0.192) | 0.016 (0.292) | -0.058 (0.239) | -0.084 (0.252) | -0.129 (0.249) | -0.124 (0.216) |
| Directed 3-cycles | 0.767 (0.858) | 0.867 (0.819) | 0.700 (0.877) | 0.833 (0.913) | 0.833 (1.147) | 0.767 (0.935) | 0.900 (1.094) | 2.667 (1.493) |
| Directed 4-cycles | 0.867 (0.860) | 0.933 (1.015) | 0.767 (0.774) | 0.800 (0.887) | 0.900 (1.029) | 1.033 (1.402) | 0.867 (1.074) | 3.867 (2.080) |
| Directed 5-cycles | 0.800 (1.031) | 1.067 (1.818) | 0.600 (0.894) | 0.600 (1.070) | 0.467 (0.819) | 0.700 (1.179) | 0.600 (0.932) | 3.767 (2.388) |
| Loop arcs recovered | 0.249 (0.033) | 0.249 (0.033) | 0.246 (0.031) | 0.249 (0.033) | 0.249 (0.033) | 0.256 (0.047) | 0.254 (0.036) | 0.982 (0.033) |

**Table S7** Signed-motif enrichment over 30 replications (null preserving positive and negative node degrees, 100 null networks, BH q < 0.10). Most reproducible disease-specific classes (fraction of replications in which the class was enriched in the network and in no control network), and direct comparison of class z scores between MDD and controls at T1

| Network | Class code (six arcs, signed) | Fraction of replications |
|:------------------------|:---------:|:---------:|
| NR-T3 | (-1, 0, 1, 0, 1, 1) | 0.30 |
| NR-T3 | (-1, 0, -1, 0, 0, 1) | 0.13 |
| NR-T3 | (-1, 1, 1, 0, 0, 1) | 0.13 |
| MDD-T1 | (-1, 0, 1, 0, 0, 1) | 0.13 |
| NR-T2 | (-1, 0, 1, 0, 0, 1) | 0.13 |
| MDD-T1 | (-1, -1, 0, 1, 0, 1) | 0.10 |
| R-T3 | (0, 0, 1, 0, 1, 1) | 0.10 |
| R-T2 | (-1, 1, -1, 1, 0, 0) | 0.10 |

| Class code | Replications compared | Mean z difference (MDD − HC) | p | p (Holm) |
|:------------------------|:---------:|:---------:|:---------:|:---------:|
| (-1, 0, 0, 0, 1, -1) | 15 | -0.96 | 0.007 | 0.026 |
| (-1, -1, 0, 0, 0, 1) | 21 | 0.17 | 0.634 | 1.000 |
| (0, 0, 1, 0, 1, 1) | 26 | 0.08 | 0.672 | 1.000 |
| (-1, 1, 0, 1, 0, 0) | 15 | -0.00 | 0.999 | 1.000 |

**Table S8** External directed-connectivity comparators on the microbiome benchmark (10 replications of the main scenario, full-size networks): precision, recall and conditional sign accuracy against the latent coupling graph

| Network | cTE_cluster | lasso_VAR | bivariate_GC | PCMCI |
|:------------------------|:---------:|:---------:|:---------:|:---------:|
| HC-T1 | 0.911 / 0.901 / 1.000 | 0.654 / 0.996 / 1.000 | 0.666 / 0.971 / 1.000 | 0.916 / 0.957 / 1.000 |
| MDD-T1 | 0.875 / 0.871 / 1.000 | 0.604 / 0.981 / 1.000 | 0.577 / 0.952 / 0.999 | 0.888 / 0.926 / 1.000 |
| R-T3 | 0.944 / 0.686 / 1.000 | 0.653 / 0.949 / 1.000 | 0.760 / 0.805 / 1.000 | 0.909 / 0.750 / 1.000 |
| NR-T3 | 0.896 / 0.756 / 1.000 | 0.605 / 0.937 / 1.000 | 0.729 / 0.840 / 1.000 | 0.912 / 0.805 / 1.000 |

**Table S9** Case-control classification, main scenario (8 replications): every feature family and learner. AUROC on the internal test partition and on SC2 (site-level recentering; for abundance also without correction), sensitivity and specificity at the frozen training-derived threshold and at the test-optimized threshold (descriptive only)

| Family, learner | AUROC test | AUROC SC2 | AUROC SC2, no correction | Sens / spec, frozen | Sens / spec, test-optimized |
|:------------------------|:---------:|:---------:|:---------:|:---------:|:---------:|
| network, lr | 0.561 | 0.559 | -- | 0.477 / 0.611 | 0.539 / 0.603 |
| network, xgb | 0.586 | 0.576 | -- | 0.428 / 0.689 | 0.533 / 0.653 |
| network, rf | 0.590 | 0.575 | -- | 0.486 / 0.621 | 0.529 / 0.651 |
| static_signed, lr | 0.572 | 0.538 | -- | 0.393 / 0.703 | 0.483 / 0.678 |
| static_signed, xgb | 0.634 | 0.610 | -- | 0.534 / 0.647 | 0.581 / 0.675 |
| static_signed, rf | 0.629 | 0.615 | -- | 0.543 / 0.634 | 0.671 / 0.559 |
| static_unsigned, lr | 0.644 | 0.637 | -- | 0.464 / 0.726 | 0.568 / 0.690 |
| static_unsigned, xgb | 0.613 | 0.626 | -- | 0.536 / 0.604 | 0.580 / 0.632 |
| static_unsigned, rf | 0.619 | 0.631 | -- | 0.533 / 0.629 | 0.583 / 0.627 |
| abundance, lr | 0.978 | 0.974 | 0.974 | 0.913 / 0.923 | 0.936 / 0.923 |
| abundance, xgb | 0.975 | 0.972 | 0.969 | 0.899 / 0.934 | 0.925 / 0.933 |
| abundance, rf | 0.972 | 0.971 | 0.966 | 0.900 / 0.926 | 0.903 / 0.939 |
| network+abundance, lr | 0.975 | 0.969 | -- | 0.898 / 0.921 | 0.922 / 0.922 |
| network+abundance, xgb | 0.976 | 0.973 | -- | 0.897 / 0.934 | 0.930 / 0.926 |
| network+abundance, rf | 0.973 | 0.969 | -- | 0.905 / 0.915 | 0.909 / 0.933 |
| network_no_TE, lr | 0.511 | 0.485 | -- | 0.414 / 0.600 | 0.430 / 0.643 |
| network_no_TE, xgb | 0.542 | 0.529 | -- | 0.492 / 0.562 | 0.648 / 0.482 |
| network_no_SPTS, lr | 0.571 | 0.576 | -- | 0.385 / 0.707 | 0.590 / 0.553 |
| network_no_SPTS, xgb | 0.597 | 0.577 | -- | 0.533 / 0.621 | 0.526 / 0.672 |
| network_no_TDA, lr | 0.565 | 0.560 | -- | 0.345 / 0.749 | 0.488 / 0.661 |
| network_no_TDA, xgb | 0.593 | 0.575 | -- | 0.500 / 0.619 | 0.467 / 0.725 |
| network_no_MOTIF, lr | 0.565 | 0.556 | -- | 0.465 / 0.630 | 0.466 / 0.682 |
| network_no_MOTIF, xgb | 0.588 | 0.575 | -- | 0.445 / 0.695 | 0.469 / 0.717 |
| fprau | 0.876 | 0.881 | -- | -- / -- | -- / -- |
| shannon | 0.524 | 0.498 | -- | -- / -- | -- / -- |
