| Classifier | thr | TPR en | TPR ur | Δ TPR (ur−en) [95% CI] | FPR en | FPR ur | Δ FPR [95% CI] | AUC en | AUC ur | Δ AUC [95% CI] |
|---|---|---|---|---|---|---|---|---|---|---|
| protectai/deberta-v3-base-prompt-injection-v2 | 0.92 | 0.743 | 0.931 | +0.188 [0.154, 0.219] | 0.001 | 0.798 | +0.798 [0.777, 0.817] | 0.994 | 0.786 | -0.208 [-0.230, -0.187] |
| meta-llama/Llama-Prompt-Guard-2-86M | 0.5 | 0.236 | 0.179 | -0.058 [-0.093, -0.020] | 0.009 | 0.000 | -0.009 [-0.014, -0.004] | 0.849 | 0.950 | +0.101 [0.084, 0.118] |
| meta-llama/Llama-Prompt-Guard-2-22M | 0.5 | 0.107 | 0.015 | -0.092 [-0.113, -0.072] | 0.001 | 0.000 | -0.001 [-0.002, 0.000] | 0.703 | 0.863 | +0.160 [0.133, 0.187] |
| Horizon-Labs/prompt-injection-guard-base | 0.5 | 0.917 | 0.842 | -0.075 [-0.106, -0.045] | 0.001 | 0.000 | -0.001 [-0.002, 0.000] | 1.000 | 0.997 | -0.002 [-0.004, -0.001] |
| proventra/mdeberta-v3-base-prompt-injection | 0.5 | 0.738 | 0.899 | +0.161 [0.128, 0.194] | 0.303 | 0.282 | -0.021 [-0.054, 0.013] | 0.833 | 0.911 | +0.078 [0.059, 0.097] |
| patronus-studio/wolf-defender-prompt-injection | 0.5 | 0.971 | 0.966 | -0.005 [-0.021, 0.011] | 0.000 | 0.041 | +0.041 [0.032, 0.052] | 1.000 | 0.995 | -0.005 [-0.007, -0.003] |

Detection rate by attack type (edge_case not in the headline TPR):

| Classifier | adversarial en | adversarial ur | jailbreak en | jailbreak ur | edge_case en | edge_case ur |
|---|---|---|---|---|---|---|
| protectai/deberta-v3-base-prompt-injection-v2 | 0.701 | 0.886 | 0.787 | 0.963 | 0.083 | 0.888 |
| meta-llama/Llama-Prompt-Guard-2-86M | 0.095 | 0.055 | 0.388 | 0.267 | 0.075 | 0.000 |
| meta-llama/Llama-Prompt-Guard-2-22M | 0.022 | 0.035 | 0.197 | 0.000 | 0.025 | 0.000 |
| Horizon-Labs/prompt-injection-guard-base | 1.000 | 0.988 | 0.829 | 0.738 | 0.092 | 0.022 |
| proventra/mdeberta-v3-base-prompt-injection | 0.835 | 0.866 | 0.634 | 0.923 | 0.075 | 0.146 |
| patronus-studio/wolf-defender-prompt-injection | 0.970 | 0.971 | 0.972 | 0.963 | 0.158 | 0.202 |

n (unique, after within-language dedup): attacks en=1041, ur=823; benign en=1492, ur=1478.
