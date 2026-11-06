# Training results

Successful trained models, one entry each. Scores are on the benchmark episodes:
paired, every policy on the same episodes, with 95% intervals.

- **Objective** = Loss ratio + Cost ratio. Do-nothing = 1; lower is better.
- **Loss ratio** = 1 + (Loss − Loss of do-nothing on the same episode) / Loss_base.
- **Cost ratio** = Σκ / N.

---

## 1. BPIC 2017: per-case recovery agent

| | |
|---|---|
| Event log | BPI Challenge 2017 (full log), top-3 worker pool (16 workers) |
| Shock types | D1 offer dispatch outage, D2 telephony outage, D3 missing-documents surge (level 2, 72–120 h) |
| Run | `Modules 2-3/RIMS_DRL-main/tmp_loan_training/loan_ppo_w1_s0_case_b0.5_shadow_fresh_100k/` |
| Model | `model_final.zip` (104,436 steps) |
| Training | from scratch, training seed 0, learning rate 3·10⁻⁴, entropy 0.01, 6 environments, β = 0.5, twin on, shadow price D1 0 / D2 0 / D3 1.19 |
| Code | commit `088871d` (with the shadow prices restored to the values above) |
| Date | 2026-11-06 |
| Evaluation score | 0.36 (24 episodes, seeds 1000+) |
| Benchmark | 16 episodes per shock type, seeds 200–215, deterministic policy |

### Overall (D1–D3)

| Policy | Objective | Loss ratio | Cost ratio | vs do-nothing |
|---|---|---|---|---|
| do-nothing | 1.000 | 1.000 | 0.000 | — |
| always retry | 0.758 | 0.742 | 0.016 | −0.242 ± 0.109 |
| always reassign | 0.923 | 0.866 | 0.057 | −0.077 ± 0.045 |
| always substitute | 0.772 | 0.665 | 0.108 | −0.228 ± 0.143 |
| always escalate | 0.872 | 0.846 | 0.026 | −0.128 ± 0.050 |
| greedy | 0.785 | 0.768 | 0.016 | −0.215 ± 0.068 |
| rule (urgent < 7 d) | 0.371 | 0.302 | 0.068 | −0.629 ± 0.168 |
| rule, best threshold (1 d) | 0.359 | 0.304 | 0.055 | −0.641 ± 0.163 |
| **DRL agent** | **0.303** | **0.229** | 0.074 | **−0.697 ± 0.151** |

Best baseline in each shock type, combined: 0.357.

### Per shock type

**D1: offer dispatch outage**

| Policy | Objective | Loss ratio | Cost ratio | vs do-nothing |
|---|---|---|---|---|
| do-nothing | 1.000 | 1.000 | 0.000 | — |
| always retry | 1.000 | 1.000 | 0.000 | +0.000 ± 0.000 |
| always reassign | 1.000 | 1.000 | 0.000 | +0.000 ± 0.000 |
| always substitute | 0.242 | 0.136 | 0.106 | −0.758 ± 0.225 |
| always escalate | 0.919 | 0.900 | 0.019 | −0.081 ± 0.037 |
| greedy | 0.572 | 0.536 | 0.037 | −0.428 ± 0.117 |
| rule (7 d) | 0.257 | 0.145 | 0.112 | −0.743 ± 0.228 |
| rule (1 d) | 0.237 | 0.145 | 0.092 | −0.763 ± 0.216 |
| **DRL agent** | **0.226** | 0.153 | **0.073** | **−0.774 ± 0.207** |

**D2: telephony outage**

| Policy | Objective | Loss ratio | Cost ratio | vs do-nothing |
|---|---|---|---|---|
| do-nothing | 1.000 | 1.000 | 0.000 | — |
| always retry | 1.000 | 1.000 | 0.000 | +0.000 ± 0.000 |
| always reassign | 1.000 | 1.000 | 0.000 | +0.000 ± 0.000 |
| always substitute | 1.075 | 0.858 | 0.217 | +0.075 ± 0.178 |
| always escalate | 0.980 | 0.969 | 0.011 | −0.020 ± 0.024 |
| greedy | 0.993 | 0.991 | 0.002 | −0.007 ± 0.028 |
| rule (7 d) | 0.965 | 0.953 | 0.013 | −0.035 ± 0.040 |
| rule (1 d) | 0.970 | 0.961 | 0.009 | −0.030 ± 0.040 |
| **DRL agent** | **0.792** | **0.714** | 0.078 | **−0.208 ± 0.125** |

**D3: missing-documents surge**

| Policy | Objective | Loss ratio | Cost ratio | vs do-nothing |
|---|---|---|---|---|
| do-nothing | 1.000 | 1.000 | 0.000 | — |
| always retry | 0.273 | 0.225 | 0.048 | −0.727 ± 0.147 |
| always reassign | 0.769 | 0.597 | 0.172 | −0.231 ± 0.101 |
| always substitute | 1.000 | 1.000 | 0.000 | +0.000 ± 0.000 |
| always escalate | 0.718 | 0.670 | 0.049 | −0.282 ± 0.108 |
| greedy | 0.789 | 0.778 | 0.011 | −0.211 ± 0.081 |
| rule (7 d) | −0.111 | −0.191 | 0.080 | −1.111 ± 0.241 |
| rule (1 d) | **−0.130** | −0.195 | 0.065 | −1.130 ± 0.204 |
| **DRL agent** | −0.108 | −0.179 | 0.072 | −1.108 ± 0.221 |

### DRL agent minus each baseline (same episodes; negative = agent better)

| Baseline | D1 | D2 | D3 | Overall |
|---|---|---|---|---|
| do-nothing | **−0.774 ± 0.207** | **−0.208 ± 0.125** | **−1.108 ± 0.221** | **−0.697 ± 0.151** |
| always retry | **−0.774 ± 0.207** | **−0.208 ± 0.125** | **−0.381 ± 0.112** | **−0.454 ± 0.110** |
| always substitute | −0.016 ± 0.032 | **−0.283 ± 0.091** | **−1.108 ± 0.221** | **−0.469 ± 0.154** |
| greedy | **−0.347 ± 0.106** | **−0.201 ± 0.124** | **−0.897 ± 0.188** | **−0.481 ± 0.118** |
| rule (7 d) | −0.031 ± 0.038 | **−0.173 ± 0.095** | +0.003 ± 0.100 | **−0.067 ± 0.051** |
| rule (1 d, best tuned) | −0.011 ± 0.016 | **−0.177 ± 0.100** | +0.022 ± 0.076 | **−0.055 ± 0.048** |

Bold: the 95% interval excludes 0.
