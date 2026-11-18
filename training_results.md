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

---

## 2. BPI 2012: per-case recovery agent

Not better than the rule overall. D1 and D2 are on par with the rule, D3 is clearly worse (see *Why D3 fails*).

| | |
|---|---|
| Event log | BPI Challenge 2012, top-3 worker pool |
| Shock types | same as entry 1: D1 offer dispatch outage, D2 telephony outage, D3 missing-documents surge (level 2, 72–120 h). D1 substitute = online channel (assumed, option A) |
| Calibration | common Loss_base for D1–D3 (0.0339, mean of the three scenario values over 64 seeds), no shadow price |
| Run | `Modules 2-3/RIMS_DRL-main/tmp_loan_training/bpi2012_loan_ppo_w1_s0_case_b0.5_noprice_300k_lr1e-4/` |
| Model | `checkpoints/model_266106_steps.zip` (266,106 steps; best evaluation score of the run) |
| Training | from scratch to 204,666 steps at learning rate 3·10⁻⁴ (`..._noprice_100k`, `..._noprice_200k`), then at 1·10⁻⁴ (run continued to 307,314 steps; the 266k checkpoint is used). Training seed 0, entropy 0.01, 6 environments, β = 0.5, twin on, cycle D1, D2, D3 |
| Code | commit `5ecbcd4` (the `train_order` option added since is not used) |
| Date | 2026-11-18 |
| Evaluation score | 0.74 at 266k steps (24 episodes, seeds 1000+; 0.83 at 235k, 0.79 at 297k) |
| Benchmark | 64 episodes per shock type, seeds 200–263, deterministic policy |

### Overall (D1–D3)

| Policy | Objective | Loss ratio | Cost ratio | vs do-nothing |
|---|---|---|---|---|
| do-nothing | 1.000 | 1.000 | 0.000 | — |
| greedy | 0.678 | 0.668 | 0.010 | −0.322 ± 0.080 |
| **rule (urgent < 7 d)** | **0.308** | **0.206** | 0.102 | **−0.692 ± 0.141** |
| DRL agent, 202k steps | 0.556 | 0.411 | 0.145 | −0.444 ± 0.132 |
| **DRL agent (266k)** | 0.552 | 0.404 | 0.148 | −0.448 ± 0.136 |

The always-* baselines were benchmarked on 16 episodes only (seeds 200–215, `bpi2012_bench_baselines_common_s200x16.log`): always retry 0.925, always reassign 0.866, always substitute 0.655, always escalate 0.959 (rule 0.175 and greedy 0.556 on the same 16 episodes).

### Per shock type

**D1: offer dispatch outage**

| Policy | Objective | Loss ratio | Cost ratio | vs do-nothing |
|---|---|---|---|---|
| do-nothing | 1.000 | 1.000 | 0.000 | — |
| greedy | 0.676 | 0.668 | 0.008 | −0.324 ± 0.111 |
| rule | 0.372 | **0.201** | 0.172 | −0.628 ± 0.195 |
| DRL agent, 202k | 0.401 | 0.282 | 0.118 | −0.599 ± 0.192 |
| **DRL agent** | **0.359** | 0.243 | **0.116** | −0.641 ± 0.177 |

**D2: telephony outage**

| Policy | Objective | Loss ratio | Cost ratio | vs do-nothing |
|---|---|---|---|---|
| do-nothing | 1.000 | 1.000 | 0.000 | — |
| greedy | 0.785 | 0.779 | 0.006 | −0.215 ± 0.124 |
| rule | 0.497 | 0.456 | 0.041 | −0.503 ± 0.210 |
| DRL agent, 202k | **0.394** | **0.288** | 0.106 | −0.606 ± 0.214 |
| **DRL agent** | 0.424 | 0.307 | 0.118 | −0.576 ± 0.210 |

**D3: missing-documents surge**

| Policy | Objective | Loss ratio | Cost ratio | vs do-nothing |
|---|---|---|---|---|
| do-nothing | 1.000 | 1.000 | 0.000 | — |
| greedy | 0.574 | 0.557 | 0.017 | −0.426 ± 0.159 |
| **rule** | **0.054** | **−0.038** | 0.092 | **−0.946 ± 0.248** |
| DRL agent, 202k | 0.873 | 0.662 | 0.210 | −0.127 ± 0.249 |
| DRL agent | 0.873 | 0.662 | 0.210 | −0.127 ± 0.249 |

### DRL agent minus each baseline (same episodes; negative = agent better)

| Baseline | D1 | D2 | D3 | Overall |
|---|---|---|---|---|
| do-nothing | **−0.641 ± 0.177** | **−0.576 ± 0.210** | −0.127 ± 0.249 | **−0.448 ± 0.136** |
| greedy | **−0.317 ± 0.145** | **−0.360 ± 0.207** | **+0.299 ± 0.281** | −0.126 ± 0.136 |
| rule (7 d) | −0.013 ± 0.118 | −0.072 ± 0.191 | **+0.819 ± 0.247** | **+0.244 ± 0.126** |
| agent, 202k | −0.042 ± 0.089 | +0.030 ± 0.118 | 0.000 | −0.004 ± 0.050 |
| agent, 307k (final) | −0.030 ± 0.076 | +0.045 ± 0.129 | 0.000 | +0.005 ± 0.052 |

Bold: the 95% interval excludes 0. Paired t-test / Wilcoxon p-values agree with the intervals; agent vs greedy overall: p = 0.073 / 0.086 (not significant). Agent vs rule: D1 p = 0.83, D2 p = 0.46, D3 and overall p < 0.001.

### Action mix per episode (agent vs rule)

| Shock | Rule | DRL agent |
|---|---|---|
| D1 | substitute 113.6, escalate 21.2 | substitute 79.3, do-nothing 46.4, escalate 12.5 |
| D2 | do-nothing 332.8, substitute 49.2, escalate 22.7 | do-nothing 265.1, substitute 193.4, escalate 31.7 |
| D3 | retry 56.3, reassign 7.9, escalate 6.9 | reassign 56.9, escalate 17.8, retry 0 |

### Why D3 fails

The agent never retries in D3; it reassigns almost every case. In BPI 2012 the per-case reward itself ranks reassign above retry (reward-level D3 check): the reassigned case's own gain counts in full, while the delay it causes the cases it jumps ahead of only reaches the reward through the shared term (weight β, diluted and noisy).

Two tests show that more training or more D3 data does not change this:
- **D3 weighted 3× in training** (cycle D1, D2, D3, D3, D3; run `..._noprice_d3x3_100k`, 16-episode benchmark): every checkpoint made exactly the same D3 decisions as the 202k agent (reassign 56.9, escalate 15.6, retry 0).
- **100k more steps at learning rate 1·10⁻⁴** (this entry): no change against the 202k agent in any shock type; D3 decisions identical at 266k and 307k.

A reassign shadow price (charging queue jumping like the escalation price) was considered and not used: it would change the experiment's frame to obtain the wanted D3 behaviour. D3 in BPI 2012 is reported as a limitation of the per-case reward.

### Other notes
- **More training has converged.** Against the 202k agent, overall: 266k −0.004 ± 0.050, 307k −0.009 ± 0.055 (64 episodes, 307k: objective 0.547, D1 0.389, D2 0.379). The 266k checkpoint is chosen by its evaluation score, not by the benchmark.
- **The 16-episode benchmark overstated the agent in D1/D2.** On seeds 200–215 the rule scored D1 0.21 / D2 0.39; on 64 episodes 0.37 / 0.50. Use 64 episodes for 2012.
- **Escalation shadow price:** measured under the 202k agent (16 seeds), inconclusive for all three teams (D1 undefined, D2 −2.5 with interval −13.2..+8.8, D3 0.02); left at 0, as D1/D2 in entry 1.
