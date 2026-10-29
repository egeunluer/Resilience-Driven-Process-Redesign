# Reward Function 2 (revised): Signed Performance Area around P*

## 1. Setup

Let $\phi(t \mid e_j)$ be the observed PPI under shock $e_j$ (average lead time, lower is better). It is computed over all cases, including those still in progress, so the signal is not delayed until trace completion.

Let $P^*$ be the acceptable performance level, defined as a high percentile (e.g. the 95th) of the PPI in shock-free operation instead of the maximum. This keeps the reward sensitive to mild shocks.

Deviation above $P^*$ (loss) and its mirror below $P^*$ (headroom):

$$d(t) = \max\{0,\ \phi(t \mid e_j) - P^*\}$$

$$h(t) = \max\{0,\ P^* - \phi(t \mid e_j)\}$$

Optionally, $h(t)$ is capped at $h_{\max}$ (e.g. $h_{\max} = P^* - \bar\phi_{\text{normal}}$), see Section 5.

## 2. SMDP formulation

Decision epochs $k = 0, 1, 2, \dots$ map to continuous simulation times $t_k = M(k)$ with $t_{k-1} < t_k$. The sojourn time is

$$\tau_k = t_k - t_{k-1}$$

## 3. Interval quantities

$$D_k = \int_{t_{k-1}}^{t_k} d(t)\,dt, \qquad H_k = \int_{t_{k-1}}^{t_k} h(t)\,dt$$

If $\phi$ cannot be logged between decisions, use the trapezoid approximation, e.g. $D_k \approx \tau_k \cdot \frac{d(t_{k-1}) + d(t_k)}{2}$, and the same for $H_k$.

Cumulative performance loss:

$$B_k = B_{k-1} + D_k, \quad B_0 = 0$$

$B_K$ is the CPL of the episode and matches the resilience-loss integral $RL = \int d(t)\,dt$.

## 4. Normalization and success criterion

Let $\text{CPL}_{\text{base}}$ be the mean CPL of a non-adaptive policy (e.g. FIFO) under the same shock $e_j$, averaged over simulation replications. The target is

$$\text{CPL}_{\max} = p \cdot \text{CPL}_{\text{base}}, \quad p \in (0, 1]$$

$\text{CPL}_{\text{base}}$ only normalizes the reward. $\text{CPL}_{\max}$ is used only for evaluation. Neither acts as a switch or constraint in the reward.

## 5. Reward

Above $P^*$ the agent pays for the exact loss area accumulated since the previous decision. Below $P^*$ it earns the mirrored area, computed the same way (distance to $P^*$ times sojourn time):

$$r_k = \frac{-D_k + \varepsilon\,H_k}{\text{CPL}_{\text{base}}}$$

With $\varepsilon = 1$ and no cap this is the fully symmetric version:

$$r_k = \frac{1}{\text{CPL}_{\text{base}}}\int_{t_{k-1}}^{t_k} \big(P^* - \phi(t)\big)\,dt$$

No further shaping terms are used.

**Caveat on symmetry.** With $\varepsilon = 1$ and no cap, $r_k$ is linear in $\phi$, so the agent maximizes $\int (P^* - \phi)\,dt$, i.e. it minimizes the time-integral of the PPI over the episode. The truncation at $P^*$ disappears, and this comes close to plain cycle-time optimization, which is what the resilience objective is meant to differ from. Two ways to keep a distinction:

- $\varepsilon < 1$ (e.g. 0.1–0.5), so a loss costs more than the same-sized headroom earns.
- Cap $h(t)$ at $h_{\max}$, so there is no credit for beating normal operation.

Treat $\varepsilon$ and the cap as ablation parameters, with $\varepsilon = 1$, uncapped as the pure symmetric variant.

## 6. Training and evaluation conditions

- Fixed simulated-time horizon (or fixed arrival count) per episode, so ending an episode early cannot raise the return.
- Use $\gamma = 1$, or very close, within an episode.
- Evaluation criterion: $u = B_K / \text{CPL}_{\max} \le 1$, i.e. the agent suffers at most $p \cdot \text{CPL}_{\text{base}}$.
- No Lagrange multiplier, no budget-utilization term in the observation, no trace-completion reward.
- For higher-is-better PPIs (e.g. throughput), swap the signs inside $d(t)$ and $h(t)$.
- Optional: clip $d(t)$ at $|P^* - \phi_{\text{fix}}|$ to keep the per-step reward bounded.

## 7. Reported metrics (not part of the reward)

- CPL ratio $B_K / \text{CPL}_{\text{base}}$
- Time-to-recovery (first time after the shock that $\phi \le P^*$)
- Maximum deviation $\max_t d(t)$
- Mean cycle time under normal conditions (normal-state cost of resilience)
- The same metrics for a baseline agent trained with the Meneghello et al. (2024) cycle-time reward, which shows how the resilience-trained agent differs

## 8. Planned ablations

- $\varepsilon \in \{0, 0.1, 0.5, 1\}$, with and without the cap $h_{\max}$.
- Percentile for $P^*$ (e.g. 90th vs. 95th vs. maximum).
- This reward vs. the Lagrangian extension below.

## 9. Documented extension (not core)

If cycle time should be the objective with CPL as a constraint, restore the Lagrangian form:

$$r_k = r_k^{\text{cycle time}} - \lambda\,\frac{D_k}{\text{CPL}_{\max}}, \qquad \lambda \leftarrow \max\{0,\ \lambda + \eta_\lambda(\bar u_K - 1)\}, \quad \eta_\lambda \ll \eta_\theta$$

with $\lambda$ updated once per batch of episodes, $\lambda_0 > 0$, and $u_k$ added to the observation.