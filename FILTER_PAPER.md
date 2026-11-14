# Novel and known regions: causal exploration and a tighter Nash bound for shared-parameter policies

*Why the decisions that carry the edge are not learned by on-policy reinforcement learning in models that share
parameters, and how filtering by an oracle's disagreement and locking where it agrees fixes it.*

---

## Abstract

Game space splits into two kinds of region. **Known** regions are those where the policy already agrees with an
oracle: a best response at every later decision. **Novel** regions are where it does not, where an action the
policy rarely takes is worth more. The two kinds are not known to correlate: nothing a learner observes in one tells
it what the other needs.

In sequential games the edge lives in the novel regions near the leaves, and those are rarely visited. In Leduc
poker, leaf decisions are 15 % of on-policy samples but 64 % of the exploitability. After 3,000 updates of
on-policy learning the early, similar regions are solved, and the leaves carry 73 % of what remains. In full-scale
poker the leaves are about 0.5 % of samples.

We show why this happens and what fixes it. Each step below is a proof, followed by an exact measurement.

1. **The edge decomposes exactly over decisions** (Lemma 1). What a step at a decision can earn is bounded by that
   decision's own regret (Lemma 2), and a known decision has little left to earn (Lemma 3).
2. **On-policy updates drown the novel regions** (Theorem 1). A decision's own, causal share of the update is
   proportional to its occupancy. In a shared-parameter policy the share pushed onto it by other decisions is not,
   so as occupancy falls the causal share vanishes. The known regions hold about $`1-f`$ of every on-policy batch.
3. **The filter is a statistical test that partitions every batch** (Theorem 1′, Proposition 1″). Each on-policy
   candidate is either included as a pivot, because its disagreement clears the sampled quantile bar, or excluded
   from the gradient. Excluded known candidates become the lock's rows. The pivots are then distributed as occupancy
   conditioned on being novel, amplifying each novel decision by $`1/f`$, and the known regions' gradient term is
   removed rather than diluted. The bar's misclassification is confined to a DKW band of width
   $`\sqrt{\ln(2/\delta)/2m}`$. Measured in Leduc: known decisions' share of gradient rows falls from 76 % to 8 %;
   novel decisions are included at 3.9× their on-policy rate.
4. **What shared parameters push onto other decisions is not causal** (Theorem 2 and Proposition 4). A label is an interventional value, so a step is
   justified only at the decisions it intervened on. Everything else it moves is moved through shared parameters,
   in a direction uncorrelated with what those decisions need. Measured: 18–42 % of each step's movement lands on
   known decisions never intervened on, with correlation between −0.54 and +0.03 to their own improvement.
5. **The lock caps the non-causal movement** (Theorem 3). A KL lock to a periodically refreshed teacher bounds the
   known region's drift by the segment's gain over $`\beta`$. Measured: 2.5–20× less drift at the best β in every coupling class.
6. **A tighter Nash bound** (Theorem 4). Exploitability is Lipschitz in the policy's movement, so the lock caps how
   much exploitability the non-causal movement can create. The reachable ε-Nash floor drops.
   Measured: Kuhn last iterate 0.159 → 0.0003 (tabular, $`\beta=8`$); Leduc 0.558 → 0.375 (tabular) and
   3.44 → 3.06 (token-shared parameters).

These are the conditions of every large model that shares parameters across positions and depths. A sequence
model's token embeddings are shared by every occurrence of the token, so its coupling density is essentially one.
The token-shared policy class here has density 0.83–0.99.

---

## 1. Setting

A player acts at decisions $`k`$ (information sets) with actions $`A(k)`$ and policy $`\pi(\cdot\mid k)`$, with
everyone else held fixed. Write $`R(k)`$ for the chance × opponent reach of $`k`$ and $`x^{\text{in}}(k)`$ for the
player's own reach. The **occupancy** is $`\text{occ}(k) = x^{\text{in}}(k)R(k)`$, the rate at which on-policy play
samples $`k`$. All values are normalised by $`R(k)`$.

| symbol | meaning |
|---|---|
| $`Q^\pi(k,a)`$ | value of **intervening** at $`k`$ with $`a`$ — do($`a`$) — then everyone playing $`\pi`$ |
| $`Q^*(k,a)`$ | the **oracle**: do($`a`$) at $`k`$, then the player best-responds at every later decision |
| $`V(k)`$ | $`\sum_a\pi(a\mid k)Q^\pi(k,a)`$ |
| $`\ell(k)`$ | $`\max_aQ^\pi(k,a)-V(k)`$, the local regret |
| $`d(k)`$ | $`\max_aQ^*(k,a)-V(k)\ge0`$, the **disagreement** with the oracle |
| $`G_i`$ | player $`i`$'s best-response gain; NashConv $`=\sum_iG_i`$ (one player: $`V_{\text{opt}}-V`$) |

**Novel and known.** For a bar $`b`$, decision $`k`$ is **known** if $`d(k)\le b`$ and **novel** otherwise. The
bar is statistical: the $`(1-f)`$ quantile of $`d`$ over the decisions sampled on-policy.

**The method.**
1. Sample candidate decisions on-policy: $`n_u`$ from the unfiltered decisions and $`m=Kn_f`$ from the filtered ones.
2. **Include or exclude** each filtered candidate by a statistical test. Compute $`\hat b`$, the $`(1-f)`$ quantile of
   the $`m`$ sampled disagreements. The first $`n_f`$ candidates above it are **pivots**: they enter the
   policy-gradient batch, labelled with their interventional values, switched to the oracle's where it finds an
   action the policy has nearly closed.
3. Every other filtered candidate is excluded from the policy-gradient batch. Those at or below $`\hat b`$ (known)
   form the **lock set**.
4. Ascend the pivots' policy-gradient surrogate minus
   $`\beta\cdot\text{mean}_{k\in\text{lock}}\mathrm{KL}(\pi_T(\cdot\mid k)\,\|\,\pi_\theta(\cdot\mid k))`$.
5. Every $`M`$ updates, refresh the teacher $`T\leftarrow\theta`$.

**Policies.** The logits are linear in one parameter vector, $`z = \Phi e`$, with three feature maps:
- **tabular**: one feature per decision, so no parameter is shared;
- **hashed**: random sharing;
- **tokens**: each information set is a bag of its tokens (cards, every action token of the history, a bias), so
  the same token at any depth is the same parameter.

The coupling density (the share of decision pairs that share a parameter) is 0, 0.05–0.25 and 0.83–0.99
respectively. Everything is computed exactly on Kuhn poker, Leduc poker and a one-player trading ladder. Only the
choice of decisions is sampled, with 6 seeds per configuration. Code in `filtering/`.

---

## 2. Where the edge is

**Lemma 1 (the edge decomposes exactly).** Let $`a^*(k)`$ be the oracle's action and $`\rho^*(k)`$ the occupancy
of $`k`$ under the best response. Then
```math
G_i \;=\; \sum_k \rho^*(k)\,r(k),\qquad r(k)=Q^\pi(k,a^*(k))-V(k)\;\le\;\ell(k)\;\le\;d(k),
```
and $`d(k)=\ell(k)+\max_a\big(Q^*(k,a)-Q^\pi(k,a)\big)`$ at most, with $`d=\ell`$ at leaf decisions (no later
decision of the player). Also $`G_i=\sum_{k\ \text{first}}R(k)\,d(k)`$.

*Proof.* The first identity is the performance-difference identity with the best response as the comparison policy.
The second follows from $`Q^*(k,a)-Q^\pi(k,a)=R(k)^{-1}\sum_{k'\in\mathrm{ch}(k,a)}R(k')d(k')`$, which holds because
the two backups differ only through their children. ∎

So a known decision's whole subtree is within $`b`$ of a best response. A novel decision's disagreement can sit in
its subtree, which a step at the decision itself cannot fix:

**Lemma 2 (what a step can earn).** A change of the policy at $`k`$ alone changes the player's value by exactly
$`\text{occ}'(k)\langle\Delta\pi(\cdot\mid k),Q^\pi(k,\cdot)\rangle`$, at most $`\text{occ}'(k)\,\ell(k)`$.

**Lemma 3 (known decisions have little to gain).** $`\mathrm{Var}_{\pi}\,Q^\pi(k,\cdot)\le2\,\mathrm{span}_k\,\ell(k)
\le2\,\mathrm{span}_k\,d(k)`$. So a unit of policy movement in the Fisher metric changes a known decision's value by
at most $`\text{occ}(k)\sqrt{2\,\mathrm{span}_k\,b}`$ to first order.

*Proof.* With $`X=Q^\pi(k,a)-V(k)`$ under $`a\sim\pi`$: $`E X=0`$, so $`E|X|=2E X^+\le2\ell`$, and
$`\mathrm{Var}X\le\mathrm{span}\cdot E|X|`$. The first-order change is $`\langle F\Delta z,X\rangle\le\|\Delta
z\|_F\sqrt{\mathrm{Var}X}`$. ∎

The oracle gap is therefore the right selection score at the leaves, where $`d=\ell`$, and the wrong one at the
root. Filtering every decision of the one-player ladder makes it 2.1–2.7× worse than on-policy sampling
($`V_{	ext{opt}}-V`$ 0.820 vs 0.303 with hashed sharing, 1.137 vs 0.530 tabular).

**Measured: the edge sits at the leaves, and on-policy learning leaves it there** (`filtering/regions.py`, 3 seeds,
3,000 updates; Lemma 1 checked to $`10^{-15}`$):

| Leduc | leaves' share of samples | leaves' share of the edge | edge left at the leaves | NashConv |
|---|---|---|---|---|
| start (uniform policy) | 15.4 % | 64 % | — | — |
| on-policy, tabular | 12.6 % | 73 % | 1.540 | 2.122 |
| filter, tabular | 9.3 % | 85 % | 0.509 | 0.596 |
| filter + lock, tabular | 9.4 % | 85 % | **0.296** | **0.351** |
| on-policy, tokens | 15.7 % | 68 % | 3.177 | 4.541 |
| filter, tokens | 10.3 % | 26 % | **1.008** | 3.651 |
| filter + lock, tokens | 14.8 % | 44 % | 1.384 | **3.289** |

On-policy learning solves the early regions and leaves most of the edge at the leaves. The filter and lock cut the
edge left there 3–5× with tabular parameters and 2.3–3.2× with token-shared ones. (The leaves' *share* rises once the
rest is solved; the absolute column is the one that matters.)

---

## 3. Why the leaves are drowned, and how inclusion and exclusion fix it

### 3.1 On-policy updates drown rare novel decisions

**Theorem 1 (drowning).** Under on-policy sampling with $`n`$ samples per update and step $`\eta`$, the expected
logit change at a decision $`j`$ is
```math
\mathbb{E}\,\delta z_j \;=\; \eta n\Big[\;p_j\,K_{jj}\,c_j \;+\; \sum_{k\ne j}p_k\,K_{jk}\,c_k\;\Big],
\qquad p_k=\frac{\text{occ}(k)}{\sum\text{occ}},\quad K_{jk}=\Phi_j\Phi_k^\top ,
```
where $`c_k`$ is the label gradient at $`k`$.
- The first term is $`j`$'s own, **causal** update, evaluated at $`j`$. It is proportional to $`p_j`$.
- The second term is pushed onto $`j`$ through shared parameters by every other decision. It is bounded by
  $`\|K_{j,-j}\|(1-p_j)\max_k\|c_k\|`$ and does not shrink with $`p_j`$. Most of it comes from the known regions,
  which hold most of the samples: a fraction $`1-\mu_N`$ of every batch, where $`\mu_N`$ is the novel region's
  occupancy mass.

So for a novel leaf with $`p_j\to0`$ (0.5 % in full-scale poker) the ratio of causal to non-causal movement goes to
zero unless $`K_{j,-j}=0`$, i.e. unless nothing is shared. *Proof:* linearity of $`z=\Phi e`$ and of the expectation
over samples. ∎

### 3.2 The filter is a statistical test that partitions every batch

Each update draws $`m=Kn_f`$ candidates on-policy from the filtered decisions and splits them:

- **included**: a candidate whose disagreement exceeds $`\hat b`$, the $`(1-f)`$ quantile of the $`m`$ sampled
  scores, becomes a pivot (the first $`n_f`$ such, in sampling order) and contributes a policy-gradient row;
- **excluded**: every other candidate contributes **no** policy-gradient row. Those at or below the bar become lock
  rows.

A sample is never down-weighted: it is either in the gradient or out of it. Write $`F`$ for the occupancy-weighted
distribution function of $`d`$ over the filtered decisions, and $`b=F^{-1}(1-f)`$ for the population bar.

**Theorem 1′ (selection law).** If the test is exact ($`\hat b=b`$), the pivots are distributed as occupancy
*conditioned on being novel*, and the lock rows as occupancy conditioned on being known:
```math
q(k)\;=\;\frac{p_k\,\mathbb{1}[d(k)>b]}{\mu_N},\qquad
\rho_K(k)\;=\;\frac{p_k\,\mathbb{1}[d(k)\le b]}{1-\mu_N},\qquad \mu_N=1-F(b)=f .
```
Hence the expected update of the filtered batch is
```math
\mathbb{E}\,\delta z_j \;=\; \eta n_f\Big[\;\frac{p_j}{\mu_N}\,K_{jj}\,c_j \;+\; \sum_{k\ \text{novel},\,k\ne j}\frac{p_k}{\mu_N}\,K_{jk}\,c_k\;\Big]
\;+\;\text{lock term}.
```
Compared with Theorem 1:
- every novel decision's causal share is **amplified by $`1/\mu_N=1/f`$**;
- the known regions' term is **removed exactly**, not diluted;
- the known samples that were removed are not wasted: they are the lock's rows, drawn from exactly the law
  $`\rho_K`$ that Theorem 3 needs.

What remains non-causal is the spill among novel decisions, and the lock term, which only pulls the known region back
toward its teacher. *Proof:* sampling $`\propto p`$ and keeping the draws with $`d>b`$ (in order) is rejection
sampling from $`p`$ restricted to $`\{d>b\}`$; the rest is linearity. ∎

**Proposition 1″ (statistical inclusion).** The bar is estimated from $`m`$ draws. By the Dvoretzky–Kiefer–Wolfowitz
inequality, with probability at least $`1-\delta`$,
```math
\big|F(\hat b)-(1-f)\big|\;\le\;\varepsilon_m=\sqrt{\ln(2/\delta)/(2m)} .
```
On that event:
- every decision with $`F(d)>1-f+\varepsilon_m`$ clears the bar, and every decision with $`F(d)<1-f-\varepsilon_m`$
  is excluded;
- misclassification is confined to a band of occupancy mass $`2\varepsilon_m`$ around the bar;
- the share of included pivots that are actually known is at most about $`\varepsilon_m/f`$.

With discrete scores, mass tied at the bar can also enter through the fill rule; the leak is bounded by that mass.
More candidates per slot (a larger $`K`$) narrow the band as $`1/\sqrt{Kn_f}`$. *Proof:* DKW for the empirical
distribution function of $`m`$ i.i.d. draws, evaluated at the empirical quantile. ∎

**Measured: what the test includes and excludes** (`filtering/inclusion.py`, tabular policies, 3 seeds × 1,500
updates; "novel" and "known" judged at the population bar $`b`$):

| | candidates per update m | included pivots that are novel | gradient rows from known decisions | novel inclusion rate vs on-policy | lock rows that are known | sampled bar within the DKW radius (δ = 0.05) |
|---|---|---|---|---|---|---|
| Kuhn, on-policy | 3 | 13 % | 87 % | 1.0× | — | — |
| Kuhn, filter + lock | 15 | 58 % | 42 % | **4.0×** | 97 % | 99.8 % (radius 0.35; mean offset 0.11) |
| Leduc, on-policy | 16 | 24 % | 76 % | 1.0× | — | — |
| Leduc, filter + lock | 80 | **92 %** | **8 %** | **3.9×** | 98 % | 99.8 % (radius 0.15; mean offset 0.04) |

In Leduc ($`m=80`$) the test behaves as Theorem 1′ and Proposition 1″ say:
- on-policy sampling feeds known decisions into about three quarters of the gradient rows ($`\approx1-f`$); the
  filter cuts that to under a tenth;
- novel decisions are included at close to $`1/f=4`$ times their on-policy rate;
- the sampled bar sits inside the DKW radius on nearly every update.

Kuhn ($`m=15`$, many tied scores) is the small-sample regime. The radius is wide, and ties at the bar enter through
the fill rule, so 42 % of its pivots are still known. Even so, novel decisions are included at 4.0× their on-policy
rate, and the share of novel pivots rises 4.5× (13 % → 58 %). Proposition 1″ says how to close the rest of the gap:
more candidates per pivot.

**Measured: the leaves.** The filter raises the leaves' share of pivots in Leduc from 20–24 % (on-policy) to
24–31 %, though the leaves are only 9–15 % of samples.

---

## 4. Causation: the movement nobody evaluated

$`Q^\pi(k,a)`$ and $`Q^*(k,a)`$ are interventional quantities: the value of setting the action at $`k`$. The exact
value change of any step is
```math
\Delta V \;=\; \sum_k \text{occ}'(k)\,\big\langle \Delta\pi(\cdot\mid k),\,Q^\pi(k,\cdot)\big\rangle
```
(performance difference). The term of a decision is evaluated by an intervention at that decision and nowhere else.

**Theorem 2 (spill law).** A step $`\Delta e = \eta\,\Phi_P^\top c`$ taken at pivots $`P`$ changes the logits of
*every* decision by $`\delta z_k = \eta\sum_{j\in P}K_{kj}c_j`$. This is exact for linear logits, and first-order
with $`J_k=\partial z_k/\partial\theta`$ in place of $`\Phi_k`$ for any differentiable model. Over $`M`$ steps
without a lock, the known region's displacement satisfies $`\|\sum_t\delta z_K^{(t)}\|\le\eta M\kappa_{KP}\,\bar c`$
($`\kappa_{KP}=\|K_{K,P}\|`$), with equality when the pivots' field is persistent.

**Assumption U (novel and known are not known to correlate).** Given what the update observes, the known region's
advantages are uncorrelated with the direction the spill moves them:
$`\mathbb{E}\langle F_kK_{kj}c_j,\,A_k\rangle=0`$ for $`k`$ known and $`j\in P`$.

**Proposition 4 (spill is exploration without evaluation).** Under U, spill has zero expected first-order value in
the known region. By Lemma 3 its largest possible gain there is $`\text{occ}\cdot b`$, while its loss can be as large
as $`\mathrm{span}\cdot\mathrm{TV}`$. It is movement that no intervention justified.

**Measured: how much movement lands where nothing was evaluated, and whether it helps** (`filtering/causal.py`;
correlation in the Fisher metric between each decision's movement and its own advantage, weighted by occupancy ×
movement):

| | movement on pivots (corr.) | movement on **known** non-pivots (corr.) | movement on novel non-pivots (corr.) |
|---|---|---|---|
| Kuhn, tabular | 100 % (1.00) | 0 % | 0 % |
| Kuhn, hashed | 60 % (0.92) | **38 % (−0.54)** | 2 % (−0.04) |
| Kuhn, tokens | 69 % (0.98) | **30 % (+0.02)** | 1 % (−0.17) |
| Leduc, tabular | 100 % (0.91) | 0 % | 0 % |
| Leduc, hashed | 70 % (0.58) | **18 % (−0.18)** | 12 % (−0.20) |
| Leduc, tokens | 46 % (0.34) | **42 % (+0.03)** | 12 % (+0.08) |

Tabular steps move only what they evaluated. Shared steps send 18–42 % of their movement into known regions. That
movement is uncorrelated with what those regions need, or works against it. With token sharing even the pivots'
own movement falls to 0.34 correlation, because pivots spill onto each other. (Tabular pivots sit at 0.91 rather
than 1 only because swapped oracle labels differ from the on-policy advantage.)

---

## 5. The lock caps the non-causal movement

**Theorem 3 (drift cap).** Fix a segment with teacher $`T`$. Let $`\bar K(\theta)=\mathbb{E}_{k\sim\rho_K}
\mathrm{KL}(\pi_T(\cdot\mid k)\,\|\,\pi_\theta(\cdot\mid k))`$ over the lock set's sampling law $`\rho_K`$. Suppose
the pivot term ascends a potential $`S`$ (the reweighted value of the selected decisions with frozen continuation
values) and the step size satisfies the descent condition
```math
\eta\,\Big(L_S+\tfrac{\beta}{2}\,\lambda_{\max}(\Phi_K^\top\Phi_K)\Big)\;\le\;1 .
```
Then for every iterate of the segment
```math
\beta\,\bar K(\theta)\;\le\;S(\theta)-S(T)\;=:\;\bar G,\qquad
\mathbb{E}_{\rho_K}\mathrm{TV}\le\sqrt{\bar G/(2\beta)} ,
```
and $`\bar G\le\sum_{k\in P}w(k)\,d_T(k)`$, the disagreement being corrected.

*Proof.* The descent condition makes $`\Phi(\theta)=-S(\theta)+\beta\bar K(\theta)`$ non-increasing along the
segment, because the KL's Hessian in $`e`$ is $`\Phi_K^\top F\Phi_K\preceq\tfrac12\Phi_K^\top\Phi_K`$. Since
$`\bar K(T)=0`$, $`\Phi(\theta)\le\Phi(T)`$ gives the bound. Pinsker and Jensen give the TV. ∎

Three consequences:
1. The known region's drift is set by the gain being made at the pivots, divided by $`\beta`$, not by how strongly
   the pivots are coupled to it.
2. As the novel disagreement closes, $`\bar G`$ shrinks and the cap tightens.
3. Dense coupling (a large $`\lambda_{\max}(\Phi_K^\top\Phi_K)`$, as with shared token embeddings) limits the usable
   $`\beta`$. Past that limit the descent condition fails and the cap is lost.

**Measured: drift of the known region per segment, KL from the teacher** (`filtering/run.py`, $`M=100`$, 6 seeds):

| | no lock | lock β 0.5 | lock β 2 | lock β 8 |
|---|---|---|---|---|
| Kuhn, tabular | 0.061 | 0.063 | 0.015 | **0.003** |
| Kuhn, hashed | 0.050 | 0.018 | **0.008** | 0.016 |
| Kuhn, tokens | 0.181 | 0.123 | 0.091 | **0.063** |
| Leduc, tabular | 0.0087 | 0.0067 | 0.0047 | **0.0035** |
| Leduc, hashed | 0.065 | 0.043 | 0.019 | **0.009** |
| Leduc, tokens | 0.520 | 0.262 | **0.162** | 0.284 |

The lock cuts the known region's drift 2.5–20× at its best β in every class. The two exceptions to monotone decrease
(Kuhn hashed and Leduc tokens at β = 8) are the descent-condition failure Theorem 3 predicts, and they occur in the
most strongly coupled settings.

**Measured: drift against segment length** (Leduc, β = 2, 4 seeds):

| segment length M | 25 | 100 | 400 |
|---|---|---|---|
| hashed: no lock / lock | 0.013 / 0.005 | 0.081 / 0.021 | 0.336 / 0.153 |
| tokens: no lock / lock | 0.375 / 0.120 | 0.609 / 0.190 | 1.147 / 0.375 |

Locked drift grows with $`M`$ as the segment's gain $`\bar G`$ grows, as the theorem allows. The lock divides it by
2.2–3.9× at every length.

---

## 6. A tighter Nash bound

**Lemma 5 (exposure).** In a two-player zero-sum game, NashConv $`=e(x_0)+e(x_1)`$, where $`e(x_i)`$ is the best
response's value against $`x_i`$ and depends only on player $`i`$'s own strategy. If player $`i`$'s policy changes
by $`\mathrm{TV}_k`$ at each decision, then
```math
e(x_i')-e(x_i)\;\le\;2U\,\max_y\sum_k \text{occ}_{x_i,y}(k)\,\mathrm{TV}_k ,
```
with $`U`$ the payoff bound. *Proof:* the terminal distributions under $`(x_i,y)`$ and $`(x_i',y)`$ differ in TV by
at most the expected sum of per-decision TVs along the path (the simulation lemma). Apply this to the maximising
$`y`$. ∎

**Theorem 4 (tighter Nash bound).** Split a segment's change of $`x_i`$ into its moves on novel decisions and its
moves on known ones. Then
```math
e(x_i^{\,s+1})\;\le\;e(x_i^{\,s})-\Gamma_i^{\,s}+2U\,E_K^{\,s},
\qquad
E_K^{\,s}=\max_y\sum_{k\ \text{known}}\text{occ}_y(k)\,\mathrm{TV}_k ,
```
where $`\Gamma_i^s`$ is the exploitability reduction of the novel moves alone. The exposure $`E_K`$ is bounded two
ways:
```math
\text{with the lock:}\quad E_K\le\sqrt{C_K\,\bar G/(2\beta)}\qquad\qquad
\text{without:}\quad E_K\le C'\,\eta M\,\kappa_{KP}\,\bar c ,
```
with $`C_K=\max_y\sum_{k}\text{occ}_y(k)^2/\rho_K(k)`$ the mismatch between the best response's reach and the lock's
sampling law.

*Proof.* Apply Lemma 5 between the policy with only the novel moves applied and the full policy, which differ only
on known decisions. Bound $`\sum\text{occ}_y\mathrm{TV}`$ by Cauchy–Schwarz and Theorem 3 (with the lock), or by
Theorem 2 and the Lipschitz softmax (without it). ∎

**Consequence: the ε-Nash floor.** Exploitability falls as long as the novel progress $`\Gamma`$ exceeds the
exposure $`2UE_K`$. Without the lock, the exposure scales with $`\eta M\kappa_{KP}`$. Spill keeps knocking known
decisions back into novel ones, and the floor rises with coupling. With the lock, the exposure is at most
$`2U\sqrt{C_K\bar G/(2\beta)}`$. If novel progress is proportional to the disagreement being corrected
($`\Gamma\ge c\,\bar G`$), the floor scales as $`O(1/\beta)`$, up to the largest β the descent condition allows. The
tightening is largest exactly where parameters are most shared.

**Measured: NashConv after 3,000 updates** (6 seeds; iterate, with the linear average in brackets):

| | on-policy | filter | filter + lock β 0.5 | β 2 | β 8 |
|---|---|---|---|---|---|
| Kuhn, tabular | 0.364 [0.071] | 0.159 [0.018] | 0.179 [0.022] | 0.092 [0.013] | **0.0003** [0.018] |
| Kuhn, hashed | 0.333 [0.334] | 0.100 [0.067] | 0.087 [**0.055**] | **0.079** [0.070] | 0.111 [0.105] |
| Kuhn, tokens | 0.743 [0.628] | 0.337 [0.034] | **0.252** [0.031] | 0.270 [**0.025**] | 0.402 [0.120] |
| Leduc, tabular | 2.254 [1.042] | 0.558 [0.316] | 0.506 [**0.308**] | **0.375** [0.323] | 0.489 [0.360] |
| Leduc, hashed | 3.530 [2.619] | **2.013** [**1.495**] | 2.749 [1.568] | 2.574 [1.735] | 2.365 [1.738] |
| Leduc, tokens | 4.327 [3.877] | 3.443 [2.897] | 3.389 [2.945] | **3.061** [**2.454**] | 3.545 [2.850] |

Standard errors are 0.01–0.05 for Kuhn, 0.01–0.04 for Leduc tabular, and 0.2–0.6 for Leduc hashed and tokens
(`filtering/results/summary_*.json`). Reading the table:

- **Filtering novel regions** is the largest effect: 1.3–18× lower than on-policy in every cell.
- **Locking known regions** tightens the result further in Kuhn (every class), in Leduc tabular (0.558 → 0.375) and
  in Leduc tokens (3.44 → 3.06; average 2.90 → 2.45). In Leduc with hashed sharing the locked arms are within noise
  of the plain filter.
- **The best β falls as coupling rises**: β = 8 for tabular Kuhn; β = 0.5–2 for the token class, where β = 8
  violates the descent condition and loses the gain.

In the one-player ladder there is no opponent, hence no exposure to tighten. The lock is a small cost there
($`V_{\text{opt}}-V`$: 0.276 filter vs 0.279, 0.292, 0.341 for β = 0.5, 2, 8 with hashed sharing), as Theorem 4
implies: its benefit is the exposure term, which only exists against an opponent.

---

## 7. Summary

| claim | status | evidence |
|---|---|---|
| the edge decomposes exactly over decisions, and the leaves hold most of it | PROVED (Lemma 1) | Leduc: leaves 15 % of samples, 64 % of the edge; 73 % after on-policy learning |
| a step can only earn its own decision's regret; known decisions have little to earn | PROVED (Lemmas 2–3) | filtering every ladder decision is 2.1–2.7× worse |
| on-policy updates drown rare novel decisions in shared-parameter models | PROVED (Theorem 1) | known decisions are 76–87 % of on-policy gradient rows |
| the filter partitions each batch: novel included, known excluded and recycled as lock rows | PROVED (Theorem 1′); bar error within the DKW band (Prop. 1″) | Leduc: known gradient rows 76 % → 8 %, novel inclusion 3.9×, bar inside the DKW radius on 99.8 % of updates; leaves' pivot share 20–24 % → 24–31 % |
| movement outside the pivots is non-causal | PROVED (Theorem 2, Prop. 4) | 18–42 % of movement on known decisions, correlation −0.54 to +0.03 |
| the lock caps that movement by gain / β | PROVED under the descent condition (Theorem 3) | 2.5–20× less drift; failure past the descent limit as predicted |
| locking known regions lowers the ε-Nash floor | PROVED as an inequality (Theorem 4); floor $`O(1/\beta)`$ under a progress condition | Kuhn 0.159 → 0.0003; Leduc 0.558 → 0.375 and 3.44 → 3.06; Leduc hashed within noise |

## 8. Limits

- Exact values replace sampled ones; only the choice of decisions is sampled. Label noise would weaken the filter's
  ranking.
- The games are small, so the leaves are 9–15 % of samples rather than 0.5 %. Theorem 1 says the drowning grows as
  that share falls.
- Theorem 3 assumes the pivot term ascends a potential and the descent condition holds. Theorem 4's $`O(1/\beta)`$
  floor assumes novel progress proportional to the disagreement corrected. Both are stated as assumptions and
  tested only through their predictions.
- Six seeds; differences inside two standard errors are not claimed.

## Reproduce

```bash
python -m pytest -q tests/test_filter.py      # Lemma 1, Lemma 3, the spill law, exact accounting, the drift cap
python filtering/run.py --seeds 6             # sections 5-6 (about an hour; --quick for T/10)
python filtering/regions.py                   # section 2
python filtering/inclusion.py                 # section 3.2
python filtering/causal.py                    # section 4
python filtering/figures.py                   # filtering/results/fig_*.svg
```

References: S. Kakade and J. Langford, *Approximately optimal approximate reinforcement learning*, ICML 2002
(performance difference). R. T. Rockafellar, *Monotone operators and the proximal point algorithm*, 1976.
S. Sokota et al., *A unified approach to reinforcement learning, quantal response equilibria, and two-player
zero-sum games*, ICLR 2023. J. Perolat et al., *From Poincaré recurrence to convergence in imperfect information
games*, ICML 2021.
