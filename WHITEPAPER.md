# Relative regret: gradients that enforce exploration when exploitation is the signal

*A clairvoyant max-flow oracle, the closed-door problem of on-policy learning, and a gate that opens doors without
biasing the limit.*

---

## Abstract

On-policy reinforcement learning learns from what its own policy does. The signal it follows, the on-policy action
value $`Q^\pi`$, is an **exploitation** signal: it scores each action by what the *current* policy would do afterwards.
When an action's value depends on a later action that the policy has also stopped taking, $`Q^\pi`$ says the first
action is bad. The gradient then closes it further, and the keys behind it receive geometrically less learning: a
**closed door** that the exact gradient keeps shut. We prove this lock (P1), and show that the obvious repair (follow
a clairvoyant oracle instead) converges to the wrong policy (P2).

The fix studied here is **relative regret**. A clairvoyant oracle (generalized maximum flow over the realised future)
values each action at a decision node as if every later decision were perfect. We do not use those values as
labels. Their *relative* form,
$`\rho(v,a) = 1 - Q^{*}(v,a)/V^{*}(v)`$, locates the doors the policy has closed but hindsight says are open. A gate
fires only on those doors and swaps a single label: the door's on-policy value is raised just above its best
sibling's. The resulting gradient is an exploitation gradient everywhere except on mass $`\le\varepsilon`$, where it
**enforces exploration**: it lifts the door at a bounded rate until on-policy play can see what lies behind it. Then
it falls silent.

We give the angle condition under which a label field makes progress. We prove, to first order for one agent, that
this tacking between the two fields costs at most $`O(\varepsilon)`$ per tack, opens every sound door in bounded time,
and has a limit bias of at most $`\varepsilon\cdot\Delta`$ per false alarm. Exact enumeration on designed games
measures it: each field alone stays locked at $`L = 1.0`$, and the tack reaches $`L = 0`$. In two-player zero-sum
games the gate cannot cancel rotation; that is the job of the magnet and the average. Finally we build the oracle
exactly on recorded Kalshi event-contract books: one lattice across every market, with a dollar node per second
bridging them. It gives the relative regret of every available decision, verified to agree with generalized max flow
to machine precision.

---

## 1. Exploitation is the signal

### 1.1 Setting

A single agent acts on an exogenous path (a deal $`d`$: hidden cards, the future order books). A decision key $`k`$ is
what the agent has observed plus its own in-flight state. $`A(k)`$ is the legal action set, and a behavioural policy
$`\pi(a\mid k)`$ chooses among them. Because the agent's actions do not move the path, $`P(d\mid k)`$ depends only on what was
observed, never on the policy (Lemma 0 of `tacking/`). The strategy space is a flow polytope: realisation plans
$`x\ge 0`$ with $`\sum_a x(k,a) = \sum_{\sigma\in\text{parents}(k)} x(\sigma)`$. A policy *is* the centroid of the
cloud of pure strategy trees it draws (Prop. 1.1).

### 1.2 The on-policy label (south)

The label the trainer follows is the exact on-policy value: an **expectation under $`\pi`$** over its own
continuation tree, backed up from the sink, never a maximum over futures:
```math
W(k,a) = g(k,a) + \sum_{k'\,\text{child of}\,(k,a)}\ \sum_{a'} \pi(a'\mid k')\, W(k',a'),\qquad
Q^\pi(k,a) = W(k,a)/R(k).
```
One update moves the logits by the centred label, weighted by on-policy occupancy:
```math
\theta(k,a) \leftarrow \theta(k,a) + \eta\, w(k)\,\frac{q(k,a) - \bar q(k)}{\sigma},\qquad w(k) = \frac{\text{occ}_\pi(k)}{\max\text{occ}}.
```

### 1.3 The angle condition

**Proposition (angle condition).** To first order, a logit step $`\Delta\theta`$ changes the value by
$`\Delta V = \sum_k \text{occ}(k)\sum_a \pi(a\mid k)\,A^\pi(k,a)\,\Delta\theta(k,a)`$, with $`A^\pi = Q^\pi - V^\pi`$.
Under the update above,
```math
\Delta V \;\approx\; \frac{\eta}{\sigma}\sum_k \text{occ}(k)\,w(k)\,\big\langle A^\pi(k,\cdot),\; q(k,\cdot)-\bar q(k)\big\rangle_\pi .
```
A label field $`q`$ makes progress exactly when its centred advantage has a positive $`\pi`$-weighted inner product
with the true advantage. This is a sailboat's *velocity made good*: any heading within 90° of the wind's useful
component moves you upwind.

### 1.4 P1: the exact exploitation gradient locks doors

**P1 (PROVED).** With exact labels $`q = Q^\pi`$, $`\langle A^\pi, Q^\pi - \bar q\rangle_\pi = \sum_a \pi(a)A^\pi(a)^2 \ge 0`$,
so south never blows backwards. Yet it can stall forever. Take a door $`a^*`$ with $`\pi(a^*) = \delta`$ whose payoff
needs a later action $`g`$ that is also closed, $`\pi(g)=\delta'`$. Then $`Q^\pi(a^*) \approx \delta' V_g < V`$, and south
**lowers** $`\theta(a^*)`$, so $`\delta_t`$ shrinks geometrically. The keys behind the door receive total learning
$`\sum_t w_t \propto \sum_t \delta_t < \infty`$. So $`\pi(g)`$ never approaches 1, $`Q^\pi(a^*)`$ stays below $`V`$, and
the lock is self-consistent. In the language of policy-gradient theory, the distribution-mismatch coefficient behind
the door is $`1/\delta_t\to\infty`$.

This is the sense in which **exploitation is the signal**. Every exact on-policy gradient pushes mass toward what the
current policy already does well. Nothing in it opens a door that only pays once a second closed door is also
opened.

### 1.5 P2: the clairvoyant signal alone is biased

The **north** field reveals the deal and plays perfectly afterwards. For one agent it is an average of per-path
maxima over the realised lattice; in a market it is generalized maximum flow (§5).

**Proposition 1.2 (PROVED).** (i) $`\text{mf}(k,a) \ge Q^{\pi^*}(k,a)`$ for every key and action. (ii) Equality
holds if no own decision follows $`a`$. (iii) The premium $`\max_a \text{mf} - \max_a Q^{\pi^*} \ge 0`$ is the value of
information about $`d`$ for the rest of the episode.

**P2 (PROVED).** Following north alone converges to greedy(mf). That policy is off the optimal set wherever a later
choice exploits revealed information, i.e. wherever $`\arg\max\text{mf}\notin\arg\max Q^{\pi^*}`$. Clairvoyant labels
make waiting nearly free and gambles look safe; in market settings, training on clairvoyant $`Q^*`$ labels tends to
collapse the policy to "wait".

So neither field alone is enough: one locks doors, the other lures the policy into gambles.

---

## 2. Relative regret

### 2.1 Definition

At a decision node $`v`$, let $`Q^*(v,a)`$ be the clairvoyant value of taking $`a`$ now and playing perfectly
afterwards, and let $`V^*(v) = \max_a Q^*(v,a)`$. The **relative regret** of $`a`$ is
```math
\rho(v,a) \;=\; 1 - \frac{Q^*(v,a)}{V^*(v)} \;\in\; [0,1].
```
It is the share of the hindsight-optimal value that $`a`$ forfeits. The additive regret $`V^* - Q^*(a)`$, used as a
label, carries the clairvoyance premium. Relative regret has three properties a label needs.

1. **The shared offset cancels.** Every sibling at $`v`$ shares the same perfect continuation machinery, so the
   premium that inflates $`Q^*`$ largely divides out. What remains orders the siblings.
2. **It is scale-free.** Clairvoyant values compound: on recorded Kalshi books one dollar can turn into
   $`e^{20}`$–$`e^{100}`$ dollars of hindsight value within minutes (§5.4). Additive regrets at different nodes are
   then incomparable, while $`\rho`$ always lies in $`[0,1]`$.
3. **It is a door detector, not a label.** $`\rho`$ is used only to decide *where* the on-policy label is wrong
   (§3). The labels themselves stay on the $`Q^\pi`$ scale, which matters because a market max flow and an
   on-policy rollout value can differ by orders of magnitude.

### 2.2 The outlier score

A door is worth opening only if hindsight singles it out. With $`u(a) = 1-\rho(v,a)`$ and $`s_u`$ the spread of
$`u`$ over the siblings,
```math
z(v,a^*) = \frac{u(a^*) - \max_{b\ne a^*} u(b)}{s_u}\qquad\text{(argmax form)},\qquad
z(v,a) = \frac{u(a) - \sum_b \pi(b)\,u(b)}{s_u}\qquad\text{(against the current mix)}.
```
$`z \ge z_{\min}`$ is the false-alarm filter.

---

## 3. The gate: a gradient that enforces exploration

### 3.1 The rule

South everywhere, except at a key $`k`$ with a **closed door**: an action $`a`$ with $`\pi(a\mid k)\le\varepsilon`$
that is a north outlier, $`z(k,a)\ge z_{\min}`$. There only that action's label is replaced:
```math
q(k,a) \;\leftarrow\; \max_b \hat Q^\pi(k,b) + v\cdot s_Q ,
```
with $`s_Q`$ the spread of the on-policy labels over the siblings and $`v`$ small (0.25). Every sibling keeps its
$`\hat Q^\pi`$. The **argmax gate** fires only on north's argmax. The **all-outliers gate** fires on every closed north
outlier, a refinement motivated by the exact experiments (§4.3).

### 3.2 Why this is the right gradient

At a flagged key the door's centred label is at least $`v\,s_Q`$ above every sibling's, so its logit rises by at least
$`\eta w v s_Q/\sigma`$ per step. The swap touches only actions of mass $`\le\varepsilon`$, so its effect on $`\Delta V`$
is $`\text{occ}\cdot w\cdot\pi(a)\cdot A^\pi(a)\cdot(q'-\hat Q(a))`$, bounded by $`O(\varepsilon)`$. The policy keeps
exploiting everywhere it can see. Where the exploitation gradient would shut a door that hindsight says leads
somewhere, the gradient is overridden just enough to keep the door at $`\pi\ge\varepsilon`$. That is long enough for
on-policy rollouts to *observe* what lies behind it. Once $`Q^\pi(a)`$ exceeds the alternatives on its own, south
keeps the door open and the gate stops firing.

The swap value is $`\max_b \hat Q + v s_Q`$, not the clairvoyant number. The door "feels" like the best rollout,
slightly better, so the clairvoyant scale never enters the loss.

### 3.3 Theorem 6 (tacking makes guaranteed progress; one player, first order)

Assume exact south labels with pruning threshold $`p_{\text{prune}} < \varepsilon`$, so pruning never cuts a door the
gate is opening, and:

- **(N1) North is sound on true doors.** At every key on an optimal path, a closed optimal action is a north
  outlier.
- **(N2) The firing rule reaches it.** The argmax gate needs the optimal door to be north's argmax; the all-outliers gate needs only N1.
- **(N3) False alarms are bounded.** A non-optimal outlier costs at most $`\Delta_k = V^*(k) - Q^*(k,a)`$.

Then:

- (a) each tack costs at most $`O(\varepsilon)`$ per flagged key;
- (b) each tack opens its door within $`O\big(\sigma\log(\varepsilon/\pi_0)/(\eta w v s_Q)\big)`$ steps;
- (c) coverage turns the wind. A door held at $`\pi\ge\varepsilon`$ caps the mismatch coefficient behind it at
  $`1/\varepsilon`$, so P1's non-negative progress applies there at a rate bounded below;
- (d) every limit point satisfies $`V \ge V_{\text{opt}} - \sum_k \text{occ}(k)\,\varepsilon\,\Delta_k`$ over the
  keys where a false outlier flutters at the $`\varepsilon`$ boundary. The bias vanishes once south's improved mix
  drops the false door's $`z`$ below $`z_{\min}`$.

*Proof sketch.* (a) and (b) follow from the angle condition with $`\pi(a^*)\le\varepsilon`$. For (c), occupancy behind
a door is the door's probability times its parent's, and P1 applies there with weight $`\ge\varepsilon`$. For (d),
at a limit point either no gate fires (a south stationary point with full coverage of every N1 door, hence
optimal), or a door flutters at $`\pi\approx\varepsilon`$ at a cost of at most $`\varepsilon\Delta_k`$. ∎ This is a
small-step argument, not a finite-step rate.

### 3.4 Two players: the gate is an exploration device, not a convergence device

In zero-sum games the on-policy field also rotates: the restricted gap can vanish off the Nash set, and iterates
cycle (matching pennies, Kuhn). The gate acts on mass $`\le\varepsilon`$ and cannot cancel an $`O(1)`$ rotation. In
matrix games there is no deal to reveal, so north equals south and the switched system *is* south. What turns
rotation into convergence is the magnet: a proximal refresh, which is Fejér-monotone and strictly dissipative. The
average policy, which turns no-regret into set convergence, does the rest. **Relative regret opens doors; the magnet
and the average converge.**

---

## 4. Measured on exact games

Everything here is exact (full enumeration, LP/DP solvers, expected updates); `tacking/` reproduces it.
$`L`$ = NashConv (one player: $`V_{\text{opt}} - V`$) of the final iterate, with the linear average's in brackets.
"fl" counts the updates in which the gate fired.

| game (T) | south (the wind) | north alone | tack (argmax gate) | tack (all outliers) | vswap (old) |
|---|---|---|---|---|---|
| DoorTrap (3000) | **1.00** (locked) | 0 | **0** (fl 164) | 0 | 0 |
| Sailboat (3000) | **1.00** (locked) | **1.00** (lured by the gamble) | **0** (fl 49) | 0 | 0.049 (holds the false door at ε) |
| WrongDoor (3000) | 1.00 | 1.20 | **1.01** (fl 3000: holds the wrong door) | **0** | 1.01 |
| FalseDoor (3000) | 0 | 0.10 | 0.0053 (bound ε·Δ = 0.005) | 0.0053 | 0.0056 |
| L1, trap door at 1% (1500) | 0.246 | 0.145 | 0.246 (never fires: south reopens it) | 0.246 | 0.153 |
| R2, entry takes at 1% (600) | 0.230 | 0.268 | 0.232 (fl 518) | 0.234 | 0.228 |
| MP (2000) | 1.94 [0.072] | = south | 1.45 [0.036] | 1.45 [0.036] | 1.94 [0.072] |
| Kuhn (2000) | 0.425 [0.027] | 0.489 [0.013] | 0.392 [0.015] | 0.392 [0.015] | 0.425 [0.027] |
| Leduc (200) | 2.39 [2.10] | 2.79 [3.12] | 2.30 [2.08] | 2.30 [2.09] | 2.30 [2.09] |

### 4.1 Reading the table

- **Sailboat** is the thesis in one row. South locks the door ($`L = 1.0`$). North is lured by a clairvoyant gamble
  ($`L = 1.0`$). The tack reaches $`L = 0`$ after the gate fires in 49 updates. Old vswap, which swaps the
  clairvoyant spread into every label and has no $`z`$ filter, holds the false gamble at $`\varepsilon`$
  ($`L = 0.049 \approx \varepsilon\times 1.0`$).
- **The angle instrument.** Across the one-player door games, the step's cosine to the south advantage stays
  0.98–1.00 under tacking, while its cosine to the north advantage rises to 0.88–0.99 during tacks. In DoorTrap,
  south alone has cos(step, north) = 0.19: the wind blows away from the door.
  `tacking/results/fig_doortrap_fields.svg` draws the three fields and trajectories on the
  $`(\pi(\text{door}),\pi(\text{good}))`$ plane.
- **FalseDoor** measures the bound of Theorem 6(d): bias 0.0052 against $`\varepsilon\Delta = 0.005`$. With a safer
  alternative, $`z`$ drops to 0.98 and the gate never fires: $`z_{\min}`$ filters false alarms.

### 4.2 Two players

In matching pennies, Kuhn and Leduc the iterate's NashConv stays near plain play's level, as §3.4 predicts, and the
average converges in every arm. Tacking still helps a little (Kuhn 0.425 → 0.392; Leduc 2.39 → 2.30), but it is not
what converges.

### 4.3 WrongDoor and the all-outliers gate

When north's argmax among the closed doors is itself a false door (N2 fails), the argmax gate holds the wrong door at
$`\varepsilon`$ forever and never fires on the true one. **Firing on every closed outlier fixes it:
$`L = 0`$.**

---

## 5. The oracle on real books: a cross-market Kalshi lattice

The theory needs a north field that is exact for the game the policy plays. For trading this is generalized maximum
flow on a time-expanded lattice of the recorded books (`oracle/`).

### 5.1 One dollar node per second bridges every market

Layer $`j`$ is the instant $`t_j = t_0 + j\Delta`$. Each layer has **one dollar node $`\text{USD}_j`$ shared by every
market**, plus a YES and a NO holding node per live market. Arcs:

| arc | from → to | gain per unit in | capacity |
|---|---|---|---|
| take YES at ask level $`(a,n)`$ | $`\text{USD}_j\to\text{YES}_{j+1}`$ | $`1/(a+f(a))`$ contracts per \$ | $`n(a+f(a))`$ \$ |
| take NO at YES-bid level $`(b,n)`$ | $`\text{USD}_j\to\text{NO}_{j+1}`$ | $`1/(1-b+f(b))`$ | $`n(1-b+f(b))`$ |
| sell YES / NO at the book | position $`\to\text{USD}_{j+1}`$ | $`b-f(b)`$ / $`1-a-f(a)`$ | $`n`$ contracts |
| make (print in $`(t_j,t_{j+1}]`$ at $`p`$) | $`\text{USD}_j\to\text{YES}_{j+1}`$ or $`\text{NO}_{j+1}`$ | $`1/p`$ or $`1/(1-p)`$, no fee | print size |
| hold | $`X_j \to X_{j+1}`$ | 1 | ∞ |
| settle | YES (NO) end node $`\to\text{USD}_s`$ | 1 if the market settled YES (NO) | ∞ |

$`f(P) = \lceil 7P(1-P)\rceil`$ cents is Kalshi's quadratic taker fee for these series, charged as for one-contract
orders. Makers pay none. Every arc points forward in time, so the lattice is a DAG with no generating cycles.
Settlement cash returns to the dollar node of the settlement instant: a 15-minute gold contract's payout can fund a
WTI position a second later. That is the bridge. Without it, each market would need its own bankroll and the
clairvoyant could not chain across them.

### 5.2 Decision nodes and their exact relative regret

Every dollar node is a decision node. Its available actions are *wait* and, for every live market,
*take YES / take NO / make YES / make NO*. Every position node is one too: *hold / sell / make-sell*. Two exact
oracles give their relative regret.

**Marginal (stake → 0).** With a stake small enough that no capacity binds, generalized max flow from a node is the
best product of gains over its paths to the sink, i.e. the zero-load node dual. On a DAG one backward pass gives
every node's value, in logs because compounding overflows floats:
```math
V(v) = \max_{v\to w}\ \big(\log\text{gain}_{v\to w} + V(w)\big),\qquad V(T) = 0,\qquad
\rho(v,a) = 1 - e^{\,Q(v,a) - V(v)}.
```

**Capacitated (stake $`E`$).** gmaxflow on the window's arrays, with $`E`$ dollars entering at $`\text{USD}_j`$ and the
first move forced: $`Q_E(a) = \text{maxflow}_a/E`$, $`\rho_E(a) = 1 - Q_E(a)/\max_b Q_E(b)`$. Depth now matters, and
compounding stops where the book runs out.

**Exactness.** `tests/test_oracle.py` solves random lattices both ways, for the free node and for **every forced first
move**. The backward pass and generalized max flow agree to a relative $`10^{-9}`$ at small stakes. As the stake
grows, the capacitated value per dollar falls monotonically. On the bundled data the browser demo's JavaScript
backward pass reproduces the Python oracle's value at every dollar node to $`< 10^{-15}`$.

### 5.3 Speed: arrays straight from memory

A regret label needs one solve per action. The oracle therefore calls generalized-maxflow through its array
interface: `gmf_solve_arrays_flows` in its C ABI, `max_flow_arrays` in Python (numpy buffers read
in place), and `solveArrays` in WebAssembly (typed arrays copied straight into the heap). That skips the JSON round
trip, which is 5–32× faster on lattices of 50–3,000 edges.

### 5.4 What the bundled three hours show

Bundle: 2026-11-12, 14:00–17:00 UTC, every recorded Kalshi event series (39: gold, silver, copper, WTI, Brent,
natural gas, the S&P 500 and Nasdaq-100 ranges, Treasury yields; 15-minute to monthly expiries). That is 6,331
markets. On a 15-second grid the lattice has 720 layers and 3.27 million cells (4,542 live markets per layer); the
exact backward pass over all of it takes seconds. Report: `results/regret_3h_15s.json`
(`python -m oracle regret`).

| | maker + taker arcs | taker only |
|---|---|---|
| decision nodes (dollar nodes) | 720 | 720 |
| available actions per node | 5,559 | 5,533 |
| clairvoyant's choice is an entry, not *wait* | **85 %** of nodes | **74 %** |
| that choice is a north outlier ($`z\ge 1`$) | 39 % | 9.7 % |
| median $`\rho`$ of *wait* | 0.25 | 0.075 |
| best 1 % / 10 % of all actions: $`\rho\le`$ | 0.29 / 0.54 | 0.087 / 0.28 |
| median action | $`\rho = 1`$ to double precision | same |
| $`\log V(\text{USD}_0)`$, shared dollar node | 315.4 | 106.2 |
| best single series on its own | 156.0 (KXGOLD15M) | 38.5 (KXINXU) |
| **bridge gain** (shared − best alone, log) | **+159.5** | **+67.7** |

How to read it:

- **The values are absurd on purpose.** At zero load, hindsight compounds without limit. It buys each 15-minute
  winner below \$1, collects at settlement, and rolls the cash into the next market a second later.
  $`e^{106}`$ is what that chain multiplies to in three hours. That is why the labels are relative: $`\rho`$ stays
  in $`[0,1]`$ however large $`V`$ gets.
- **Hindsight says act; on-policy play says wait.** At 74–85 % of nodes the clairvoyant's move is an entry, and
  waiting forfeits a quarter of the node's value (median, maker lattice). An on-policy learner that has collapsed to
  *wait* is facing exactly the closed doors of §1.4.
- **The bridge is most of the value.** Any single series on its own reaches at most half the log-value of the shared
  lattice. Chaining settlements *across* markets through the dollar node is where the rest comes from.
- **Outliers are rare enough to gate on.** With makers, the clairvoyant choice stands out from its siblings
  ($`z\ge 1`$) at 39 % of nodes; taker-only, at under 10 %. The gate fires on doors that are both closed and
  outliers, so its firing rate is bounded by these shares.

**A real stake changes the answer.** One node (14:29 UTC, cross-asset window, 5-second grid, trading horizon 2
minutes, 352 actions), solved by gmaxflow with the stake entering at the dollar node:

| stake | $`V_E`$ per \$ | best first move | $`\rho_E`$ of the zero-stake choice |
|---|---|---|---|
| \$1 | 1,319 | make NO, silver 15-min | 0 |
| \$100 | 47.0 | take NO, 30-year yield 15-min | 0.71 |
| \$10,000 | 10.7 | wait | 0.99 |

Depth caps compounding. The bigger the stake, the more the best move is one the book can absorb. The oracle reports
both: $`\rho`$ (marginal) as a scale-free door detector, and $`\rho_E`$ (capacitated) for the stake a policy
actually trades. The browser demo solves the second with gmaxflow's WebAssembly build (7 solves on 275k arcs in about
5 s), and its numbers match the native Python oracle to six decimals.

### 5.5 What the oracle does not model

- **Maker arcs are clairvoyant.** They assume we were the resting order a print traded against, ignoring queue
  position. That is right for a ruler (north is optimistic by design: the gate only uses its ordering and outliers)
  and wrong as a fill model. The taker-only lattice (`--taker-only`, the demo's checkbox) is the conservative
  bound.
- **Grid.** Books are sampled each $`\Delta`$; an ask that appears and vanishes within one interval is not offered.
  The oracle is a lower bound on the continuous-time clairvoyant.
- **YES/NO netting** (holding both legs of one market redeems \$1 at once) needs a flow-merging arc and is omitted.
  This is another lower-bound effect.
- **Fees** are rounded per contract (the one-contract schedule), an upper bound on a larger order's rounded fee.
- **Unknown results.** Markets whose settlement is not in the bundle (some weeklies and monthlies) have no settle
  arc; their contracts are worth only what the book pays before the window ends.

---

## 6. Summary

| | exploitation signal (south) | clairvoyant signal (north) | relative-regret gate (tack) |
|---|---|---|---|
| what it scores | what the current policy does next | what perfect hindsight does next | south, plus north's *ordering* at closed doors |
| failure | locks doors (P1) | lured by revealed information (P2) | false doors held at ε (bias ≤ εΔ) |
| scale | $`Q^\pi`$ | compounding, unbounded | $`Q^\pi`$ (the swap is max Q̂ + v·s_Q) |
| guarantee | monotone, may stall | biased limit | progress per tack, doors open in bounded time (Thm 6) |

The exploitation gradient is the right gradient almost everywhere. The relative-regret gate makes it the right
gradient everywhere, by enforcing exploration on exactly the mass where exploitation is blind.

## Reproduce

```bash
python -m pytest -q tests                                   # exactness: DP = generalized max flow
python -m oracle build --start 2026-11-12T14:00:00Z --end 2026-11-12T17:00:00Z --step 15 --out tape.npz
python -m oracle regret --tape tape.npz --report results/regret.json
python -m oracle node --tape tape.npz --at 2026-11-12T14:30:00Z --stake 10
python tacking/experiments/e5_tack.py --quick               # the §4 table at T/10 (about 10 min)
```
