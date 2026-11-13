# clairvoyant-oracle

An exact hindsight oracle for Kalshi event contracts, and a whitepaper on **relative-regret reinforcement
learning**: gradients that enforce exploration where the exploitation signal alone would close the door.

Every Kalshi event market in a recording goes onto **one lattice**. Each time layer has a single **dollar node**
shared by all markets, plus YES and NO holding nodes per market. Arcs are the recorded books (taker fees included),
the prints a resting maker order could have filled against, holds, and settlement at the known result. Cash from
a 15-minute gold contract that settles can fund an oil position one layer later: the dollar nodes bridge the markets.

For every **decision node** (each dollar node and each position) the oracle gives the exact **relative regret** of
every available action:

```math
\rho(v,a) \;=\; 1 - \frac{Q^*(v,a)}{V^*(v)} \in [0,1]
```

That is the share of hindsight's value at $`v`$ that acting $`a`$ now forfeits. It is computed two ways that agree to
machine precision:

- **Marginal** (stake → 0): one backward pass over the DAG, in logs.
- **Capacitated** (a real stake): generalized maximum flow with the first move forced, through the array interface
  of [generalized-maxflow](https://github.com/wesboyt/generalized-maxflow) (vendored as a submodule in
  `third_party/`).

## Try it

### **[→ wesboyt.github.io/clairvoyant-oracle](https://wesboyt.github.io/clairvoyant-oracle/)**

The oracle running in the page, on three windows cut from the bundled hours:

- the exact backward pass over 1.3–1.6 million arcs, in JavaScript, checked against the Python oracle at every
  dollar node;
- the relative regret of every action at any second you pick;
- the capacitated oracle, solved by generalized-maxflow compiled to WebAssembly through its new `solveArrays` call,
  for the stake and horizon you choose.

Nothing is uploaded; everything is solved locally. To run it yourself:

```bash
git clone --recursive https://github.com/wesboyt/clairvoyant-oracle && cd clairvoyant-oracle
third_party/generalized-maxflow/bindings/wasm/build.sh      # needs the Emscripten SDK
cp third_party/generalized-maxflow/bindings/wasm/pkg/{index.mjs,gmaxflow.js,gmaxflow.wasm} docs/
python -m http.server -d docs 8000                          # then open localhost:8000
```

Every push builds the demo in CI (artifact `demo-site` of the **demo** workflow); running that workflow by hand
publishes it.

## The whitepaper

**[WHITEPAPER.md](WHITEPAPER.md): Relative regret: gradients that enforce exploration when exploitation is the
signal.**

- **Exploitation is the signal.** The exact on-policy label $`Q^\pi`$ never blows backwards, but it locks
  *closed doors*. An action whose payoff needs a second closed action is pushed further shut, and the keys behind it
  starve of learning (P1).
- **Hindsight alone is biased.** Following the clairvoyant max-flow field converges to the greedy-in-hindsight
  policy, lured by revealed information (P2).
- **The gate.** Relative regret detects doors that are closed ($`\pi\le\varepsilon`$) but that hindsight singles out
  ($`z\ge z_{\min}`$). At those doors only, the label is lifted just above the best sibling's on-policy value. The
  gradient stays exploitation everywhere except on mass $`\le\varepsilon`$, where it enforces exploration until
  on-policy play can see behind the door.
- **Theorem 6** (one player, first order): each tack costs $`O(\varepsilon)`$, sound doors open in bounded time, and
  the limit bias is at most $`\varepsilon\Delta`$ per false alarm. Exact games measure it: on Sailboat, south and
  north each stay at $`L=1.0`$ and the tack reaches $`L=0`$. In two-player games the gate cannot cancel rotation;
  the magnet and the average do that.
- **On real books.** Over 2026-11-12 14:00–17:00 UTC (39 series, 6,331 markets, 720 decision nodes, ~5,500
  actions each), hindsight's move is an entry, not *wait*, at 74–85 % of nodes. The shared dollar node is worth more
  than twice the log-value of any single series alone. A real stake changes the best move: at \$10,000, *wait* wins
  a node that a \$1 stake would trade.

## Use it

```bash
pip install -r requirements.txt
pip install ./third_party/generalized-maxflow          # >= 0.4: the array interface (capacitated oracle, tests)

python -m pytest -q tests                              # DP = generalized max flow on every forced first move
python -m oracle build  --start 2026-11-12T14:00:00Z --end 2026-11-12T17:00:00Z --step 15 --out tape.npz
python -m oracle regret --tape tape.npz --report results/regret.json
python -m oracle build  --start 2026-11-12T14:20:00Z --end 2026-11-12T14:50:00Z --step 5 --series KXINX        KXNASDAQ100 KXGOLD15M KXSILVER15M KXWTI15M KX30YRRATE15M --out cross.npz
python -m oracle node   --tape cross.npz --at 2026-11-12T14:29:00Z --stake 100 --horizon 24
python tools/make_demo.py                              # rebuild docs/data from the bundled hours
python tacking/experiments/e5_tack.py --quick          # the whitepaper's exact tacking table (T/10)
```

`node` prints one decision node's actions with $`\rho`$ over the window, $`\rho`$ for the horizon sub-game (trading
stops after `--horizon` layers; positions settle), and the capacitated $`Q_E`$ and $`\rho_E`$ for `--stake` dollars.

```python
from oracle import books, regret

tape = books.build("data/kalshi", books.iso_us("2026-11-12T14:00:00Z"), books.iso_us("2026-11-12T14:20:00Z"), 1_000_000)
table, V, L = regret.relative_regret_table(tape)        # every action at every dollar node, one pass
table["rho"], table["kind"], table["layer"], table["cell"]
```

## The data

`data/kalshi/<series>/2026-11-12/` holds three hours (14:00–17:00 UTC) of every recorded Kalshi event series: 39
series, 117 hourly files, 146 MB of klob1. That is the full order-book delta stream and every trade, as recorded
from Kalshi's public market-data websocket. Next to them, `settlements.jsonl` (the results) and
`markets_2026-11-12.jsonl` (market definitions) are filtered to the markets those hours define.
`oracle/klob.py` documents and reads the format; `tools/bundle_kalshi.py` shows how the bundle was cut.

## Layout

```
oracle/klob.py         klob1 reader
oracle/books.py        replay every series onto one time grid: the tape (books, prints, settlement)
oracle/lattice.py      the cross-market lattice: dollar nodes, fees, maker and taker arcs, gmaxflow arrays
oracle/regret.py       exact relative regret: backward pass (marginal), gmaxflow (capacitated), D23 z-scores
oracle/__main__.py     command line: build, regret, node, demo
docs/                  the browser demo (index.html + exported windows in docs/data)
tacking/               exact tacking experiments behind the whitepaper's section 4 (north/south fields, gate)
results/               the regret report of the bundled hours
tests/                 exactness tests
third_party/           generalized-maxflow (submodule)
```

## What the oracle does not model

Maker arcs are clairvoyant (they ignore queue position); use `--taker-only` for the conservative lattice. Books are
sampled once per layer. YES/NO netting is omitted. Fees are rounded per contract. Markets whose results are not in
the bundle have no settlement arc. Details are in WHITEPAPER §5.5.

License: MIT.
