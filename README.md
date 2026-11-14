# clairvoyant-oracle

An exact hindsight oracle for Kalshi event contracts, and a whitepaper on relative-regret reinforcement learning.

All Kalshi event markets in a recording are placed on **one time-expanded lattice**. Each time layer has a single
**dollar node** shared by every market, plus YES and NO holding nodes per market. Arcs are the recorded order books
(taker fees included), trades a resting maker order could have filled against, holds, and settlement at the known
result. Because the dollar node is shared, the payout of one market can fund a position in another a layer later.

For every **decision node** (each dollar node and each open position), the oracle computes the exact **relative
regret** of every available action:

```math
\rho(v,a) \;=\; 1 - \frac{Q^*(v,a)}{V^*(v)} \in [0,1]
```

where $`V^*(v)`$ is the hindsight-optimal value from $`v`$ and $`Q^*(v,a)`$ is the value of taking $`a`$ first and
playing optimally afterwards. Two solvers are provided, and they agree to machine precision:

- **Marginal** (stake → 0): a single backward pass over the lattice, in log space.
- **Capacitated** (a finite stake): generalized maximum flow with the first move forced, using
  [generalized-maxflow](https://github.com/wesboyt/generalized-maxflow) (included as a submodule in `third_party/`).

## Demo

### **[→ wesboyt.github.io/clairvoyant-oracle](https://wesboyt.github.io/clairvoyant-oracle/)**

The oracle runs entirely in the browser on three windows cut from the bundled data:

- the marginal backward pass over 1.3–1.6 million arcs, verified against the Python implementation at every dollar
  node;
- the relative regret of every action at any selected second;
- the capacitated oracle, solved by generalized-maxflow compiled to WebAssembly, for a chosen stake and trading
  horizon.

To serve it locally:

```bash
git clone --recursive https://github.com/wesboyt/clairvoyant-oracle && cd clairvoyant-oracle
third_party/generalized-maxflow/bindings/wasm/build.sh      # requires the Emscripten SDK
cp third_party/generalized-maxflow/bindings/wasm/pkg/{index.mjs,gmaxflow.js,gmaxflow.wasm} docs/
python -m http.server -d docs 8000                          # open http://localhost:8000
```

The **demo** workflow builds the site on every push (artifact `demo-site`) and publishes it when run manually.

## Whitepaper

**[WHITEPAPER.md](WHITEPAPER.md): Relative regret: gradients that enforce exploration when exploitation is the
signal.**

On-policy action values reward what the current policy already does, so an action whose payoff depends on another
rarely taken action is pushed toward zero probability and never revisited. Clairvoyant values avoid this but bias
the policy toward outcomes it could not have known. The paper uses relative regret from a clairvoyant oracle only
to *locate* such closed actions, and raises their on-policy label just above the best alternative. The resulting
gradient explores only on a bounded amount of probability mass. It proves first-order progress guarantees for the
single-agent case, measures them on exactly solvable games, and applies the oracle to three hours of Kalshi order
books.

**[FILTER_PAPER.md](FILTER_PAPER.md): Novel and known regions: causal exploration and a tighter Nash bound for
shared-parameter policies.**

Game space splits into known regions, where the policy agrees with a best-response oracle, and novel ones, where it
does not, and the two are not known to correlate. The edge sits in rare novel decisions near the leaves, which
on-policy updates drown out in any model that shares parameters across decisions. The paper proves that a decision's
own, causal share of the update is proportional to how often it is visited, while the share pushed onto it through
shared parameters is not. It shows that the filter is a statistical test that partitions every batch (novel
candidates included as pivots, known ones excluded from the gradient and recycled as lock rows), that this restores
the causal share, that simply removing the excluded samples biases the update toward optimistic strategies (it
subtracts the known regions' gradient and their restoring force), that a
KL lock on the known regions caps the non-causal movement at the segment's gain over β, and that this lowers the
reachable ε-Nash floor. Everything is measured exactly on Kuhn and Leduc poker and a one-player ladder, with
tabular, hashed and token-shared policies (`filtering/`).

## Usage

```bash
pip install -r requirements.txt
pip install ./third_party/generalized-maxflow          # >= 0.4, needed for the capacitated oracle and tests

python -m pytest -q tests
python -m oracle build  --start 2026-11-12T14:00:00Z --end 2026-11-12T17:00:00Z --step 15 --out tape.npz
python -m oracle regret --tape tape.npz --report results/regret.json
python -m oracle build  --start 2026-11-12T14:20:00Z --end 2026-11-12T14:50:00Z --step 5 \
       --series KXINX KXNASDAQ100 KXGOLD15M KXSILVER15M KXWTI15M KX30YRRATE15M --out cross.npz
python -m oracle node   --tape cross.npz --at 2026-11-12T14:29:00Z --stake 100 --horizon 24
python tools/make_demo.py                              # rebuild docs/data from the bundled data
python tacking/experiments/e5_tack.py --quick          # the whitepaper's tacking experiments at T/10
python filtering/run.py --seeds 6 && python filtering/figures.py   # FILTER_PAPER.md's tables and figures
python filtering/regions.py && python filtering/inclusion.py && python filtering/bias.py && python filtering/causal.py   # sections 2-4
```

`node` prints one decision node's actions with $`\rho`$ over the whole window, $`\rho`$ for the horizon sub-game
(trading stops after `--horizon` layers; open positions settle), and the capacitated $`Q_E`$ and $`\rho_E`$ for
`--stake` dollars.

```python
from oracle import books, regret

tape = books.build("data/kalshi", books.iso_us("2026-11-12T14:00:00Z"), books.iso_us("2026-11-12T14:20:00Z"), 1_000_000)
table, V, L = regret.relative_regret_table(tape)        # every action at every dollar node, one pass
table["rho"], table["kind"], table["layer"], table["cell"]
```

## Data

`data/kalshi/<series>/2026-11-12/` contains 2026-11-12 14:00–17:00 UTC for 39 Kalshi event series: 117 hourly
klob1 files (146 MB) with the full order-book delta stream and all trades from Kalshi's public market-data
websocket. Each series directory also holds `settlements.jsonl` (results) and `markets_2026-11-12.jsonl` (market
definitions), filtered to the markets in those hours. `oracle/klob.py` documents and reads the format;
`tools/bundle_kalshi.py` reproduces the cut.

## Layout

```
oracle/klob.py         klob1 reader
oracle/books.py        replays every series onto one time grid (books, trades, settlement)
oracle/lattice.py      the cross-market lattice: dollar nodes, fees, maker and taker arcs, solver arrays
oracle/regret.py       relative regret: backward pass (marginal), generalized max flow (capacitated), outlier scores
oracle/__main__.py     command line: build, regret, node, demo
docs/                  browser demo (index.html and exported windows in docs/data)
tacking/               exact experiments for the whitepaper's section 4
filtering/             the filter and lock on exact games, results and figures for FILTER_PAPER.md
results/               regret report for the bundled data
tests/                 exactness tests
third_party/           generalized-maxflow (submodule)
```

## Limitations

Maker arcs assume the order was first in queue; `--taker-only` gives a conservative lattice. Books are sampled once
per layer. YES/NO netting is not modeled. Fees are rounded per contract. Markets without a result in the bundle have
no settlement arc. See WHITEPAPER §5.5.

## License

MIT
