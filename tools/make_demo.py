"""Build the browser demo's windows from the bundled hours: docs/data/<name>.bin.gz + .json + windows.json.

    python tools/make_demo.py [--data data/kalshi] [--tmp <dir for intermediate tapes>]
"""
import argparse
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from oracle.__main__ import main as cli  # noqa: E402

METALS = ['KXGOLD15M', 'KXGOLDH', 'KXGOLDD', 'KXGOLDW', 'KXGOLDMON',
          'KXSILVER15M', 'KXSILVERH', 'KXSILVERD', 'KXSILVERW', 'KXSILVERMON']
ENERGY = ['KXWTI15M', 'KXWTIH', 'KXWTI', 'KXWTIW', 'KXBRENT15M', 'KXBRENTD', 'KXBRENTW', 'KXBRENTMON',
          'KXNATGAS15M', 'KXNATGASD', 'KXNATGASW', 'KXNATGASMON']
CROSS = ['KXINX', 'KXNASDAQ100', 'KXGOLD15M', 'KXSILVER15M', 'KXWTI15M', 'KXBRENT15M', 'KXNATGAS15M',
         'KXCOPPER15M', 'KX10YRRATE15M', 'KX2YRRATE15M', 'KX5YRRATE15M', 'KX30YRRATE15M', 'KXGOLDD', 'KXWTI']

WINDOWS = [
    dict(file='metals_1415', label='Gold + silver, 14:13:30-14:16:30 UTC, 1 s (the 14:15 contracts settle)',
         series=METALS, start='2026-11-12T14:13:30Z', end='2026-11-12T14:16:30Z', step=1, minutes=3),
    dict(file='cross_open', label='Indices, rates, metals, energy, 14:20-14:50 UTC, 5 s (the stock open)',
         series=CROSS, start='2026-11-12T14:20:00Z', end='2026-11-12T14:50:00Z', step=5, minutes=30),
    dict(file='energy_15', label='Oil + natural gas, 15:00-15:20 UTC, 5 s',
         series=ENERGY, start='2026-11-12T15:00:00Z', end='2026-11-12T15:20:00Z', step=5, minutes=20),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.path.join(HERE, 'data', 'kalshi'))
    ap.add_argument('--tmp', default=tempfile.gettempdir())
    ap.add_argument('--out', default=os.path.join(HERE, 'docs', 'data'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for w in WINDOWS:
        tape = os.path.join(a.tmp, 'demo_%s.npz' % w['file'])
        cli(['build', '--data', a.data, '--start', w['start'], '--end', w['end'], '--step', str(w['step']),
             '--series', *w['series'], '--out', tape])
        cli(['demo', '--tape', tape, '--at', w['start'], '--minutes', str(w['minutes']),
             '--out', os.path.join(a.out, w['file'])])
        os.remove(tape)
    with open(os.path.join(a.out, 'windows.json'), 'w', encoding='utf-8') as fh:
        json.dump([dict(file=w['file'], label=w['label']) for w in WINDOWS], fh, indent=1)


if __name__ == '__main__':
    main()
