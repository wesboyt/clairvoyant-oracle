"""How data/kalshi was cut: copy the hourly klob1 files of a few hours, every event series, and keep only the
settlement and market-definition rows of the markets those files define.

    python tools/bundle_kalshi.py SRC_ROOT --day 2026-11-12 --hours 14 15 16 [--out data/kalshi]

SRC_ROOT/<series>/<day>/<series>_<YYYYMMDD>-<HH>00.klob.gz, SRC_ROOT/<series>/settlements.jsonl and
SRC_ROOT/<series>/markets_<day>.jsonl are the recorder's layout.
"""
import argparse
import glob
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from oracle import klob  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--day', required=True)
    ap.add_argument('--hours', nargs='+', type=int, required=True)
    ap.add_argument('--out', default=os.path.join(HERE, 'data', 'kalshi'))
    a = ap.parse_args()
    stamp = a.day.replace('-', '')
    total = 0
    for sdir in sorted(glob.glob(os.path.join(a.src, '*'))):
        series = os.path.basename(sdir)
        tickers = set()
        files = []
        for h in a.hours:
            files += sorted(glob.glob(os.path.join(sdir, a.day, '%s_%s-%02d00*.klob.gz' % (series, stamp, h))))
        if not files:
            continue
        dst = os.path.join(a.out, series, a.day)
        os.makedirs(dst, exist_ok=True)
        for f in files:
            _, it = klob.read(f)
            tickers |= {tk for tk, r in it if r[0] == klob.MARKETDEF}
            shutil.copy2(f, dst)
            total += os.path.getsize(f)
        for name in ('settlements.jsonl', 'markets_%s.jsonl' % a.day):
            src = os.path.join(sdir, name)
            if not os.path.exists(src):
                continue
            keep = []
            with open(src, encoding='utf-8') as fh:
                for line in fh:
                    if line.strip() and json.loads(line).get('ticker') in tickers:
                        keep.append(line if line.endswith('\n') else line + '\n')
            with open(os.path.join(a.out, series, name), 'w', encoding='utf-8', newline='\n') as fh:
                fh.writelines(keep)
        print('%-16s %d files, %d markets' % (series, len(files), len(tickers)))
    print('klob bytes: %.1f MB' % (total / 1e6))


if __name__ == '__main__':
    main()
