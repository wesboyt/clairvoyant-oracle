"""Pre-commit secret scan: every file under the repo except .git. Exit 1 on any hit.
Checks: private-key blocks, cloud / API key shapes, credential-like assignments, long high-entropy tokens, files
whose names suggest credentials, and anything under models/ or data/ other than the bundled market data
(data/kalshi: klob1 recordings and settlement JSON, scanned as text all the same). Binary files (.gz, .wasm, .bin)
are checked by name only."""
import math
import os
import re
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PATTERNS = [
    re.compile(r'-----BEGIN [A-Z ]*PRIVATE KEY-----'),
    re.compile(r'AKIA[0-9A-Z]{16}'),
    re.compile(r'ghp_[0-9A-Za-z]{30,}'),
    re.compile(r'github_pat_[0-9A-Za-z_]{30,}'),
    re.compile(r'sk-[0-9A-Za-z]{20,}'),
    re.compile(r'xox[baprs]-[0-9A-Za-z-]{10,}'),
    re.compile(r'(?i)(api[_-]?key|secret|password|passwd|token|private[_-]?key)\s*[:=]\s*["\'][^"\']{8,}["\']'),
]
BAD_NAMES = re.compile(r'(?i)(creds|credential|kalshi.*\.json|cme\.json|\.pem$|\.p12$|\.key$|id_rsa)')
TOKEN = re.compile(r'[A-Za-z0-9+/=_-]{40,}')


def entropy(s):
    from collections import Counter
    c = Counter(s)
    return -sum(v / len(s) * math.log2(v / len(s)) for v in c.values())


def main():
    hits = []
    n = 0
    for dp, dn, fn in os.walk(ROOT):
        dn[:] = [d for d in dn if d not in ('.git', '__pycache__', 'third_party', '.pytest_cache')]
        rel = os.path.relpath(dp, ROOT)
        parts = rel.split(os.sep)
        if parts[0] == 'models' or (parts[0] == 'data' and len(parts) > 1 and parts[1] != 'kalshi'):
            hits.append((rel, 'forbidden directory'))
        for f in fn:
            p = os.path.join(dp, f)
            r = os.path.relpath(p, ROOT)
            n += 1
            if BAD_NAMES.search(f):
                hits.append((r, 'credential-like file name'))
            if f.endswith(('.gz', '.wasm', '.bin', '.npz')):
                continue
            try:
                txt = open(p, encoding='utf-8', errors='ignore').read()
            except OSError:
                continue
            for pat in PATTERNS:
                for m in pat.finditer(txt):
                    hits.append((r, 'pattern %s' % pat.pattern[:30]))
            for m in TOKEN.finditer(txt):
                tok = m.group(0)
                if entropy(tok) > 4.5 and not tok.replace('-', '').replace('_', '').isalpha():
                    hits.append((r, 'high-entropy token %s...' % tok[:8]))
    for h in hits:
        print('HIT', *h)
    print('scanned %d files, %d hits' % (n, len(hits)))
    sys.exit(1 if hits else 0)


if __name__ == '__main__':
    main()
