"""A dependency-free SVG plotter (lines and scatter, linear or log axes) for the results figures."""
import math

PALETTE = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#17becf',
           '#bcbd22']


def _tr(v, lo, hi, a, b, log):
    if log:
        v, lo, hi = math.log10(max(v, 1e-300)), math.log10(lo), math.log10(hi)
    if hi == lo:
        return (a + b) / 2
    return a + (v - lo) / (hi - lo) * (b - a)


def plot(path, series, title='', xlabel='', ylabel='', logx=False, logy=False, w=640, h=400, ylim=None,
         xlim=None, scatter=False, diag=False):
    """series: [(label, xs, ys)] ; scatter: points instead of lines; diag: draw y = x."""
    L, R, T, B = 70, 170, 34, 50
    pts = [(x, y) for _, xs, ys in series for x, y in zip(xs, ys)
           if x is not None and y is not None and math.isfinite(x) and math.isfinite(y)
           and (not logx or x > 0) and (not logy or y > 0)]
    if not pts:
        return
    x0, x1 = (min(p[0] for p in pts), max(p[0] for p in pts)) if xlim is None else xlim
    y0, y1 = (min(p[1] for p in pts), max(p[1] for p in pts)) if ylim is None else ylim
    if logy:
        y0 = max(y0, 1e-12)
    if logx:
        x0 = max(x0, 1e-12)
    if y0 == y1:
        y1 = y0 + 1
    if x0 == x1:
        x1 = x0 + 1
    X = lambda v: _tr(min(max(v, x0), x1), x0, x1, L, w - R, logx)  # noqa: E731
    Y = lambda v: _tr(min(max(v, y0), y1), y0, y1, h - B, T, logy)  # noqa: E731
    o = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" font-family="sans-serif" '
         'font-size="11">' % (w, h), '<rect width="100%" height="100%" fill="white"/>',
         '<text x="%d" y="18" font-size="13" font-weight="bold">%s</text>' % (L, title),
         '<rect x="%d" y="%d" width="%d" height="%d" fill="none" stroke="#444"/>' % (L, T, w - R - L, h - B - T)]

    def ticks(lo, hi, log):
        if log:
            a, b = math.floor(math.log10(lo)), math.ceil(math.log10(hi))
            st = max(1, (b - a) // 6)
            return [10.0 ** e for e in range(a, b + 1, st) if lo <= 10.0 ** e <= hi]
        n = 5
        return [lo + (hi - lo) * i / n for i in range(n + 1)]

    for v in ticks(x0, x1, logx):
        xx = X(v)
        o.append('<line x1="%.1f" y1="%d" x2="%.1f" y2="%d" stroke="#ddd"/>' % (xx, T, xx, h - B))
        o.append('<text x="%.1f" y="%d" text-anchor="middle">%s</text>' % (xx, h - B + 14, _fmt(v, logx)))
    for v in ticks(y0, y1, logy):
        yy = Y(v)
        o.append('<line x1="%d" y1="%.1f" x2="%d" y2="%.1f" stroke="#ddd"/>' % (L, yy, w - R, yy))
        o.append('<text x="%d" y="%.1f" text-anchor="end">%s</text>' % (L - 4, yy + 4, _fmt(v, logy)))
    o.append('<text x="%d" y="%d" text-anchor="middle">%s</text>' % ((L + w - R) / 2, h - 12, xlabel))
    o.append('<text x="14" y="%d" transform="rotate(-90 14 %d)" text-anchor="middle">%s</text>' % (
        (T + h - B) / 2, (T + h - B) / 2, ylabel))
    if diag:
        lo, hi = max(x0, y0), min(x1, y1)
        o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#999" stroke-dasharray="4"/>' % (
            X(lo), Y(lo), X(hi), Y(hi)))
    for j, (lab, xs, ys) in enumerate(series):
        c = PALETTE[j % len(PALETTE)]
        P = [(X(x), Y(y)) for x, y in zip(xs, ys) if x is not None and y is not None and math.isfinite(x)
             and math.isfinite(y) and (not logx or x > 0) and (not logy or y > 0)]
        if scatter:
            for px, py in P:
                o.append('<circle cx="%.1f" cy="%.1f" r="3" fill="%s" fill-opacity="0.75"/>' % (px, py, c))
        elif P:
            o.append('<polyline fill="none" stroke="%s" stroke-width="1.5" points="%s"/>' % (
                c, ' '.join('%.1f,%.1f' % p for p in P)))
        ly = T + 12 + 15 * j
        o.append('<rect x="%d" y="%d" width="10" height="10" fill="%s"/>' % (w - R + 8, ly - 9, c))
        o.append('<text x="%d" y="%d">%s</text>' % (w - R + 22, ly, lab))
    o.append('</svg>')
    with open(path, 'w') as f:
        f.write('\n'.join(o))


def _fmt(v, log):
    if log:
        return '1e%d' % round(math.log10(v))
    if abs(v) >= 100 or (abs(v) < 0.01 and v != 0):
        return '%.1e' % v
    return '%.3g' % v


def quiver(path, fields, trajs, title='', xlabel='', ylabel='', w=640, h=520):
    """fields: [(label, [[x, y, dx, dy], ...])] drawn as arrows (normalised length); trajs: [(label, xs, ys)].
    Axes [0, 1] x [0, 1]."""
    L, R, T, B = 60, 170, 34, 50
    X = lambda v: L + v * (w - R - L)  # noqa: E731
    Y = lambda v: h - B - v * (h - B - T)  # noqa: E731
    o = ['<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" font-family="sans-serif" '
         'font-size="11">' % (w, h), '<rect width="100%" height="100%" fill="white"/>',
         '<text x="%d" y="18" font-size="13" font-weight="bold">%s</text>' % (L, title),
         '<rect x="%d" y="%d" width="%d" height="%d" fill="none" stroke="#444"/>' % (L, T, w - R - L, h - B - T)]
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        o.append('<text x="%.1f" y="%d" text-anchor="middle">%.2f</text>' % (X(v), h - B + 14, v))
        o.append('<text x="%d" y="%.1f" text-anchor="end">%.2f</text>' % (L - 4, Y(v) + 4, v))
    o.append('<text x="%d" y="%d" text-anchor="middle">%s</text>' % ((L + w - R) / 2, h - 12, xlabel))
    o.append('<text x="14" y="%d" transform="rotate(-90 14 %d)" text-anchor="middle">%s</text>' % (
        (T + h - B) / 2, (T + h - B) / 2, ylabel))
    j = 0
    for lab, arr in fields:
        c = PALETTE[j % len(PALETTE)]
        for x, y, dx, dy in arr:
            n = math.hypot(dx, dy)
            if n < 1e-12:
                o.append('<circle cx="%.1f" cy="%.1f" r="1.5" fill="%s"/>' % (X(x), Y(y), c))
                continue
            s = 0.03 / n
            x2, y2 = x + dx * s, y + dy * s
            o.append('<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="%s" stroke-width="1.2"/>' % (
                X(x), Y(y), X(x2), Y(y2), c))
            o.append('<circle cx="%.1f" cy="%.1f" r="1.6" fill="%s"/>' % (X(x2), Y(y2), c))
        ly = T + 12 + 15 * j
        o.append('<rect x="%d" y="%d" width="10" height="10" fill="%s"/>' % (w - R + 8, ly - 9, c))
        o.append('<text x="%d" y="%d">%s</text>' % (w - R + 22, ly, lab))
        j += 1
    for lab, xs, ys in trajs:
        c = PALETTE[j % len(PALETTE)]
        o.append('<polyline fill="none" stroke="%s" stroke-width="2.2" points="%s"/>' % (
            c, ' '.join('%.1f,%.1f' % (X(x), Y(y)) for x, y in zip(xs, ys))))
        ly = T + 12 + 15 * j
        o.append('<rect x="%d" y="%d" width="10" height="10" fill="%s"/>' % (w - R + 8, ly - 9, c))
        o.append('<text x="%d" y="%d">%s</text>' % (w - R + 22, ly, lab))
        j += 1
    o.append('</svg>')
    with open(path, 'w') as f:
        f.write('\n'.join(o))
