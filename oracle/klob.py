"""Reader for klob1, the recorder format of the bundled Kalshi event-contract data.

A klob1 file is a gzip stream: one JSON header line, then binary records. One file holds every tracked market of
one series for one rotation period (an hour). Prices are integers in units of $0.0001 (price_exp 4), sizes in
units of 0.01 contracts (size_exp 2). A Kalshi book is two bid ladders, YES bids and NO bids; klob1 stores it as one
book in YES-price terms:

    BID       a YES bid level's new total size at YES price p (0 = level removed)
    ASK       a NO bid level, shown as a YES offer at 1 - p (0 = removed)
    BUY/SELL  a print; BUY = the taker bought YES at p, SELL = the taker sold YES (bought NO) at p
    SNAPSHOT  replaces the market's book: bids best-first, asks best-first
    GAP       the market's book is unknown until the next SNAPSHOT (reason 3: the market closed)
    MARKETDEF defines market idx: JSON {ticker, strike, close_time, ...}
    MARKET    following records belong to market idx

Record framing (all varints are LEB128; "zigzag" maps signed to unsigned):
    BID/ASK   tag, dt zigzag, dprice zigzag, qty uvarint
    BUY/SELL  tag, dt zigzag, dprice zigzag, size uvarint, dtid zigzag
    SNAPSHOT  tag, t_abs uvarint, reason byte, nbid uvarint, nask uvarint, then (dprice zigzag, qty uvarint) each
    GAP       tag, t_abs uvarint, reason byte
dt is microseconds since the previous record's time; t_abs resets it. dprice is the change from the previous price
written in the file (any record type, any market). A record cut off by truncation is dropped.
"""
import json
import zlib

BID, ASK, BUY, SELL, SNAPSHOT, GAP = 1, 2, 3, 4, 16, 17
MARKET, MARKETDEF = 32, 33
GAP_CLOSED = 3
PRICE_ONE = 10_000          # $1 in price units
SIZE_ONE = 100              # one contract in size units


def read_bytes(path):
    """Decompress a gzip file, tolerating a truncated tail (the recorder flushes with Z_SYNC_FLUSH)."""
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    out = bytearray()
    with open(path, 'rb') as fh:
        while True:
            chunk = fh.read(1 << 20)
            if not chunk:
                break
            try:
                out += d.decompress(chunk)
            except zlib.error:
                break
    return bytes(out)


def read(path):
    """-> (header dict, iterator of (ticker, record)). Records are tuples:
        (BID|ASK, t_us, price, qty)
        (BUY|SELL, t_us, price, size, trade_id)
        (SNAPSHOT, t_us, reason, [(p, q) bids best-first], [(p, q) asks best-first])
        (GAP, t_us, reason)
        (MARKETDEF, meta)  with ticker = meta['ticker']
    """
    data = read_bytes(path)
    nl = data.index(b'\n')
    return json.loads(data[:nl]), _records(data, nl + 1)


def _records(buf, i):
    n = len(buf)
    t = p = tid = 0
    names = {}
    cur = None

    def uv():
        nonlocal i
        shift = v = 0
        while True:
            c = buf[i]
            i += 1
            v |= (c & 0x7F) << shift
            if c < 0x80:
                return v
            shift += 7

    def sv():
        v = uv()
        return (v >> 1) ^ -(v & 1)

    while i < n:
        start = i
        try:
            tag = buf[i]
            i += 1
            if tag == MARKET:
                cur = names[uv()]
                continue
            if tag == MARKETDEF:
                idx, ln = uv(), uv()
                if i + ln > n:
                    return
                meta = json.loads(buf[i:i + ln])
                i += ln
                names[idx] = meta['ticker']
                yield meta['ticker'], (MARKETDEF, meta)
                continue
            if tag == BID or tag == ASK:
                t += sv()
                p += sv()
                rec = (tag, t, p, uv())
            elif tag == BUY or tag == SELL:
                t += sv()
                p += sv()
                size = uv()
                tid += sv()
                rec = (tag, t, p, size, tid)
            elif tag == SNAPSHOT:
                t = uv()
                reason = buf[i]
                i += 1
                nb, na = uv(), uv()
                levels = []
                for _ in range(nb + na):
                    p += sv()
                    levels.append((p, uv()))
                rec = (SNAPSHOT, t, reason, levels[:nb], levels[nb:])
            elif tag == GAP:
                t = uv()
                reason = buf[i]
                i += 1
                rec = (GAP, t, reason)
            else:
                raise ValueError('bad tag %d at byte %d' % (tag, start))
        except IndexError:
            return
        yield cur, rec

