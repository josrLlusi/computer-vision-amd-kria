"""Compare RTL simulation output in DIR against the golden files. usage: python check.py DIR [--noact]"""
import sys, json
d = sys.argv[1]
noact = '--noact' in sys.argv
nobox = '--nobox' in sys.argv
meta = json.load(open(f'{d}/meta.json'))
def lines(p): return [l.strip() for l in open(p) if l.strip() and not l.startswith('//')]
ok = True
if not noact:
    dump = lines(f'{d}/act_dump.hex'); gold = lines(f'{d}/act_golden.hex')
    n = meta['act_words']
    bad = [i for i in range(n) if int(dump[i], 16) != int(gold[i], 16)]
    print(f'activation RAM : {n - len(bad)}/{n} words match' + ('' if not bad else f'   first mismatch @ {bad[0]}: rtl={dump[bad[0]]} gold={gold[bad[0]]}'))
    ok &= not bad
    if nobox: bd = None
    else: bd = lines(f'{d}/box_dump.hex'); bg = lines(f'{d}/box_golden.hex'); n = meta['box_words']
    bad = [] if nobox else [i for i in range(n) if int(bd[i], 16) != int(bg[i], 16)]
    if not nobox: print(f'box RAM        : {n - len(bad)}/{n} words match' + ('' if not bad else f'   first mismatch @ {bad[0]}: rtl={bd[bad[0]]} gold={bg[bad[0]]}'))
    ok &= not bad
rtl = [l.split() for l in open(f'{d}/rows.txt') if l.strip()]
gld = [l.split() for l in open(f'{d}/rows_golden.txt') if l.strip()]
nbad = sum(1 for a, b in zip(rtl, gld) if a != b)
print(f'detection rows : {len(gld) - nbad - abs(len(rtl)-len(gld))}/{len(gld)} rows match (rtl has {len(rtl)})')
if len(rtl) != len(gld) or nbad:
    ok = False
    for i, (a, b) in enumerate(zip(rtl, gld)):
        if a != b: print(f'  row {i}: rtl={a} gold={b}'); break
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
