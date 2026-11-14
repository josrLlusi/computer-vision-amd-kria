"""Compare RTL simulation output in DIR against the golden files. usage: python check.py DIR"""
import sys, json
d = sys.argv[1]
meta = json.load(open(f'{d}/meta.json'))
def s16(h):
    v = int(h, 16); return v - 65536 if v >= 32768 else v
ok = True
# 1) whole activation memory (validates conv_engine)
dump = [s16(l.strip()) for l in open(f'{d}/act_dump.hex') if l.strip() and not l.startswith('//')]
gold = [s16(l.strip()) for l in open(f'{d}/act_golden.hex') if l.strip()]
n = meta['act_words']
bad = [i for i in range(n) if dump[i] != gold[i]]
print(f'activation memory : {n - len(bad)}/{n} words match' + ('' if not bad else f'   first mismatch @ {bad[0]}: rtl={dump[bad[0]]} gold={gold[bad[0]]}'))
ok &= not bad
# 2) detection rows (validates det_post)
rtl = [l.split() for l in open(f'{d}/rows.txt') if l.strip()]
gld = [l.split() for l in open(f'{d}/rows_golden.txt') if l.strip()]
nbad = sum(1 for a, b in zip(rtl, gld) if a != b)
print(f'detection rows    : {len(gld) - nbad - abs(len(rtl)-len(gld))}/{len(gld)} rows match (rtl has {len(rtl)})')
if len(rtl) != len(gld) or nbad:
    ok = False
    for i, (a, b) in enumerate(zip(rtl, gld)):
        if a != b: print(f'  row {i}: rtl={a} gold={b}'); break
print('PASS' if ok else 'FAIL')
sys.exit(0 if ok else 1)
