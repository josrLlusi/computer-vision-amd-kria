import re, sys
t = open(sys.argv[1]).read()
g = lambda k: sum(int(x) for x in re.findall(rf'\n\s+{k}\s+(\d+)', t))
lut = sum(g(f'LUT{i}') for i in range(1, 7)); ff = g('FDRE') + g('FDSE'); dsp = g('DSP48E2')
bram = g('RAMB36E2') + g('RAMB18E2') / 2; uram = g('URAM288')
print(f"LUT {lut} ({lut/117120:.1%}) | FF {ff} ({ff/234240:.1%}) | DSP {dsp} ({dsp/1248:.1%}) | BRAM36eq {bram:.1f} ({bram/144:.1%}) | URAM {uram} ({uram/64:.1%})")
