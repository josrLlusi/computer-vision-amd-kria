"""UNTESTED host driver for the AXI4-Lite wrapper on the Kria (PYNQ / Linux).  Needs `pynq` (MMIO) or /dev/mem access.
Block design: Zynq UltraScale+ PS M_AXI_HPM0_FPD -> yolo26_head_kria_top s_axi (assign an 8 MB range, e.g. 0xA000_0000).
Files come from py/golden.py::write_files (or py/run_image.py): prog.hex wgt.hex par.hex lut.hex sig.hex act_init.hex meta.json
usage:  h = Yolo26Head(0xA0000000, 'sim/real160_L16'); rows = h.run()
"""
import json, time
import numpy as np
from pynq import MMIO

PROG, PAR, WGT, LUT, SIG, ACT, OUT = 0x100000, 0x200000, 0x300000, 0x400000, 0x500000, 0x600000, 0x700000


class Yolo26Head:
    def __init__(self, base, outdir, lanes=None):
        self.m = MMIO(base, 0x800000)
        self.d = outdir
        self.meta = json.load(open(f'{outdir}/meta.json'))
        self.L = lanes or self.meta['L']
        assert self.m.read(0x10) == 0x59323648, 'ID mismatch: wrong base address or bitstream'
        assert (self.m.read(0x14) >> 16) == self.L, 'LANES mismatch between bitstream and data files'

    def _words(self, name):
        return [int(l, 16) for l in open(f'{self.d}/{name}') if l.strip()]

    def _write_wide(self, base, words, nch):
        for i, w in enumerate(words):
            for c in range(nch):
                self.m.write(base + (i * nch + c) * 4, (w >> (32 * c)) & 0xFFFFFFFF)

    def load(self):
        self._write_wide(PROG, self._words('prog.hex'), 1)
        self._write_wide(PAR, self._words('par.hex'), 2)
        self._write_wide(WGT, self._words('wgt.hex'), self.L // 4)
        self._write_wide(LUT, self._words('lut.hex'), 1)
        self._write_wide(SIG, self._words('sig.hex'), 1)
        self._write_wide(ACT, self._words('act_init.hex'), self.L // 4)   # activation RAM is writable only while idle

    def run(self, timeout=10.0):
        self.m.write(0x00, 1)
        t = time.time()
        while not (self.m.read(0x04) & 2):
            if time.time() - t > timeout:
                raise TimeoutError('head did not finish')
        n = self.m.read(0x08); self.cycles = self.m.read(0x0C)
        rows = np.zeros((n, 6), dtype=np.int64)
        for i in range(n):
            for f in range(6):
                v = self.m.read(OUT + (i * 6 + f) * 4)
                rows[i, f] = (v - (1 << 32) if v & 0x80000000 else v) if f < 4 else (v & 0xFFFF)
        out = rows.astype(float); out[:, :4] /= 4.0; out[:, 4] /= 65535.0      # pixels, score
        return out
