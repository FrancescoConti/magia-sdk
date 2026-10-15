#!/usr/bin/env python3
# Copyright 2026 ETH Zurich, University of Bologna and Fondazione Chips-IT.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#
# Francesco Conti <francesco.conti@chips.it>
"""mgc_sweep - run the tests in tests/magia/mgc over a sweep of configurations on gvsoc and
compare cycles, MAC/cycle and the utilization of RedMulE, iDMA and Spatz.

Each configuration is built and run, one at a time, with
    make clean build run_profiling platform=gvsoc tiles=<N> test=<test> pulp_cluster=0
and measured on gvsoc_work/trace.perfetto-trace:
  - cycles: from tile 0's entry in main to the exit of the last tile other than tile 0, i.e.
    without crt0 init and without tile 0's final check()
  - MAC/cycle: matmul MACs only (2*H*S*S*P for attention, softmax counts 0) / cycles
  - per-tile utilization in that window, averaged over all tiles: RedMulE busy, iDMA backend
    busy (L2->L1, L1->L2, either), Spatz running a task or its VFU active, and empty cycles
    (none of them busy: CV32 alone, or CV32 asleep waiting for an event)

Parameters (S, D, P, H from include/test.h, S_BLOCK_SIZE_Q/KV from the .mgc) default to the
test's own; any other value regenerates test.h (golden generators) and src/test.c (mgcc) in
place. The test sources are restored at the end (also on Ctrl-C, or on the next start if the
script was killed). D and H are written to test.h but not used by the current attention kernels.

Sweep: a JSON file with a list of groups, each the cartesian product of its lists (missing
parameters take each test's default), e.g.
    {"groups": [
      {"tests": ["unfused_attn", "flat_attn_pipeopt3"], "tiles": [2, 4]},
      {"tests": ["flat_attn_pipeopt3"], "tiles": [4], "S_BLOCK_SIZE_KV": [256, 512]}
    ]}
or a single group from the command line:
    tests/magia/mgc/scripts/mgc_sweep.py --tests unfused_attn flat_attn_pipeopt3 --tiles 2 4 [--S 512 1024 ...]
    tests/magia/mgc/scripts/mgc_sweep.py --sweep sweep.json [--out DIR] [--dry-run] [--keep-traces]
    tests/magia/mgc/scripts/mgc_sweep.py --list

Output in --out: summary.log (human-readable table), results.csv (one row per run, plain
comma-separated numbers, empty for N/A: Excel, or numpy.genfromtxt(f, delimiter=',',
names=True, dtype=None, encoding='utf-8')), runs/<id>/ (make log, per-tile metrics).

Needs the `perfetto` Python package (pip install perfetto); the golden generators need torch
(--gen-python), mgcc needs xDSL (mgc/.venv, see mgc/README.md).
"""

import argparse
import csv
import datetime
import glob
import itertools
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))
MGC_TESTS = os.path.join(REPO, 'tests', 'magia', 'mgc')
FLAT_GEN = os.path.join(REPO, 'tests', 'magia', 'mesh', 'mm_generator', 'flat_attn.py')
SWEEP_DIR = os.path.join(REPO, 'mgc_sweep')
BACKUP = os.path.join(SWEEP_DIR, '.backup')
CYCLE_NS = 5  # gvsoc clock: 200 MHz, perfetto timestamps in ns

PARAMS = ['S', 'D', 'P', 'H', 'S_BLOCK_SIZE_Q', 'S_BLOCK_SIZE_KV']
HEADER_DEFINES = {'S': 'SEQ_LENGTH', 'D': 'EMBED_SIZE', 'P': 'PROJ_SIZE', 'H': 'NUM_HEADS'}
MGC_DEFINES = ['S_BLOCK_SIZE_Q', 'S_BLOCK_SIZE_KV']
# Spatz runtime outside of any task (boot, dispatcher, interrupt handling)
SPATZ_IDLE_FUNCS = {'_start', 'dispatcher_loop', '_trap_handler', '_handle_exception',
                    '_handle_interrupt', '_trap_ret'}

CSV_COLUMNS = [
    'run_id', 'test', 'tiles', 'n_tiles', *PARAMS, 'status', 'errors', 'cycles', 'macs',
    'mac_per_cycle', 'mac_per_cycle_per_tile', 'redmule_pct', 'idma_pct', 'idma_in_pct',
    'idma_out_pct', 'spatz_pct', 'spatz_vfu_pct', 'empty_pct', 'empty_cv32_active_pct',
    'empty_cv32_sleep_pct', 'main_start_cycle', 'end_cycle', 'sim_seconds']

# (header, csv column, format) of the human-readable table
TABLE = [
    ('id', 'run_id', '{}'), ('test', 'test', '{}'), ('tiles', 'tiles', '{}'), ('S', 'S', '{}'),
    ('D', 'D', '{}'), ('P', 'P', '{}'), ('H', 'H', '{}'), ('BQ', 'S_BLOCK_SIZE_Q', '{}'),
    ('BKV', 'S_BLOCK_SIZE_KV', '{}'), ('status', 'status', '{}'), ('cycles', 'cycles', '{:,}'),
    ('MAC/cyc', 'mac_per_cycle', '{:.1f}'), ('MAC/cyc/tile', 'mac_per_cycle_per_tile', '{:.2f}'),
    ('RedMulE%', 'redmule_pct', '{:.1f}'), ('iDMA%', 'idma_pct', '{:.1f}'),
    ('in%', 'idma_in_pct', '{:.1f}'), ('out%', 'idma_out_pct', '{:.1f}'),
    ('Spatz%', 'spatz_pct', '{:.1f}'), ('VFU%', 'spatz_vfu_pct', '{:.1f}'),
    ('empty%', 'empty_pct', '{:.1f}'), ('CV32 run%', 'empty_cv32_active_pct', '{:.1f}'),
    ('CV32 sleep%', 'empty_cv32_sleep_pct', '{:.1f}')]


# --------------------------------------------------------------------------------------------
# Tests and their parameters
# --------------------------------------------------------------------------------------------

def read(path):
    with open(path) as f:
        return f.read()


def write(path, text):
    with open(path, 'w') as f:
        f.write(text)


class Test:
    """One test directory of tests/magia/mgc (built by tests/magia/mgc/CMakeLists.txt)."""

    def __init__(self, name):
        self.name = name
        self.dir = os.path.join(MGC_TESTS, name)
        cmake = read(os.path.join(self.dir, 'CMakeLists.txt'))
        self.target = re.search(r'set\(TEST_NAME\s+(\S+)\)', cmake).group(1)
        m = re.search(r'TEST_NAME\s+(\w+_spatz)\b', cmake)
        self.spatz_name = m.group(1) if m else None
        self.header = os.path.join(self.dir, 'include', 'test.h')
        self.test_c = os.path.join(self.dir, 'src', 'test.c')
        self.mgc = os.path.join(self.dir, name + '.mgc')
        if not os.path.exists(self.mgc):
            self.mgc = sorted(glob.glob(os.path.join(self.dir, '*.mgc')))[0]
        if name.startswith('flat_attn'):
            self.kind = 'flat_attn'
        elif name == 'unfused_attn':
            self.kind = 'unfused_attn'
        else:
            self.kind = 'mm'
        self.defaults = self._defaults()
        self.goldens = [int(t) for t in re.findall(r'^#(?:el)?if MESH_X_TILES == (\d+)', read(self.header), re.M)]

    def _defaults(self):
        h, src = read(self.header), read(self.mgc)
        d = {}
        for p, macro in HEADER_DEFINES.items():
            m = re.search(r'#define\s+%s\s+\(?(\d+)\)?' % macro, h)
            d[p] = int(m.group(1)) if m else None
        for p in MGC_DEFINES:
            m = re.search(r'^%s\s*=\s*define\((\d+)\)' % p, src, re.M)
            d[p] = int(m.group(1)) if m else None
        if self.kind == 'mm':
            self.mnk = {k: int(re.search(r'#define\s+%s\s+\(?(\d+)\)?' % k, h).group(1))
                        for k in ('M_SIZE', 'N_SIZE', 'K_SIZE')}
            m = re.search(r'^N_ITERATIONS\s*=\s*define\((\d+)\)', src, re.M)
            self.mnk['N_ITERATIONS'] = int(m.group(1)) if m else 1
        return d

    def applicable(self):
        """Parameters this test has (the others are N/A)."""
        return [p for p in PARAMS if self.defaults[p] is not None]

    def macs(self, cfg):
        if self.kind == 'mm':
            k = self.mnk
            return k['M_SIZE'] * k['N_SIZE'] * k['K_SIZE'] * k['N_ITERATIONS']
        # Q @ Kt and attn @ V, one head per H
        return 2 * (cfg['H'] or 1) * cfg['S'] * cfg['S'] * cfg['P']

    def check(self, cfg, tiles):
        """Reason why (cfg, tiles) cannot run, or None."""
        if self.kind == 'mm' and any(cfg[p] != self.defaults[p] for p in self.applicable()):
            return 'mm tests run with their default sizes only'
        if self.kind == 'flat_attn':
            S, bq, bkv = cfg['S'], cfg['S_BLOCK_SIZE_Q'], cfg['S_BLOCK_SIZE_KV']
            if tiles & (tiles - 1):
                return 'tiles must be a power of 2'
            if S % bq or S % bkv:
                return 'S must be a multiple of S_BLOCK_SIZE_Q and S_BLOCK_SIZE_KV'
            if bq % tiles or bkv % tiles:
                return 'S_BLOCK_SIZE_Q and S_BLOCK_SIZE_KV must be multiples of tiles'
            if S // bq > 255 or S // bkv > 255:
                return 'more than 255 blocks'
        return None

    def needs_regen(self, cfg, tiles):
        changed = any(cfg[p] != self.defaults[p] for p in self.applicable())
        return changed or (self.goldens and tiles not in self.goldens)


def list_tests():
    cm = read(os.path.join(MGC_TESTS, 'CMakeLists.txt'))
    return re.findall(r'^add_subdirectory\((\w+)\)', cm, re.M)


# --------------------------------------------------------------------------------------------
# In-place regeneration of test.h / test.c, with backup and restore
# --------------------------------------------------------------------------------------------

class Sources:
    """Modifies a test's sources in place; restore() puts back the originals."""

    def __init__(self, log):
        self.log = log
        self.saved = {}  # original path -> backup path

    def save(self, path):
        if path in self.saved:
            return
        os.makedirs(BACKUP, exist_ok=True)
        bak = os.path.join(BACKUP, '%d_%s' % (len(self.saved), os.path.basename(path)))
        shutil.copy2(path, bak)
        self.saved[path] = bak
        write(os.path.join(BACKUP, 'manifest.json'), json.dumps(self.saved, indent=1))

    def restore(self):
        for path, bak in self.saved.items():
            shutil.copy2(bak, path)
        if self.saved:
            self.log('restored %d test source file(s)' % len(self.saved))
        self.saved = {}
        shutil.rmtree(BACKUP, ignore_errors=True)

    @staticmethod
    def recover(log):
        """Restore the sources left modified by a killed sweep."""
        man = os.path.join(BACKUP, 'manifest.json')
        if os.path.exists(man):
            for path, bak in json.loads(read(man)).items():
                shutil.copy2(bak, path)
                log('recovered %s from a previous interrupted sweep' % os.path.relpath(path, REPO))
        shutil.rmtree(BACKUP, ignore_errors=True)


def generate(test, cfg, tiles, args, srcs, log):
    """Write test.h, the .mgc defines and test.c for cfg; return None or an error string."""
    for p in (test.header, test.test_c, test.mgc):
        srcs.save(p)
    # restart from the originals: a previous configuration may have changed them
    for p in (test.header, test.test_c, test.mgc):
        shutil.copy2(srcs.saved[p], p)

    key = '%s_S%s_D%s_P%s_H%s_KV%s_T%d' % (test.kind, cfg['S'], cfg['D'], cfg['P'], cfg['H'],
                                          cfg['S_BLOCK_SIZE_KV'], tiles)
    cache = os.path.join(args.cache, key)
    cached = os.path.join(cache, 'test.h')
    if not os.path.exists(cached):
        log('  generating test.h (%s)...' % key)
        os.makedirs(cache, exist_ok=True)
        env = dict(os.environ)
        if test.kind == 'flat_attn':
            cmd = [args.gen_python, FLAT_GEN, '--s_size', cfg['S'], '--e_size', cfg['D'],
                   '--p_size', cfg['P'], '--n_heads', cfg['H'], '--block_kv', cfg['S_BLOCK_SIZE_KV'],
                   '--tiles', tiles, '--inc_dir', cache]
        else:
            # writes the test's own include/test.h (restored at the end)
            cmd = [args.gen_python, os.path.join(test.dir, 'gen_golden.py'),
                   '--s_size', cfg['S'], '--p_size', cfg['P']]
            env['PYTHONPATH'] = os.path.dirname(FLAT_GEN)
        r = subprocess.run([str(c) for c in cmd], cwd=REPO, env=env, capture_output=True, text=True)
        write(os.path.join(cache, 'gen.log'), r.stdout + r.stderr)
        if r.returncode:
            shutil.rmtree(cache, ignore_errors=True)
            return 'golden generation failed: ' + (r.stderr.strip().splitlines() or ['?'])[-1]
        if test.kind == 'unfused_attn':
            shutil.copy2(test.header, cached)
    shutil.copy2(cached, test.header)

    src = read(test.mgc)
    for p in MGC_DEFINES:
        if cfg[p] is not None:
            src = re.sub(r'^(%s\s*=\s*define\()\d+(\))' % p, r'\g<1>%d\2' % cfg[p], src, flags=re.M)
    write(test.mgc, src)
    r = subprocess.run([args.mgcc_python, os.path.join(REPO, 'mgc', 'mgcc.py'), test.mgc, '-o', test.test_c],
                       cwd=REPO, capture_output=True, text=True)
    if r.returncode:
        return 'mgcc failed: ' + ' '.join(r.stderr.split())[:200]
    return None


# --------------------------------------------------------------------------------------------
# Build + run
# --------------------------------------------------------------------------------------------

def run_make(test, tiles, logf):
    cmd = 'make clean build run_profiling platform=gvsoc tiles=%d test=%s pulp_cluster=0' % (tiles, test.target)
    env_setup = 'source sourceme.sh && ' if os.path.exists(os.path.join(REPO, 'sourceme.sh')) else ''
    out = ''
    for _ in range(2):
        r = subprocess.run(['bash', '-c', env_setup + cmd], cwd=REPO, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, errors='replace')
        out = r.stdout
        # a fresh build dir can fail the binary check: rerun once the binary exists
        if 'No test found' not in out:
            break
    write(logf, '$ ' + cmd + '\n' + out)
    return r.returncode, out


def parse_run(out):
    kills = {int(m.group(2)): int(m.group(1)) for m in re.finditer(
        r'^\d+: (\d+): .*kill-module.*Received kill request from port (\d+)', out, re.M)}
    errs = [int(e) for e in re.findall(r'TILE \d+: Number of errors: (\d+)', out)]
    return kills, errs


# --------------------------------------------------------------------------------------------
# Trace analysis
# --------------------------------------------------------------------------------------------

def merge(iv):
    """Union of [start, end) intervals, as a sorted disjoint (n, 2) array."""
    iv = np.asarray(iv, dtype=np.int64).reshape(-1, 2)
    if not len(iv):
        return iv
    iv = iv[np.argsort(iv[:, 0])]
    ends = np.maximum.accumulate(iv[:, 1])
    new = np.r_[True, iv[1:, 0] > ends[:-1]]
    starts = iv[new, 0]
    stop = np.r_[np.flatnonzero(new)[1:] - 1, len(iv) - 1]
    return np.stack([starts, ends[stop]], axis=1)


def clip(iv, lo, hi):
    iv = np.clip(np.asarray(iv, dtype=np.int64).reshape(-1, 2), lo, hi)
    return iv[iv[:, 1] > iv[:, 0]]


def length(iv):
    return int((iv[:, 1] - iv[:, 0]).sum()) if len(iv) else 0


def overlap(a, b):
    """Length of the intersection of two merged interval sets."""
    i = j = t = 0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i, 0], b[j, 0]), min(a[i, 1], b[j, 1])
        t += max(0, hi - lo)
        if a[i, 1] < b[j, 1]:
            i += 1
        else:
            j += 1
    return int(t)


def spatz_idle_pcs(test):
    """Absolute [lo, hi) PC ranges of the Spatz runtime outside tasks, and the binary's range."""
    m = re.search(r'^CMAKE_NM:FILEPATH=(.+)$', read(os.path.join(REPO, 'build', 'CMakeCache.txt')), re.M)
    nm = m.group(1) if m else 'riscv64-unknown-elf-nm'
    syms = subprocess.run([nm, os.path.join(REPO, 'build', 'bin', test.target)], capture_output=True,
                          text=True).stdout
    addr = {m.group(2): int(m.group(1), 16) for m in re.finditer(r'^([0-9a-f]+) \w (\w+)$', syms, re.M)}
    base, end = addr['_spatz_binary_start'], addr['_spatz_binary_end']
    dump = glob.glob(os.path.join(REPO, 'build', 'bin', 'magia', test.spatz_name, 'spatz_task', '*.dump'))[0]
    funcs, in_text = [], False
    for line in open(dump):
        if line.startswith('Disassembly of section'):
            in_text = line.strip().endswith('.text:')
        m = re.match(r'^([0-9a-f]+) <([^>]+)>:', line)
        if in_text and m and not m.group(2).startswith('.L'):
            funcs.append((int(m.group(1), 16), m.group(2)))
    funcs.append((end - base, None))
    idle = [(base + a, base + b) for (a, f), (b, _) in zip(funcs, funcs[1:]) if f in SPATZ_IDLE_FUNCS]
    return idle, (base, end)


def analyze(test, trace, kills, n_tiles):
    from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig
    tp = TraceProcessor(trace=trace, config=TraceProcessorConfig(load_timeout=1800))
    try:
        tracks = {}  # (tile, track name) -> track id
        for r in tp.query("select t.id, t.name, p.name pname from track t join track p on t.parent_id = p.id "
                          "where p.name like 'magia-tile %'"):
            tracks[(int(r.pname.split()[-1]), r.name)] = r.id

        def slices(tile, name):
            tid = tracks.get((tile, name))
            if tid is None:
                return np.zeros((0, 2), np.int64)
            q = tp.query('select ts, dur from slice where track_id = %d' % tid).as_pandas_dataframe()
            ts, dur = q.ts.to_numpy(np.int64), q.dur.to_numpy(np.int64)
            dur = np.where(dur < 0, np.iinfo(np.int64).max // 4, dur)  # unterminated: until the end
            return np.stack([ts, ts + dur], axis=1)

        main0 = tp.query("select min(ts) t from slice where track_id = %d and name = 'main'"
                         % tracks[(0, 'cv32-core.function')]).as_pandas_dataframe().t[0]
        others = [c for t, c in kills.items() if t != 0] or list(kills.values())
        lo, hi = int(main0), max(others) * CYCLE_NS

        # Spatz busy: Snitch PC outside the runtime's idle code (dispatcher, traps, boot)
        idle, (base, end) = spatz_idle_pcs(test) if test.spatz_name else ([], (0, 0))
        spatz_busy = {}
        pc_tracks = {t: tid for (t, n), tid in tracks.items() if n == 'snitch-spatz.pc'}
        if pc_tracks:
            names = tp.query('select distinct name from slice where track_id in (%s)'
                             % ','.join(map(str, pc_tracks.values()))).as_pandas_dataframe().name
            idle_names = [n for n in names if not (base <= int(n, 16) < end)
                          or any(a <= int(n, 16) < b for a, b in idle)]
            idle_sql = ','.join("'%s'" % n for n in idle_names) or "''"
            # islands of consecutive busy PCs, one interval each
            q = tp.query(f"""
                with p as (select track_id, ts, dur, name not in ({idle_sql}) as b from slice
                           where track_id in ({','.join(map(str, pc_tracks.values()))})),
                     q as (select *, (b != lag(b, 1, -1) over (partition by track_id order by ts)) as chg from p),
                     r as (select *, sum(chg) over (partition by track_id order by ts rows unbounded preceding) as g from q)
                select track_id, min(ts) as a, max(ts + dur) as e from r where b group by track_id, g
            """).as_pandas_dataframe()
            inv = {tid: t for t, tid in pc_tracks.items()}
            for tid, grp in q.groupby('track_id'):
                spatz_busy[inv[tid]] = np.stack([grp.a.to_numpy(np.int64), grp.e.to_numpy(np.int64)], axis=1)

        per_tile = []
        for t in range(n_tiles):
            def get(name):
                return merge(clip(slices(t, name), lo, hi))
            rm = get('redmule.busy')
            din, dout = get('idma-axi2obi.be.busy'), get('idma-obi2axi.be.busy')
            vfu = get('snitch-spatz.vfu.active')
            sp = merge(np.r_[clip(spatz_busy.get(t, np.zeros((0, 2))), lo, hi), vfu])
            cv = get('cv32-core.busy')
            acc = merge(np.r_[rm, din, dout, sp])
            w = hi - lo
            empty = w - length(acc)
            cv_alone = length(cv) - overlap(cv, acc)
            per_tile.append(dict(redmule=length(rm) / w, idma=length(merge(np.r_[din, dout])) / w,
                                 idma_in=length(din) / w, idma_out=length(dout) / w, spatz=length(sp) / w,
                                 vfu=length(vfu) / w, empty=empty / w, cv32_active=cv_alone / w,
                                 cv32_sleep=(empty - cv_alone) / w))
        return lo // CYCLE_NS, hi // CYCLE_NS, per_tile
    finally:
        tp.close()


# --------------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------------

def fmt_cell(fmt, v):
    if v is None or v == '' or (isinstance(v, float) and np.isnan(v)):
        return '-'
    return fmt.format(v)


def table(rows):
    cells = [[h for h, _, _ in TABLE]] + [[fmt_cell(f, r.get(c)) for _, c, f in TABLE] for r in rows]
    w = [max(len(row[i]) for row in cells) for i in range(len(TABLE))]
    left = {'test', 'status'}
    line = lambda row: '  '.join(c.ljust(w[i]) if TABLE[i][0] in left else c.rjust(w[i])
                                 for i, c in enumerate(row))
    return '\n'.join([line(cells[0]), '  '.join('-' * x for x in w)] + [line(r) for r in cells[1:]])


class Report:
    def __init__(self, out):
        self.out = out
        self.logf = open(os.path.join(out, 'summary.log'), 'a')
        self.csvf = os.path.join(out, 'results.csv')
        with open(self.csvf, 'w', newline='') as f:
            csv.writer(f, lineterminator='\n').writerow(CSV_COLUMNS)

    def log(self, msg=''):
        print(msg, flush=True)
        self.logf.write(msg + '\n')
        self.logf.flush()

    def add(self, row):
        with open(self.csvf, 'a', newline='') as f:
            csv.writer(f, lineterminator='\n').writerow(
                ['' if row.get(c) is None else round(row[c], 4) if isinstance(row[c], float) else row[c]
                 for c in CSV_COLUMNS])


# --------------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------------

def expand(groups, tests):
    """List of (test, cfg, tiles) of all groups, without duplicates."""
    runs, seen = [], set()
    for g in groups:
        for name in g['tests']:
            if name not in tests:
                sys.exit('unknown test %r (see --list)' % name)
            t = tests[name]
            app = t.applicable()
            values = [g.get(p) or [t.defaults[p]] for p in app]
            for tiles in g['tiles']:
                for combo in itertools.product(*values):
                    cfg = {p: None for p in PARAMS}
                    cfg.update(zip(app, combo))
                    key = (name, tiles, tuple(cfg[p] for p in PARAMS))
                    if key not in seen:
                        seen.add(key)
                        runs.append((t, cfg, tiles))
    # tests and parameters together, so the sources change as little as possible
    runs.sort(key=lambda r: (r[0].name, [r[1][p] or 0 for p in PARAMS], r[2]))
    return runs


def on_sigterm(*_):
    raise KeyboardInterrupt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sweep', help='JSON sweep file (list of groups)')
    ap.add_argument('--tests', nargs='+', help='tests (directory names in tests/magia/mgc)')
    ap.add_argument('--tiles', nargs='+', type=int, help='mesh sides (default: 2)')
    for p in PARAMS:
        ap.add_argument('--' + p, nargs='+', type=int, help='%s values (default: the test\'s own)' % p)
    ap.add_argument('--out', help='output directory (default: mgc_sweep/<date>_<time>)')
    ap.add_argument('--cache', default=os.path.join(SWEEP_DIR, 'cache'),
                    help='cache of generated test.h (default: %(default)s)')
    ap.add_argument('--gen-python', default='python3', help='Python with torch for the golden generators')
    ap.add_argument('--mgcc-python', default=os.path.join(REPO, 'mgc', '.venv', 'bin', 'python')
                    if os.path.exists(os.path.join(REPO, 'mgc', '.venv', 'bin', 'python')) else 'python3',
                    help='Python with xDSL for mgcc (default: %(default)s)')
    ap.add_argument('--keep-traces', action='store_true', help='keep each run\'s perfetto trace')
    ap.add_argument('--dry-run', action='store_true', help='only list the configurations')
    ap.add_argument('--list', action='store_true', help='list the tests and their default parameters')
    args = ap.parse_args()

    tests = {n: Test(n) for n in list_tests()}
    if args.list:
        for t in tests.values():
            d = ' '.join('%s=%s' % (p, t.defaults[p]) for p in t.applicable())
            print('%-22s %-24s %s%s' % (t.name, t.target, d, '  goldens for tiles=%s' % t.goldens if t.goldens else ''))
        return 0

    if args.sweep:
        groups = json.loads(read(args.sweep))['groups']
    elif args.tests:
        groups = [{'tests': args.tests, **{p: getattr(args, p) for p in PARAMS if getattr(args, p)}}]
    else:
        ap.error('give --sweep FILE or --tests')
    for g in groups:
        g.setdefault('tiles', args.tiles or [2])
    runs = expand(groups, tests)

    if args.dry_run:
        for i, (t, cfg, tiles) in enumerate(runs):
            why = t.check(cfg, tiles)
            print('%3d  %-20s tiles=%-2d %s%s%s' % (i, t.name, tiles, ' '.join('%s=%s' % (p, cfg[p]) for p in t.applicable()),
                                                  '  [regenerate]' if t.needs_regen(cfg, tiles) else '',
                                                  '  SKIP: ' + why if why else ''))
        return 0

    out = args.out or os.path.join(SWEEP_DIR, datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(os.path.join(out, 'runs'), exist_ok=True)
    rep = Report(out)
    Sources.recover(rep.log)
    srcs = Sources(rep.log)
    signal.signal(signal.SIGTERM, on_sigterm)
    rep.log('mgc sweep: %d configurations, output in %s' % (len(runs), out))

    rows = []
    try:
        for i, (t, cfg, tiles) in enumerate(runs):
            rid = '%03d' % i
            rdir = os.path.join(out, 'runs', rid)
            os.makedirs(rdir, exist_ok=True)
            row = dict(run_id=rid, test=t.name, tiles=tiles, n_tiles=tiles * tiles, **cfg)
            desc = ' '.join('%s=%s' % (p, cfg[p]) for p in t.applicable())
            rep.log('[%d/%d] %s tiles=%d %s' % (i + 1, len(runs), t.name, tiles, desc))
            t0 = time.time()
            why = t.check(cfg, tiles)
            if why is None and t.needs_regen(cfg, tiles):
                why = generate(t, cfg, tiles, args, srcs, rep.log)
            elif why is None:
                for p in list(srcs.saved):  # back to the committed sources
                    shutil.copy2(srcs.saved[p], p)
            if why:
                row['status'] = 'SKIP'
                rep.log('  skipped: ' + why)
            else:
                rc, outp = run_make(t, tiles, os.path.join(rdir, 'make.log'))
                kills, errs = parse_run(outp)
                row['sim_seconds'] = round(time.time() - t0, 1)
                row['errors'] = sum(errs) if errs else None
                if rc or len(kills) != tiles * tiles:
                    row['status'] = 'BUILD/RUN FAIL' if rc else 'NO EXIT'
                    rep.log('  %s (rc=%d, %d/%d tiles exited), see %s' % (row['status'], rc, len(kills),
                                                                           tiles * tiles, rdir))
                else:
                    row['status'] = 'PASS' if errs and sum(errs) == 0 else 'FAIL'
                    trace = os.path.join(REPO, 'gvsoc_work', 'trace.perfetto-trace')
                    if args.keep_traces:
                        shutil.copy2(trace, rdir)
                    lo, hi, per_tile = analyze(t, trace, kills, tiles * tiles)
                    cyc = hi - lo
                    macs = t.macs(cfg)
                    mean = {k: 100 * np.mean([pt[k] for pt in per_tile]) for k in per_tile[0]}
                    row.update(cycles=cyc, macs=macs, mac_per_cycle=macs / cyc,
                               mac_per_cycle_per_tile=macs / cyc / (tiles * tiles),
                               redmule_pct=mean['redmule'], idma_pct=mean['idma'], idma_in_pct=mean['idma_in'],
                               idma_out_pct=mean['idma_out'], spatz_pct=mean['spatz'], spatz_vfu_pct=mean['vfu'],
                               empty_pct=mean['empty'], empty_cv32_active_pct=mean['cv32_active'],
                               empty_cv32_sleep_pct=mean['cv32_sleep'], main_start_cycle=lo, end_cycle=hi)
                    with open(os.path.join(rdir, 'per_tile.txt'), 'w') as f:
                        f.write('tile ' + ' '.join('%11s' % k for k in per_tile[0]) + '   (%)\n')
                        for k, pt in enumerate(per_tile):
                            f.write('%4d ' % k + ' '.join('%11.1f' % (100 * v) for v in pt.values()) + '\n')
                    rep.log('  %s: %s cycles, %.1f MAC/cycle, RedMulE %.1f%%, iDMA %.1f%%, Spatz %.1f%%, '
                            'empty %.1f%% (%.1fs)' % (row['status'], format(cyc, ','), macs / cyc,
                                                      mean['redmule'], mean['idma'], mean['spatz'],
                                                      mean['empty'], time.time() - t0))
            rows.append(row)
            rep.add(row)
    except KeyboardInterrupt:
        rep.log('interrupted')
    finally:
        srcs.restore()

    rep.log('')
    rep.log('Utilization: % of the cycles from tile 0\'s main to the last exit of tiles != 0, mean over tiles.')
    rep.log('MAC: matmuls only (softmax = 0). empty = no RedMulE/iDMA/Spatz activity: CV32 running '
            'alone (CV32 run) or asleep (CV32 sleep).')
    rep.log('')
    rep.log(table(rows))
    rep.log('')
    rep.log('CSV: ' + rep.csvf)
    return 0


if __name__ == '__main__':
    sys.exit(main())
