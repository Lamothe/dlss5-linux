"""Controlled presentation-interval benchmark; no GPU-pass timestamps."""
import ctypes as c
import csv
from datetime import datetime
import re
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / 'demo/DLSS_Sample_App/bin/ngx_dlss_demo/ReShade.log'
X = c.CDLL('libX11.so.6')
T = c.CDLL('libXtst.so.6')
X.XOpenDisplay.restype = c.c_void_p
X.XDefaultRootWindow.argtypes = [c.c_void_p]
X.XDefaultRootWindow.restype = c.c_ulong
X.XQueryTree.argtypes = [c.c_void_p, c.c_ulong, c.POINTER(c.c_ulong), c.POINTER(c.c_ulong), c.POINTER(c.POINTER(c.c_ulong)), c.POINTER(c.c_uint)]
X.XFetchName.argtypes = [c.c_void_p, c.c_ulong, c.POINTER(c.c_void_p)]
X.XFree.argtypes = [c.c_void_p]
X.XSetInputFocus.argtypes = [c.c_void_p, c.c_ulong, c.c_int, c.c_ulong]
X.XKeysymToKeycode.argtypes = [c.c_void_p, c.c_ulong]
X.XFlush.argtypes = [c.c_void_p]
X.XCloseDisplay.argtypes = [c.c_void_p]
T.XTestFakeKeyEvent.argtypes = [c.c_void_p, c.c_uint, c.c_int, c.c_ulong]


def toggle_off():
    display = X.XOpenDisplay(None)
    if not display:
        raise RuntimeError('Cannot open X display')
    def find(window):
        name = c.c_void_p()
        if X.XFetchName(display, window, c.byref(name)) and name.value:
            title = c.string_at(name).decode(errors='replace')
            X.XFree(name)
            if title.startswith('NVIDIA NGX DLSS Sample'):
                return window
        root, parent, children, count = c.c_ulong(), c.c_ulong(), c.POINTER(c.c_ulong)(), c.c_uint()
        if not X.XQueryTree(display, window, c.byref(root), c.byref(parent), c.byref(children), c.byref(count)):
            return None
        try:
            for i in range(count.value):
                found = find(children[i])
                if found:
                    return found
        finally:
            if children:
                X.XFree(children)
        return None
    try:
        window = find(X.XDefaultRootWindow(display))
        if not window:
            raise RuntimeError('Demo window not found; no key sent')
        X.XSetInputFocus(display, window, 1, 0)
        X.XFlush(display)
        time.sleep(1)
        key = X.XKeysymToKeycode(display, 0xffc7)  # F10
        T.XTestFakeKeyEvent(display, key, 1, 0)
        X.XFlush(display)
        time.sleep(0.15)
        T.XTestFakeKeyEvent(display, key, 0, 0)
        X.XFlush(display)
    finally:
        X.XCloseDisplay(display)


def summarise(path):
    lines = path.read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith('fps,frametime,'))
    rows = list(csv.DictReader(lines[header:]))
    values = [float(row['frametime']) for row in rows]
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise RuntimeError('Invalid frame intervals')
    elapsed = float(rows[-1]['elapsed']) - float(rows[0]['elapsed'])
    if elapsed < 19_000_000_000:
        raise RuntimeError(f'Capture too short: {elapsed} ns')
    ordered = sorted(values)
    percentile = lambda p: ordered[math.ceil(p * len(ordered)) - 1]
    return {'frames': len(values), 'mean_ms': statistics.mean(values), 'median_ms': statistics.median(values),
            'p95_ms': percentile(.95), 'p99_ms': percentile(.99), 'fps': 1000 / statistics.mean(values),
            'gpu_temp_mean_c': statistics.mean(float(r['gpu_temp']) for r in rows),
            'gpu_load_mean_percent': statistics.mean(float(r['gpu_load']) for r in rows)}


def run(width, height, mode, index, output):
    directory = output / f'{width}x{height}-{mode}-{index}'
    directory.mkdir()
    if LOG.exists():
        shutil.copy2(LOG, directory / 'previous-ReShade.log')
        LOG.unlink()
    launched = time.time()
    (directory / 'metadata.json').write_text(json.dumps({'launched': launched, 'mode': mode, 'width': width, 'height': height}))
    env = os.environ | {'DLSS_TEST_NEURAL': '1', 'DLSS_TEST_LOG_LEVEL': 'error', 'DLSS_BENCHMARK': '1',
        'MANGOHUD_LOG_LEVEL': 'critical', 'DXVK_LOG_LEVEL': 'error',
        'MANGOHUD_CONFIG': f'fps,autostart_log=30,log_duration=20,log_interval=0,output_folder={directory}'}
    with (directory / 'process.txt').open('w') as log:
        process = subprocess.Popen(['bash', str(ROOT / 'run-contained.sh'), '-width', str(width), '-height', str(height)],
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            time.sleep(8)
            if process.poll() is not None:
                raise RuntimeError('Demo exited during warm-up')
            if mode == 'off':
                toggle_off()
                time.sleep(1)
                if 'nr-fwd: pass DISABLED' not in LOG.read_text():
                    raise RuntimeError('Neural toggle not confirmed')
            time.sleep(46)
            if process.poll() is not None:
                raise RuntimeError('Demo exited during capture')
            text = LOG.read_text()
            if LOG.stat().st_mtime < launched:
                raise RuntimeError('Stale ReShade log')
            geometries = re.findall(r'swapchain (?:created|resized): (\d+)x(\d+) ', text)
            outputs = re.findall(r'Output = \S+ (\d+)x(\d+) ', text)
            expected = (str(width), str(height))
            if not geometries or any(g != expected for g in geometries) or not outputs or any(g != expected for g in outputs):
                raise RuntimeError('Swapchain or neural output resolution mismatch/change')
            if re.search(r'giving up|consecutive failures|pass disabled|EvaluateFeature.*(?!Success)Failure', text):
                raise RuntimeError('Neural failure or automatic disable')
            if 'CreateFeature(18) => 0x1 (Success)' not in text:
                raise RuntimeError('Neural feature not created')
            if 'nr-fwd: pass ENABLED' in text or (mode == 'on' and 'nr-fwd: pass DISABLED' in text):
                raise RuntimeError('Unexpected toggle')
            if 'nr-fwd: EvaluateFeature #1 => 0x1 (Success)' not in text:
                raise RuntimeError('Neural evaluation not confirmed')
            shutil.copy2(LOG, directory / 'ReShade.log')
            files = [p for p in directory.glob('*.csv') if not p.name.endswith('_summary.csv')]
            if len(files) != 1:
                raise RuntimeError(f'Expected one capture, got {files}')
            capture_start = datetime.strptime(files[0].stem.split('_', 3)[-1], '%Y-%m-%d_%H-%M-%S')
            start_seconds = capture_start.hour * 3600 + capture_start.minute * 60 + capture_start.second
            events = []
            for line in text.splitlines():
                match = re.match(r'(\d+):(\d+):(\d+):(\d+).*nr-fwd: EvaluateFeature #\d+ => (.*)', line)
                if match:
                    h, m, sec, ms = map(int, match.groups()[:4])
                    events.append((h * 3600 + m * 60 + sec + ms / 1000, match.group(5)))
            if any('Success' not in status for _, status in events):
                raise RuntimeError('Neural evaluation failure')
            if mode == 'on':
                during = [t for t, _ in events if start_seconds <= t <= start_seconds + 21]
                if not during or during[0] > start_seconds + 10 or during[-1] < start_seconds + 10:
                    raise RuntimeError('Missing neural heartbeats spanning capture')
            elif any(t >= start_seconds - 1 for t, _ in events):
                raise RuntimeError('Neural evaluation resumed during off capture')
            result = {'resolution': f'{width}x{height}', 'mode': mode, 'repeat': index, **summarise(files[0])}
            (directory / 'result.json').write_text(json.dumps(result, indent=2))
            from validate import validate
            validate(directory)
            print(json.dumps(result), flush=True)
            return result
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == '__main__':
    if subprocess.run(['pgrep', '-f', '[n]gx_dlss_demo.exe'], capture_output=True).returncode == 0:
        raise SystemExit('Close existing demo before benchmarking')
    output = ROOT / 'benchmark' / time.strftime('capture-%Y%m%d-%H%M%S')
    output.mkdir()
    results = []
    for width, height in [(2560, 1358), (1280, 720)]:
        for index in range(1, 4):
            for mode in (['off', 'on'] if index % 2 else ['on', 'off']):
                results.append(run(width, height, mode, index, output))
                (output / 'results.json').write_text(json.dumps(results, indent=2))
                time.sleep(2)
    print(f'Saved to {output}', flush=True)
