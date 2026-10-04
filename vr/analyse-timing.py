#!/usr/bin/env python3
"""Validate complete XR timing captures; summarise post-warm-up samples as JSON."""
import argparse
import csv
import io
import json
import math
from pathlib import Path
import re

HEADER = 'frame,resolution,neural,scene_ms,neural_ms,compose_ms,gpu_ms,frame_wall_ms,wait_ms,cadence_ms,period_ms'
METRICS = HEADER.split(',')[3:]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarise(path: Path, warmup: int) -> dict:
    text = path.read_text()
    passes = re.findall(r'^PASS: XR submitted frames=(\d+) neural=([01])\.', text, re.MULTILINE)
    if len(passes) != 1 or passes[0][0] != '600':
        raise ValueError(f'{path}: requires exactly one complete 600-frame PASS')
    headers = [line.removeprefix('TIMING_HEADER ') for line in text.splitlines() if line.startswith('TIMING_HEADER ')]
    if headers != [HEADER]:
        raise ValueError(f'{path}: missing, duplicate or unexpected header')
    lines = [line.removeprefix('TIMING ') for line in text.splitlines() if line.startswith('TIMING ')]
    records = list(csv.DictReader(io.StringIO(HEADER + '\n' + '\n'.join(lines))))
    if len(records) != 600 or not 0 <= warmup < 600:
        raise ValueError(f'{path}: incomplete capture or invalid warm-up')
    modes: set[int] = set()
    resolutions: set[int] = set()
    samples = []
    for frame, record in enumerate(records, 1):
        if set(record) != set(HEADER.split(',')) or None in record.values():
            raise ValueError(f'{path}: malformed row {frame}')
        if int(record['frame']) != frame:
            raise ValueError(f'{path}: non-sequential frames')
        modes.add(int(record['neural']))
        resolutions.add(int(record['resolution']))
        values = {metric: float(record[metric]) for metric in METRICS}
        if any(not math.isfinite(value) or value < 0 for value in values.values()):
            raise ValueError(f'{path}: invalid numeric sample {frame}')
        if values['gpu_ms'] <= 0 or values['period_ms'] <= 0 or values['frame_wall_ms'] <= 0:
            raise ValueError(f'{path}: missing positive timing {frame}')
        if sum(values[name] for name in ('scene_ms', 'neural_ms', 'compose_ms')) > values['gpu_ms'] + .00001:
            raise ValueError(f'{path}: stages exceed total GPU span {frame}')
        if frame > warmup:
            samples.append(values)
    if modes != {int(passes[0][1])} or len(resolutions) != 1 or not resolutions <= {720, 1080, 1440}:
        raise ValueError(f'{path}: inconsistent mode/resolution')
    stats = {}
    for metric in METRICS:
        # Zero cadence denotes a first frame or a discontinuity, not infinite speed.
        values = [sample[metric] for sample in samples if metric != 'cadence_ms' or sample[metric] > 0]
        stats[metric] = None if not values else {
            'samples': len(values), 'mean': sum(values) / len(values),
            'p50': percentile(values, .5), 'p95': percentile(values, .95),
            'p99': percentile(values, .99), 'max': max(values),
        }
    cadence_samples = [sample for sample in samples if sample['cadence_ms'] > 0]
    return {
        'file': str(path), 'resolution': next(iter(resolutions)), 'neural': next(iter(modes)),
        'warmup_frames': warmup, 'measured_frames': len(samples), 'metrics_ms': stats,
        'gpu_span_exceeds_display_period': sum(sample['gpu_ms'] > sample['period_ms'] for sample in samples),
        'frame_wall_exceeds_display_period': sum(sample['frame_wall_ms'] > sample['period_ms'] for sample in samples),
        'cadence_samples': len(cadence_samples),
        'cadence_above_1_5_periods': sum(sample['cadence_ms'] > 1.5 * sample['period_ms'] for sample in cadence_samples),
        'note': 'Budget/cadence flags are not confirmed compositor drops or headset latency. Frame wall time includes CPU work and blocking GPU/runtime waits. Captures require separate launcher/Proton diagnostic checks.',
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('captures', type=Path, nargs='+')
    parser.add_argument('--warmup', type=int, default=120)
    args = parser.parse_args()
    try:
        print(json.dumps([summarise(path, args.warmup) for path in args.captures], indent=2))
    except (ValueError, OSError) as error:
        parser.exit(1, f'{error}\n')
