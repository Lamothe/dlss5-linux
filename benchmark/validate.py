"""Post-capture validation of logged state, not a guarantee of no unlogged failures."""
import csv
from datetime import datetime, timedelta
import json
from pathlib import Path
import re
import sys


def validate(directory):
    files = [p for p in directory.glob('*.csv') if not p.name.endswith('_summary.csv')]
    if len(files) != 1:
        raise ValueError('Expected one CSV')
    start = datetime.strptime(files[0].stem.split('_', 3)[-1], '%Y-%m-%d_%H-%M-%S')
    lines = files[0].read_text().splitlines()
    header = next(i for i, line in enumerate(lines) if line.startswith('fps,frametime,'))
    rows = list(csv.DictReader(lines[header:]))
    duration = (float(rows[-1]['elapsed']) - float(rows[0]['elapsed'])) / 1e9
    end = start + timedelta(seconds=duration)
    result = json.loads((directory / 'result.json').read_text())
    expected = tuple(result['resolution'].split('x'))
    text = (directory / 'ReShade.log').read_text()
    dimensions = re.findall(r'swapchain (?:created|resized): (\d+)x(\d+) ', text)
    outputs = re.findall(r'Output = \S+ (\d+)x(\d+) ', text)
    if not dimensions or not outputs or any(g != expected for g in dimensions + outputs):
        raise ValueError('Geometry mismatch/change')
    if re.search(r'giving up|consecutive failures|pass disabled', text):
        raise ValueError('Logged neural failure/disable')
    day = start.date()
    previous = None
    events, toggles = [], []
    for line in text.splitlines():
        match = re.match(r'(\d+):(\d+):(\d+):(\d+)', line)
        if not match:
            continue
        h, m, s, ms = map(int, match.groups())
        stamp = datetime.combine(day, datetime.min.time()).replace(hour=h, minute=m, second=s, microsecond=ms*1000)
        # Initial log can precede a capture just after midnight.
        if previous is None and stamp > start + timedelta(hours=12):
            day -= timedelta(days=1)
            stamp -= timedelta(days=1)
        if previous is not None and stamp < previous - timedelta(hours=12):
            day += timedelta(days=1)
            stamp += timedelta(days=1)
        previous = stamp
        if 'nr-fwd: EvaluateFeature #' in line:
            if '=> 0x1 (Success)' not in line:
                raise ValueError('Logged evaluation failure')
            events.append(stamp)
        if 'nr-fwd: pass ' in line:
            toggles.append((stamp, line))
    if result['mode'] == 'off':
        disabled = [stamp for stamp, line in toggles if 'DISABLED' in line]
        if len(disabled) != 1 or (start-disabled[0]).total_seconds() < 15:
            raise ValueError('Insufficient off-condition warm-up')
        if any(stamp >= start-timedelta(seconds=1) for stamp in events):
            raise ValueError('Evaluation during off capture')
        if any('ENABLED' in line for _, line in toggles):
            raise ValueError('Unexpected enable')
    else:
        if toggles:
            raise ValueError('Toggle during on run')
        # 600 evaluations take ~7.4s at the slower tested resolution.
        # Require bracketing evidence and no gap >10s, including capture edges.
        during = [stamp for stamp in events if start <= stamp <= end]
        if not during:
            raise ValueError('No neural heartbeat during capture')
        gaps = [(b-a).total_seconds() for a,b in zip([start]+during, during+[end])]
        if max(gaps) > 10:
            raise ValueError(f'Neural heartbeat coverage gap: {max(gaps)}s')
    return {'directory': directory.name, 'duration_s': duration, 'logged_state_valid': True,
            'limit': 'Sparse heartbeats cannot exclude unlogged transient neural failures'}


if __name__ == '__main__':
    output = Path(sys.argv[1])
    results = [validate(p.parent) for p in sorted(output.glob('*/result.json'))]
    if len(results) != 12:
        raise SystemExit(f'Expected 12 completed captures, got {len(results)}')
    (output / 'validation.json').write_text(json.dumps(results, indent=2))
    print('Validated all 12 captures: geometry, logged state, warm-up and heartbeat coverage')
