"""Saved event browsing and Linux thermal sensors, outside camera workers."""
import json
import math
from pathlib import Path
import re
import threading
import time

EVENT_ID = re.compile(r'\d{8}_\d{6}_\d{6}\Z', re.ASCII)
EVENT_FILES = {'original.jpg', 'heatmap.jpg', 'anomaly_map.npy', 'event.json'}


class EventStore:
    def __init__(self, root):
        self.root = Path(root).resolve()

    def file(self, event_id, filename):
        if not EVENT_ID.fullmatch(event_id) or filename not in EVENT_FILES:
            raise ValueError('Invalid event or filename')
        folder = self.root/event_id
        path = folder/filename
        if folder.is_symlink() or path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        if path.resolve().parent != folder or folder.parent != self.root:
            raise FileNotFoundError(path)
        return path

    def entries(self):
        try:
            folders = sorted(self.root.iterdir(), key=lambda p:p.name, reverse=True)
        except FileNotFoundError:
            return []
        events = []
        for folder in folders:
            if not EVENT_ID.fullmatch(folder.name) or not folder.is_dir() or folder.is_symlink():
                continue
            try:
                # event.json is the completion marker; partial events stay hidden.
                metadata = json.loads(self.file(folder.name, 'event.json').read_text())
                self.file(folder.name, 'original.jpg')
                self.file(folder.name, 'heatmap.jpg')
                score, threshold = float(metadata['score']), float(metadata['threshold'])
                if not math.isfinite(score) or not math.isfinite(threshold):
                    continue
                events.append(dict(id=folder.name, utc=metadata.get('utc', folder.name),
                                   score=score, threshold=threshold, backend=metadata.get('backend', 'unknown')))
            except (OSError, ValueError, KeyError, TypeError):
                continue
        return events


class ThermalReader:
    """Cache sysfs millidegree readings; never guess sensor identity by zone number."""
    def __init__(self, root='/sys/class/thermal'):
        self.root = Path(root)
        self.lock = threading.Lock()
        self.expires = 0
        self.cached = dict(cpu_celsius=None, npu_celsius=None, sensors=[])

    def read(self):
        with self.lock:
            now = time.monotonic()
            if now < self.expires:
                return self.cached
            sensors = []
            for zone in sorted(self.root.glob('thermal_zone*')):
                try:
                    name = (zone/'type').read_text().strip()
                    kind = name.lower()
                    category = 'npu' if 'npu' in kind else 'cpu' if 'cpu' in kind else None
                    if category is None:
                        continue
                    value = float((zone/'temp').read_text().strip())/1000
                    if not math.isfinite(value) or not -40 <= value <= 150:
                        continue
                    sensors.append(dict(kind=category, name=name, celsius=round(value, 1)))
                except (OSError, ValueError):
                    continue
            self.cached = dict(sensors=sensors)
            for category in ('cpu', 'npu'):
                values = [s['celsius'] for s in sensors if s['kind'] == category]
                self.cached[category+'_celsius'] = max(values) if values else None
            self.expires = now+2
            return self.cached
