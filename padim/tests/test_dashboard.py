import json
from pathlib import Path
import tempfile
import unittest

from padim_monitor.app import create_app
from padim_monitor.dashboard import ThermalReader
from padim_monitor.pipeline import Monitor, load_config

ROOT = Path(__file__).resolve().parents[2]


class DashboardTests(unittest.TestCase):
    def setUp(self):
        self.storage = tempfile.TemporaryDirectory()
        self.root = Path(self.storage.name)
        cfg = load_config(ROOT/'padim_monitor/config/config.json')
        cfg['events_dir'] = str(self.root/'events')
        self.events = Path(cfg['events_dir']); self.events.mkdir()
        self.monitor = Monitor(cfg)
        self.client = create_app(self.monitor).test_client()

    def tearDown(self):
        self.storage.cleanup()

    def event(self, index, complete=True):
        name = f'20261019_120000_{index:06d}'
        folder = self.events/name; folder.mkdir()
        for filename in ('original.jpg', 'heatmap.jpg', 'anomaly_map.npy'):
            (folder/filename).write_bytes(b'fixture')
        if complete:
            (folder/'event.json').write_text(json.dumps(dict(utc='2026-10-19T12:00:00+00:00', score=.5, threshold=.3, backend='nb')))
        return name

    def test_gallery_completion_pagination_and_downloads(self):
        self.assertEqual(self.client.get('/api/events').json['total'],0)
        for i in range(14): self.event(i)
        self.event(14,complete=False)
        first = self.client.get('/api/events?limit=12').json
        self.assertEqual(first['total'],14)
        self.assertEqual(len(first['events']),12)
        self.assertTrue(first['has_more'])
        before = first['events'][-1]['id']
        # Inserting a new event must not shift the older-events cursor.
        self.event(15)
        second = self.client.get('/api/events?before='+before).json
        self.assertEqual(len(second['events']),2)
        self.assertFalse(second['has_more'])
        for name, url in first['events'][0]['files'].items():
            with self.client.get(url) as response:
                self.assertEqual(response.status_code,200)
                if name.endswith('.npy'):
                    self.assertIn('attachment',response.headers['Content-Disposition'])
        (self.events/first['events'][0]['id']/'event.json').write_text('{')
        self.assertEqual(self.client.get('/api/events').json['total'],14)

    def test_invalid_paths_and_symlinks(self):
        name = self.event(1)
        for query in ('limit=bad','offset=-1','limit=49','before=../x'):
            self.assertEqual(self.client.get('/api/events?'+query).status_code,400)
        self.assertEqual(self.client.get('/events/bad/original.jpg').status_code,400)
        self.assertEqual(self.client.get('/events/'+name+'/secret.txt').status_code,400)
        image = self.events/name/'original.jpg'; image.unlink()
        outside = self.root/'outside.jpg'; outside.write_bytes(b'private')
        image.symlink_to(outside)
        self.assertEqual(self.client.get('/events/'+name+'/original.jpg').status_code,404)
        self.assertEqual(self.client.get('/api/events').json['total'],0)
        (self.events/'20261019_120000_000002').symlink_to(self.events/name,target_is_directory=True)
        self.assertEqual(self.client.get('/api/events').json['total'],0)

    def test_thermal_names_units_cache_and_unavailable(self):
        reader = ThermalReader(self.root/'thermal')
        self.assertIsNone(reader.read()['cpu_celsius'])
        def sensor(index, name, value):
            folder = self.root/'thermal'/f'thermal_zone{index}'
            folder.mkdir(parents=True)
            (folder/'type').write_text(name);(folder/'temp').write_text(value)
        sensor(0,'gpu_thermal_zone','55000')
        sensor(1,'npu_thermal_zone','61750')
        sensor(2,'cpu-thermal','53200')
        sensor(3,'cpu_big','58200')
        sensor(4,'cpu_bad','invalid')
        reader.expires = 0
        values = reader.read()
        self.assertEqual(values['npu_celsius'],61.8)
        self.assertEqual(values['cpu_celsius'],58.2)
        self.assertEqual(len(values['sensors']),3)
        (self.root/'thermal/thermal_zone1/temp').write_text('nan')
        self.assertEqual(reader.read()['npu_celsius'],61.8)
        reader.expires=0
        self.assertIsNone(reader.read()['npu_celsius'])

    def test_dashboard_status_has_temperatures(self):
        response = self.client.get('/api/status')
        self.assertIn('temperatures',response.json)
        self.assertIn('cpu_celsius',response.json['temperatures'])
        with self.client.get('/') as page:
            for element in (b'events-gallery',b'cpu-temperature',b'npu-temperature'):
                self.assertIn(element,page.data)


if __name__ == '__main__':
    unittest.main()
