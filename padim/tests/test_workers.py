import copy,json,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import cv2
from padim_monitor.pipeline import Monitor,Alarm,load_config,render_views
from padim_monitor.app import create_app
ROOT=Path(__file__).resolve().parents[2]
class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.cfg=load_config(ROOT/'padim_monitor/config/config.json')
        self.cfg['threshold']=.45
        self.cfg['stale_seconds']=2.0
        self.frame=np.zeros((720,1280,3),np.uint8)
        self.frame[:]=[20,80,190]
    def test_alarm_persistence_latch_reset(self):
        alarm=Alarm(.45,3)
        self.assertFalse(alarm.update(.5));self.assertFalse(alarm.update(.1))
        self.assertEqual(alarm.hits,0)
        self.assertFalse(alarm.update(.5));self.assertFalse(alarm.update(.5))
        self.assertTrue(alarm.update(.5));self.assertFalse(alarm.update(.6))
        alarm.update(.0);self.assertTrue(alarm.latched)
        alarm.reset();self.assertFalse(alarm.latched);self.assertEqual(alarm.hits,0)

    def test_fixed_heatmap_scale_and_modes(self):
        a=np.full((64,64),.3,np.float32)
        b=a.copy();b[0,0]=1
        one=render_views(self.frame,a,self.cfg)
        two=render_views(self.frame,b,self.cfg)
        self.assertEqual(set(one),{"original","heatmap","side-by-side"})
        self.assertEqual(one["side-by-side"].shape,self.frame.shape)
        # A hotspot elsewhere must not change the color of an unchanged region.
        np.testing.assert_array_equal(one["heatmap"][400:410,700:710],two["heatmap"][400:410,700:710])
        np.testing.assert_array_equal(one["heatmap"][:,:100],self.frame[:,:100])

    def test_routes_and_unknown_on_stale_data(self):
        monitor=Monitor(self.cfg)
        app=create_app(monitor);client=app.test_client()
        self.assertEqual(client.get("/").status_code,200)
        for name in ("style.css","app.js"):
            with client.get('/static/'+name) as response:self.assertEqual(response.status_code,200)
        self.assertEqual(client.get("/api/status").json["status"],"UNKNOWN")
        self.assertEqual(client.get("/api/frame.png").status_code,503)
        monitor.publish_camera(self.frame)
        now=time.monotonic()
        monitor.result=(self.frame,np.zeros((64,64),np.float32),.0,now,now)
        self.assertEqual(client.get("/api/status").json["status"],"NORMAL")
        snapshot = client.get("/api/frame.png")
        self.assertEqual(snapshot.status_code,200)
        np.testing.assert_array_equal(cv2.imdecode(np.frombuffer(snapshot.data,np.uint8),cv2.IMREAD_COLOR),self.frame)
        monitor.alarm.latched=True
        self.assertEqual(client.get("/api/status").json["status"],"ANOMALY")
        monitor.result=(*monitor.result[:3],now-3,now-3)
        stale=client.get("/api/status").json
        self.assertEqual(stale["status"],"UNKNOWN");self.assertTrue(stale["alarm"])
        self.assertEqual(client.post("/api/reset_alarm").status_code,415)
        self.assertFalse(client.post("/api/reset_alarm",json={}).json["alarm"])
        self.assertEqual(client.get("/video_feed?mode=bad").status_code,400)

    def test_consumers_share_encoded_frames(self):
        monitor=Monitor(self.cfg)
        monitor.jpegs={"heatmap":b"jpeg"};monitor.jpeg_sequence=1
        client=create_app(monitor).test_client()
        with client.get("/video_feed",buffered=False) as first, client.get("/video_feed",buffered=False) as second:
            a=next(first.response);b=next(second.response)
            self.assertEqual(a,b);self.assertIn(b"Content-Length: 4",a)
            self.assertEqual(monitor.result_sequence,0)  # HTTP did not run inference.
        monitor.stop()

    def test_slow_inference_uses_latest_frame_and_records_event(self):
        cfg=copy.deepcopy(self.cfg)
        with tempfile.TemporaryDirectory() as storage:
            cfg["events_dir"]=storage
            received=[];entered=threading.Event();release=threading.Event()
            class Backend:
                def __init__(self,cfg):pass
                def run(self,image):
                    received.append(float(image[0,0,0,0]))
                    if len(received)==1:
                        entered.set();release.wait(3)
                    return np.ones((64,64),np.float32),1.0
                def close(self):pass
            monitor=Monitor(cfg,Backend)
            # Replace only camera capture; exercise all other real workers.
            with patch.object(monitor,"_capture",side_effect=lambda:monitor.stop_event.wait()):
                monitor.start()
                try:
                    monitor.publish_camera(np.full((720,1280,3),10,np.uint8))
                    self.assertTrue(entered.wait(3))
                    for value in range(20,61,10):monitor.publish_camera(np.full((720,1280,3),value,np.uint8))
                    release.set()
                    deadline=time.monotonic()+3
                    while len(received)<2 and time.monotonic()<deadline:time.sleep(.01)
                    self.assertAlmostEqual(received[1],(60/255-.485)/.229,places=6)
                    monitor.publish_camera(np.full((720,1280,3),70,np.uint8))
                    while not monitor.status()["alarm"] and time.monotonic()<deadline:time.sleep(.01)
                    status = monitor.status()
                    self.assertTrue(status["alarm"])
                    stages = status["latency_ms"]
                    self.assertEqual(set(stages), {"preprocess_ms", "backend_ms", "validation_ms"})
                    self.assertAlmostEqual(sum(stages.values()), status["inference_ms"], places=5)
                    monitor.event_queue.join()
                    events=list(Path(storage).glob("*/event.json"))
                    self.assertEqual(len(events),1)
                    self.assertEqual(json.loads(events[0].read_text())["score"],1.0)
                finally:monitor.stop()

    def test_inflight_result_is_discarded_across_camera_reconnect(self):
        entered=threading.Event();release=threading.Event();calls=[]
        class Backend:
            def __init__(self,cfg):pass
            def run(self,image):
                calls.append(1)
                if len(calls)==1:
                    entered.set();release.wait(3)
                    return np.ones((64,64),np.float32),1.0
                return np.zeros((64,64),np.float32),0.0
            def close(self):pass
        monitor=Monitor(self.cfg,Backend)
        with patch.object(monitor,"_capture",side_effect=lambda:monitor.stop_event.wait()):
            monitor.start()
            try:
                monitor.publish_camera(self.frame)
                self.assertTrue(entered.wait(3))
                monitor._disconnect("test camera unplug")
                monitor.publish_camera(self.frame)
                release.set()
                deadline=time.monotonic()+3
                while monitor.result is None and time.monotonic()<deadline:time.sleep(.01)
                self.assertIsNotNone(monitor.result)
                self.assertEqual(monitor.result_sequence,2)  # disconnect + new result only
                self.assertEqual(monitor.status()["score"],0)
                self.assertEqual(monitor.alarm.hits,0)
            finally:
                release.set();monitor.stop()