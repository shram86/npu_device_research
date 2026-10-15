"""Camera-free route checks: python3 -m unittest -v test_motion_web.py."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


class DashboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.storage = tempfile.TemporaryDirectory()
        source = Path(__file__).with_name("motion_web.py")
        spec = importlib.util.spec_from_file_location("motion_web_test", source)
        cls.web = importlib.util.module_from_spec(spec)
        # Flask resolves resource paths from the module's file location.
        import sys
        sys.modules[spec.name] = cls.web
        with patch.dict(os.environ, {"MOTION_DATA_DIR": cls.storage.name}):
            spec.loader.exec_module(cls.web)
        cls.client = cls.web.app.test_client()

    @classmethod
    def tearDownClass(cls):
        cls.storage.cleanup()

    def test_dashboard_and_static_assets(self):
        page = self.client.get("/")
        self.assertEqual(page.status_code, 200)
        self.assertIn(b"No completed motion events yet", page.data)
        self.assertNotIn(b"<style>", page.data)
        for filename in ("dashboard.css", "dashboard.js"):
            self.assertIn(f"/static/{filename}".encode(), page.data)
            with self.client.get(f"/static/{filename}") as asset:
                self.assertEqual(asset.status_code, 200)

    def test_events_pagination_and_downloads(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            for index in range(14):
                event = base / f"20261015_120000_{index:06d}"
                event.mkdir()
                (event / "peak.jpg").write_bytes(b"snapshot")
                (event / "motion.avi").write_bytes(b"video")
            (base / "unfinished").mkdir()
            with patch.object(self.web, "BASE", base):
                first = self.client.get("/api/events?limit=12").json
                second = self.client.get("/api/events?offset=12&limit=12").json
                self.assertEqual(first["total"], 14)
                self.assertTrue(first["has_more"])
                self.assertEqual(len(second["events"]), 2)
                self.assertFalse(second["has_more"])
                event_id = first["events"][0]["id"]
                response = self.client.get(f"/events/{event_id}/motion.avi")
                self.assertEqual(response.data, b"video")
                self.assertIn("attachment", response.headers["Content-Disposition"])
                response.close()
                self.assertEqual(self.client.get("/api/status").json["events"], 14)
                self.assertIn(event_id.encode(), self.client.get("/").data)

    def test_invalid_requests(self):
        self.assertEqual(self.client.get("/api/events?offset=invalid").status_code, 400)
        self.assertEqual(self.client.get("/events/not-valid/peak.jpg").status_code, 400)
        self.assertEqual(self.client.get("/events/123/secret.txt").status_code, 400)
        self.assertEqual(self.client.get("/events/123/peak.jpg").status_code, 404)

    def test_stream_framing(self):
        with patch.object(self.web, "latest_jpeg", b"jpeg"), patch.object(self.web, "running", True):
            response = self.client.get("/stream", buffered=False)
            self.assertIn("multipart/x-mixed-replace", response.content_type)
            self.assertEqual(next(response.response), b"--frame\r\nContent-Type: image/jpeg\r\n\r\njpeg\r\n")
            response.close()


if __name__ == "__main__":
    unittest.main()
