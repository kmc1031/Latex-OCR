"""Integration checks against a running ui_server.py; uses the real checkpoint."""
import base64
import io
import json
from pathlib import Path
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from PIL import Image

ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8765"


def request(path, data=None, **headers):
    if path.startswith('/api/') and 'profile=' not in path:path+=('&' if '?' in path else '?')+'profile=legacy'
    try:
        response = urlopen(Request(URL + path, data=data, headers=headers), timeout=30)
    except HTTPError as error:
        response = error
    with response:
        return response.status, response.read()


class UIIntegration(unittest.TestCase):
    def test_model_status_and_static_page(self):
        code, body = request("/api/status")
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["model"], "P")
        code, body = request("/")
        self.assertEqual(code, 200)
        self.assertIn(b"/app.js", body)

    def test_real_inference_and_prepared_image(self):
        raw = (ROOT / "comparison/images/slash/test_00002.png").read_bytes()
        code, body = request("/api/recognize", raw, **{"Content-Type": "image/png"})
        self.assertEqual(code, 200)
        result = json.loads(body)
        self.assertEqual(result["prediction"], "y _ { 6 } + c + x")
        self.assertTrue(result["syntax_valid"])
        self.assertTrue(result["terminated"])
        self.assertIn("_{6}", result["latex"])
        prepared = Image.open(io.BytesIO(base64.b64decode(result["prepared_image"].split(",")[1])))
        self.assertEqual(prepared.height, 64)
        self.assertLessEqual(prepared.width, 384)
        self.assertEqual(prepared.width % 4, 0)

    def test_transparent_margin_does_not_change_model_input(self):
        image = Image.open(ROOT / "comparison/images/slash/test_00002.png").convert("RGBA")
        wrapped = Image.new("RGBA", (image.width + 100, image.height + 80), (0, 0, 0, 0))
        wrapped.paste(image, (50, 40))
        buffer = io.BytesIO()
        wrapped.save(buffer, "PNG")
        code, body = request("/api/recognize", buffer.getvalue(), **{"Content-Type": "image/png"})
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(body)["prediction"], "y _ { 6 } + c + x")

    def test_empty_and_invalid_images_are_recoverable_errors(self):
        buffer = io.BytesIO()
        Image.new("L", (80, 64), 255).save(buffer, "PNG")
        for raw in [buffer.getvalue(), b"not a png"]:
            code, body = request("/api/recognize", raw, **{"Content-Type": "image/png"})
            self.assertEqual(code, 400)
            self.assertIn("error", json.loads(body))
        self.assertEqual(request("/api/status")[0], 200)

    def test_only_app_uploads_and_whitelisted_files(self):
        self.assertEqual(request("/api/recognize", b"x", **{"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(request("/api/recognize", b"x", **{"Content-Type": "image/png", "Origin": "https://example.com"})[0], 403)
        self.assertEqual(request("/comparison/best_model.pt")[0], 404)
        self.assertEqual(request("/api/sample/unknown")[0], 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
