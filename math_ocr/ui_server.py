"""Local formula editor and CNN/LSTM inference service (no extra dependencies)."""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import threading
import time
from urllib.parse import urlsplit,parse_qs

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
import torch

from model import FormulaOCR, ModelConfig
from prepare_data import preprocess
from predict_best import parse_output,default_checkpoint
from decoding import generate

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "ui"
SAMPLES = ("test_00269", "test_00002", "test_00378")
MAX_BODY = 20 * 1024 * 1024
Image.MAX_IMAGE_PIXELS = 20_000_000


def png_data(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


class Recognizer:
    def __init__(self, checkpoint):
        torch.set_num_threads(4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.package = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if self.package["model_type"] != "seq2seq":
            raise ValueError("This UI requires the CNN/LSTM seq2seq package")
        self.model = FormulaOCR(ModelConfig(**self.package["config"]))
        self.model.load_state_dict(self.package["model"])
        self.model.to(self.device).eval()
        self.lock = threading.Lock()

    def status(self):
        from physics_syntax import GLYPHS,FUNCTIONS
        return {"ready": True, "device": str(self.device),
                "model": self.package["selected_variant"],
                "input_notation": self.package["input_notation"],
                "vocabulary": self.package["vocab"][4:],
                "syntax": self.package.get('syntax','legacy'),
                "preprocessing":self.package.get('preprocessing','default'),
                "evaluation":self.package.get('evaluation',{}),
                "glyphs":GLYPHS,"functions":sorted(FUNCTIONS)}

    def predict(self, raw):
        start = time.perf_counter()
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.width * opened.height > Image.MAX_IMAGE_PIXELS:
                raise ValueError("이미지가 너무 큽니다. 수식 부분을 더 작게 잘라 주세요.")
            source = ImageOps.exif_transpose(opened)
            if self.package.get('preprocessing')=='handwriting-v1':
                from handwriting_data import preprocess_handwriting
                processor=preprocess_handwriting
            else:processor=preprocess
            image = processor(source,height=self.package['config']['height'],max_width=self.package.get('max_width',384))
        with self.lock, torch.inference_mode():
            tensor = torch.from_numpy((255 - np.array(image)).astype(np.float32) / 255)
            tensor = tensor[None, None].to(self.device)
            ids = generate(self.model,self.package,tensor,torch.tensor([image.width]))[0].cpu().tolist()
        terminated = 2 in ids
        if terminated:
            ids = ids[:ids.index(2)]
        tokens = [self.package["vocab"][i] for i in ids]
        tree,converted=parse_output(tokens,self.package)
        return {"prediction": " ".join(tokens), "tokens": tokens,
                "latex": converted, "ast": tree,
                "syntax_valid": tree is not None, "terminated": terminated,
                "prepared_image": png_data(image), "prepared_size": list(image.size),
                "elapsed_ms": round((time.perf_counter() - start) * 1000),
                "model": self.package["selected_variant"], "device": str(self.device)}


class Models:
    def __init__(self,checkpoint):
        first=Recognizer(checkpoint)
        self.default='physics' if first.package.get('syntax')=='physics-v1' else 'legacy'
        self.loaded={self.default:first};self.paths={self.default:checkpoint}
        if self.default=='physics':self.paths['legacy']=ROOT/'comparison/best_model.pt'
        if (ROOT/'segmentation/models/settings.json').exists() and (ROOT/'segmentation/test_metrics.json').exists():self.paths['segments']=ROOT/'segmentation/models/settings.json'
        self.lock=threading.Lock()

    def get(self,key):
        if key not in self.paths:raise ValueError('Unknown model profile')
        with self.lock:
            if key not in self.loaded:
                if key=='segments':
                    from segmentation_ui import SegmentRecognizer
                    self.loaded[key]=SegmentRecognizer()
                else:self.loaded[key]=Recognizer(self.paths[key])
            return self.loaded[key]

    def samples(self,key):
        if key=='segments':return self.get(key).samples()
        if key=='legacy':
            return [{'id':name,'label':label,'symbol':symbol,'path':ROOT/'comparison/images/slash'/f'{name}.png'} for name,label,symbol in zip(SAMPLES,['근호','첨자','분수'],['√','xₙ','a / b'])]
        package=self.get(key).package
        if package.get('preprocessing')=='handwriting-v1':
            rows=[json.loads(x) for x in (ROOT/'data_handwriting/val.jsonl').read_text(encoding='utf-8').splitlines()]
            lookup={r['ink_id']:r for r in rows}
            return [{'id':'handwriting_'+str(i),'label':sample['label'],'symbol':sample['symbol'],
                     'path':ROOT/'data_handwriting'/lookup[sample['ink_id']]['image']} for i,sample in enumerate(package['demo_samples'])]
        rows=[json.loads(x) for x in (ROOT/'data_physics/physics_laws.jsonl').read_text(encoding='utf-8').splitlines()]
        choices=[('physics_energy',r'E=mc^{2}','에너지','E'),
                 ('physics_wave',r'\omega=2\pi f','그리스 문자','ω'),
                 ('physics_fraction',r'T=\frac{1}{f}','분수','1 / f')]
        return [{'id':name,'label':label,'symbol':symbol,'path':ROOT/'data_physics'/next(r['image'] for r in rows if r['formula']==formula and r['font']=='stix' and r['layout'] in ['native','stack'])} for name,formula,label,symbol in choices]


class Handler(BaseHTTPRequestHandler):
    def profile(self):
        key=parse_qs(urlsplit(self.path).query).get('profile',[self.server.models.default])[0]
        if key not in self.server.models.paths:raise ValueError('Unknown model profile')
        return key

    def reply(self, status, content, mime="application/json; charset=utf-8"):
        if isinstance(content, dict):
            content = json.dumps(content, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' data: blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path == "/api/status":
            try:
                key=self.profile();info=self.server.models.get(key).status()
            except ValueError:
                self.reply(400,{'error':'지원하지 않는 모델입니다.'});return
            names={'physics':'물리 수식 · 손글씨 (실험)','legacy':'기존 기본 수식 · P','segments':'기호 분할 · CNN (실험)'}
            info.update({'profile':key,'profiles':[{'id':p,'name':names[p]} for p in self.server.models.paths],
                         'samples':[{k:v for k,v in row.items() if k!='path'} for row in self.server.models.samples(key)]})
            self.reply(200, info)
            return
        if path.startswith("/api/sample/"):
            name = path.rsplit("/", 1)[-1]
            try:samples={row['id']:row['path'] for row in self.server.models.samples(self.profile())}
            except ValueError:samples={}
            if name not in samples:
                self.reply(404, {"error": "예제를 찾을 수 없습니다."})
                return
            self.reply(200, samples[name].read_bytes(), "image/png")
            return
        static = {"/": ("index.html", "text/html; charset=utf-8"),
                  "/favicon.svg": ("favicon.svg", "image/svg+xml"),
                  "/favicon.ico": ("favicon.svg", "image/svg+xml"),
                  "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/styles.css": ("styles.css", "text/css; charset=utf-8"),
                  "/editor.js": ("editor.js", "text/javascript; charset=utf-8")}
        if path not in static:
            self.reply(404, {"error": "페이지를 찾을 수 없습니다."})
            return
        name, mime = static[path]
        self.reply(200, (WEB / name).read_bytes(), mime)

    def do_POST(self):
        if urlsplit(self.path).path != "/api/recognize":
            self.reply(404, {"error": "요청을 찾을 수 없습니다."})
            return
        # A loopback-only service; reject cross-origin browser uploads.
        origin = self.headers.get("Origin")
        if origin and origin != "http://" + self.headers.get("Host", ""):
            self.reply(403, {"error": "이 페이지에서 다시 요청해 주세요."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                self.reply(413, {"error": "20MB 이하의 이미지 한 장을 넣어 주세요."})
                return
            if self.headers.get("Content-Type", "").split(";")[0] != "image/png":
                self.reply(415, {"error": "편집한 PNG 이미지가 필요합니다."})
                return
            result = self.server.models.get(self.profile()).predict(self.rfile.read(length))
            self.reply(200, result)
        except (ValueError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning, OSError) as exc:
            message = "수식이 비어 있습니다. 글자가 있는 영역을 선택해 주세요." if str(exc) == "Blank formula image" else "이미지를 읽을 수 없습니다. 다른 이미지로 다시 시도해 주세요."
            self.reply(400, {"error": message})
        except Exception:
            import traceback
            traceback.print_exc()
            self.reply(500, {"error": "인식 중 오류가 발생했습니다. 서버 실행 창을 확인해 주세요."})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--checkpoint", type=Path, default=default_checkpoint())
    parser.add_argument("--open", action="store_true", help="Open the editor in the default browser")
    args = parser.parse_args()
    print('Loading formula model...',flush=True)
    models=Models(args.checkpoint)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    server.models=models
    print(f"Formula Studio: http://127.0.0.1:{args.port} ({models.get(models.default).device})", flush=True)
    if args.open:
        import webbrowser
        webbrowser.open(f"http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
