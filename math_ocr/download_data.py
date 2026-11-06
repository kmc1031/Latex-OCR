"""Download the author's IM2LATEX Hugging Face export, pinned by revision/hash."""
from concurrent.futures import ThreadPoolExecutor
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
REPO = "yuntian-deng/im2latex-100k"
REVISION = "2cf5f2534bc6d9fb2a333cc99b46c0d331553200"


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    dest = ROOT / "data" / "raw"
    dest.mkdir(parents=True, exist_ok=True)
    revision = args.revision
    files = get_json(f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}/data")
    files = [f for f in files if f["path"].endswith(".parquet")]

    def download(entry):
        target = dest / Path(entry["path"]).name
        expected = entry["lfs"]["oid"]
        if target.exists() and hashlib.file_digest(target.open("rb"), "sha256").hexdigest() == expected:
            print(f"cached {target.name}", flush=True)
            return
        url = f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{entry['path']}?download=true"
        tmp = target.with_suffix(".partial")
        digest = hashlib.sha256()
        total = 0
        last_report = 0
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as stream:
            while chunk := response.read(1024 * 1024):
                stream.write(chunk)
                digest.update(chunk)
                total += len(chunk)
                if total - last_report >= 32 * 1024 * 1024:
                    print(f"{target.name}: {total / 1e6:.0f}/{entry['size'] / 1e6:.0f} MB", flush=True)
                    last_report = total
        if digest.hexdigest() != expected or total != entry["size"]:
            raise ValueError(f"Checksum/size mismatch: {target.name}")
        tmp.replace(target)
        print(f"downloaded {target.name} ({total / 1e6:.1f} MB)", flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, files))
    (dest / "provenance.json").write_text(json.dumps({
        "repository": REPO, "revision": revision, "files": files,
        "source": "https://github.com/harvardnlp/im2markup",
        "note": "This is the author's processed HF subset, not the complete original benchmark.",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
