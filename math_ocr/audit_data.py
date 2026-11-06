"""Independent audit of the saved images and split manifests."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent


def main():
    result = {}
    names = ["train", "val", "test", "long_test"]
    for folder in ["data", "data_controlled"]:
        data = ROOT / folder
        records = {s:[json.loads(x) for x in (data/f"{s}.jsonl").read_text(encoding="utf-8").splitlines()] for s in names}
        sets = {s:{} for s in names}
        for s, rows in records.items():
            for field in ["latex", "source_filename", "image_sha256"]:
                sets[s][field] = {r[field] for r in rows}
                assert len(sets[s][field]) == len(rows), (folder,s,field,"within-split duplicates")
            for r in rows:
                with Image.open(data/r["image"]) as image:
                    assert image.height == 64 and image.width <= 384 and image.width % 4 == 0
                    digest = hashlib.sha256(np.asarray(image).tobytes()+str(image.size).encode()).hexdigest()
                    assert digest == r["image_sha256"], (folder,s,r["image"],"pixel checksum")
                    assert np.asarray(image).min() < 200, "blank image"
        overlap=[]
        for i,a in enumerate(names):
            for b in names[i+1:]:
                counts={field:len(sets[a][field] & sets[b][field]) for field in sets[a]}
                assert not any(counts.values()), (folder,a,b,counts)
                overlap.append({"a":a,"b":b,"overlap":counts})
        result[folder] = {"samples":{s:len(rows) for s,rows in records.items()},
                          "all_saved_pixel_checksums_match":True,"split_overlaps":overlap}
    (ROOT/"data_audit.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
    print("PASS: all image checksums, shapes, nonblank contents, within/across split duplicates.")


if __name__=="__main__":
    main()
