"""Download the official MathWriting archive with resumable local writes."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import time
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT=Path(__file__).resolve().parent/'data_mathwriting/raw'


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--excerpt',action='store_true');args=ap.parse_args()
    ROOT.mkdir(parents=True,exist_ok=True)
    name='mathwriting-2024'+('-excerpt' if args.excerpt else '')+'.tgz'
    url='https://storage.googleapis.com/mathwriting_data/'+name
    target=ROOT/name;partial=ROOT/(name+'.part')
    if target.is_file():print(f'Already downloaded: {target}',flush=True);return
    session=requests.Session();session.mount('https://',HTTPAdapter(max_retries=Retry(total=5,backoff_factor=1,status_forcelist=[429,500,502,503,504])))
    start=partial.stat().st_size if partial.exists() else 0
    response=session.get(url,headers={'Range':f'bytes={start}-'} if start else {},stream=True,timeout=(20,120));response.raise_for_status()
    if start and response.status_code!=206:start=0
    total=start+int(response.headers['Content-Length']);received=start;last=time.monotonic();began=last
    with partial.open('ab' if start else 'wb') as output:
        for chunk in response.iter_content(1024*1024):
            output.write(chunk);received+=len(chunk)
            if time.monotonic()-last>10:
                print(f'{received/1048576:.0f}/{total/1048576:.0f} MiB ({100*received/total:.1f}%)',flush=True);last=time.monotonic()
    if received!=total:raise ValueError('Incomplete download')
    md5=hashlib.md5();sha256=hashlib.sha256()
    with partial.open('rb') as source:
        for block in iter(lambda:source.read(4*1024*1024),b''):md5.update(block);sha256.update(block)
    google_hash=response.headers.get('x-goog-hash','')
    hashes=dict(piece.strip().split('=',1) for piece in google_hash.split(',') if '=' in piece)
    if 'md5' in hashes and base64.b64encode(md5.digest()).decode()!=hashes['md5']:raise ValueError('GCS MD5 mismatch')
    partial.replace(target)
    metadata={'url':url,'bytes':received,'sha256':sha256.hexdigest(),'google_md5':hashes.get('md5'),
              'google_md5_verified':'md5' in hashes,'license':'CC-BY-NC-SA-4.0',
              'source':'https://github.com/google-research/google-research/tree/master/mathwriting',
              'seconds':round(time.monotonic()-began,1)}
    (ROOT/(name+'.json')).write_text(json.dumps(metadata,indent=2),encoding='utf-8');print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()
