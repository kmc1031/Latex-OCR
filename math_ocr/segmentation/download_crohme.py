"""Fetch the official CROHME archive; keep provenance and avoid extraction."""
from html.parser import HTMLParser
import hashlib,json
from pathlib import Path
from urllib.parse import urljoin
import requests

ROOT=Path(__file__).resolve().parent
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        if tag=='a':self.links.extend(v for k,v in attrs if k=='href')

def main():
    folder=ROOT/'raw';folder.mkdir(parents=True,exist_ok=True)
    target=folder/'crohme2014.zip'
    if target.exists():print('Already downloaded');return
    page='https://tc11.cvc.uab.es/datasets/CROHME-2014_2/gt_1_1'
    try:
        response=requests.get(page,timeout=8);response.raise_for_status()
        parser=Links();parser.feed(response.text)
        urls=[urljoin(page,u) for u in parser.links if '.zip' in u.lower()]
        if not urls:raise RuntimeError('Official download link not found')
        url=next((u for u in urls if '2014' in u),urls[0]);source='official'
    except (requests.RequestException,RuntimeError):
        repository='ThomasLech/CROHME_extractor'
        revision=requests.get('https://api.github.com/repos/'+repository+'/commits/master',timeout=20).json()['sha']
        url=f'https://raw.githubusercontent.com/{repository}/{revision}/data/CROHME_full_v2.zip';source='public extractor mirror, pinned commit'
    print('Downloading',url,flush=True)
    response=requests.get(url,timeout=(20,90),stream=True);response.raise_for_status()
    digest=hashlib.sha256();size=0
    with target.with_suffix('.part').open('wb') as output:
        for block in response.iter_content(1024*1024):output.write(block);digest.update(block);size+=len(block)
    target.with_suffix('.part').replace(target)
    (folder/'provenance.json').write_text(json.dumps({'page':page,'url':url,'source':source,'bytes':size,'sha256':digest.hexdigest(),
        'license':'See official CROHME TC11 dataset page; non-commercial academic use.'},indent=2),encoding='utf-8')
    print('Downloaded',size,'bytes',flush=True)

if __name__=='__main__':main()
