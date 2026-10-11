"""Download CC0 Poly Haven surface maps; no executables or installers."""
import hashlib
import json
from pathlib import Path
from urllib.request import Request, build_opener, HTTPRedirectHandler
from urllib.parse import urlparse
def validate_url(url):
    parsed=urlparse(url)
    if parsed.scheme!='https' or parsed.hostname not in {'api.polyhaven.com','dl.polyhaven.org'} or parsed.username or parsed.password or parsed.port not in (None,443):
        raise ValueError('Unapproved surface URL')
class SurfaceRedirects(HTTPRedirectHandler):
    def redirect_request(self,request,response,code,message,headers,new_url):
        validate_url(new_url)
        return super().redirect_request(request,response,code,message,headers,new_url)
opener=build_opener(SurfaceRedirects())
def urlopen(request,timeout):
    validate_url(request.full_url)
    return opener.open(request,timeout=timeout)
ROOT = Path(__file__).resolve().parent / 'assets/city/downloads/surfaces'
ROOT.mkdir(parents=True, exist_ok=True)
records=[]
for asset in ['asphalt_02', 'concrete_floor_02']:
    with urlopen(Request(f'https://api.polyhaven.com/files/{asset}',headers={'User-Agent':'Mozilla/5.0'}), timeout=60) as response:
        info=json.load(response)
    for channel in ['Diffuse', 'nor_gl']:
        item=info[channel]['2k']['jpg']
        url=item['url']
        if not url.startswith('https://dl.polyhaven.org/file/ph-assets/Textures/jpg/2k/'):
            raise ValueError('Unexpected surface URL')
        target=ROOT/f'{asset}-{channel}.jpg'
        if not target.exists():
            with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0'}), timeout=90) as response:
                data=response.read(32*1024*1024+1)
            if len(data)!=item['size'] or hashlib.md5(data).hexdigest()!=item['md5']:
                raise ValueError('Surface integrity mismatch')
            target.write_bytes(data)
        data=target.read_bytes()
        if len(data)!=item['size'] or hashlib.md5(data).hexdigest()!=item['md5']:
            raise ValueError('Existing surface integrity mismatch')
        records.append({'url':url,'path':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
(ROOT/'sources.json').write_text(json.dumps(records,indent=2)+'\n')
