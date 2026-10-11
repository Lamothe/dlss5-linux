"""Fetch a small allowlisted set of city assets; payloads stay git-ignored.
Run from repository root with Python 3. No accounts, installers or scripts executed.
"""
import concurrent.futures
import hashlib
import json
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler

ROOT = Path(__file__).resolve().parent / 'assets' / 'city' / 'downloads'
ROOT.mkdir(parents=True, exist_ok=True)


def validate_url(url):
    parsed = urlparse(url)
    if parsed.scheme != 'https' or parsed.hostname not in {'dl.polyhaven.org', 'raw.githubusercontent.com'} or parsed.username or parsed.password:
        raise ValueError('Unapproved download URL')


class AllowlistedRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, response, code, message, headers, new_url):
        validate_url(new_url)
        return super().redirect_request(request, response, code, message, headers, new_url)


opener = build_opener(AllowlistedRedirects())


def fetch(url, destination):
    validate_url(url)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        temporary = destination.with_suffix(destination.suffix + '.part')
        try:
            with opener.open(Request(url, headers={'User-Agent': 'dlss5-vr-city-test/1.0'}), timeout=90) as response, temporary.open('wb') as output:
                total = 0
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > 128 * 1024 * 1024:
                        raise ValueError('Asset exceeds 128 MiB limit')
                    output.write(chunk)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    return {'url': url, 'path': str(destination.relative_to(ROOT)), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}


records = []
for asset in ['modular_urban_apartments_facade', 'covered_car']:
    base = f'https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/{asset}/'
    path = ROOT / asset / f'{asset}_1k.gltf'
    records.append(fetch(base + path.name, path))
    model = json.loads(path.read_text())
    dependencies = [item['uri'] for key in ('buffers', 'images') for item in model.get(key, []) if 'uri' in item]
    jobs = []
    for uri in sorted(set(dependencies)):
        if urlparse(uri).scheme or uri.startswith('/') or '..' in Path(uri).parts:
            raise ValueError('Unsafe glTF relative dependency')
        download_url = (f'https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/{asset}/' + Path(uri).name) if uri.startswith('textures/') else urljoin(base.replace('/1k/', '/8k/'), uri)
        jobs.append((download_url, path.parent / uri))
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        records.extend(executor.map(lambda job: fetch(*job), jobs))

base = 'https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/CarConcept/'
for remote, local in [('glTF-Binary/CarConcept.glb', 'CarConcept.glb'), ('LICENSE.md', 'LICENSE.md'), ('README.md', 'README.md')]:
    records.append(fetch(base + remote, ROOT / 'car-concept' / local))

base = 'https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/'
records.append(fetch(base + 'LICENSE.md', ROOT / 'rocketbox' / 'LICENSE.md'))
for avatar, prefix in [('Male_Adult_01', 'm002'), ('Female_Adult_03', 'f003')]:
    folder = f'Assets/Avatars/Adults/{avatar}/'
    records.append(fetch(base + folder + f'Export/{avatar}.fbx', ROOT / 'rocketbox' / avatar / 'Export' / f'{avatar}.fbx'))
    for suffix in ['body_color', 'head_color', 'opacity_color']:
        name = f'{prefix}_{suffix}.tga'
        records.append(fetch(base + folder + 'Textures/' + name, ROOT / 'rocketbox' / avatar / 'Textures' / name))
(ROOT / 'download-hashes.json').write_text(json.dumps(records, indent=2) + '\n')
print(f'Fetched/hashed {len(records)} files under {ROOT}')
