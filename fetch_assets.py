"""Fetch a known official asset in resumable, validated HTTP ranges."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import time
import urllib.request


def fetch(url, output, size, sha256=None):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    size = int(size)
    if output.exists() and output.stat().st_size == size:
        if sha256 is None or hashlib.sha256(output.read_bytes()).hexdigest() == sha256:
            return output
    chunk_size = 256 * 1024
    partials = []

    def part(index):
        start, end = index * chunk_size, min(size, (index + 1) * chunk_size) - 1
        target = output.with_name(output.name + f'.part{index:03}')
        if target.exists() and target.stat().st_size == end - start + 1:
            return target
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={'Range': f'bytes={start}-{end}',
                                                           'User-Agent': 'G1-Simulation-Asset-Fetcher'})
                with urllib.request.urlopen(req, timeout=25) as response:
                    expected = f'bytes {start}-{end}/{size}'
                    if response.status != 206 or response.headers.get('Content-Range') != expected:
                        raise RuntimeError(f'Server did not return the requested range: {response.status}')
                    data = response.read()
                if len(data) != end - start + 1:
                    raise RuntimeError('Incomplete range')
                target.write_bytes(data)
                print(f'Asset {output.name}: chunk {index + 1}/{(size + chunk_size - 1)//chunk_size}', flush=True)
                return target
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(.3)

    count = (size + chunk_size - 1) // chunk_size
    with ThreadPoolExecutor(max_workers=12) as workers:
        partials = list(workers.map(part, range(count)))
    temporary = output.with_suffix(output.suffix + '.complete')
    with temporary.open('wb') as stream:
        for path in partials:
            stream.write(path.read_bytes())
    if temporary.stat().st_size != size:
        raise RuntimeError('Incorrect assembled size')
    if sha256 and hashlib.sha256(temporary.read_bytes()).hexdigest() != sha256:
        raise RuntimeError('Official asset SHA256 mismatch')
    temporary.replace(output)
    print(f'Saved {output.name}: {size} bytes', flush=True)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('url')
    parser.add_argument('output')
    parser.add_argument('size', type=int)
    parser.add_argument('--sha256')
    args = parser.parse_args()
    fetch(args.url, args.output, args.size, args.sha256)
