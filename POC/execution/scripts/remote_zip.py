"""Read members of a large remote zip over HTTP range requests.

Lets us index LEVIR-MCI (2.8 GB) and extract only the pairs in the frozen
manifest, so unselected images are never downloaded or inspected.
Standard library only.
"""
import io
import struct
import time
import urllib.error
import urllib.request
import zipfile
import zlib

LEVIR_MCI_URL = (
    "https://huggingface.co/datasets/lcybuaa/LEVIR-MCI/resolve/"
    "dff09c709999b5b1f1f1806024811c65a6b14dfe/LEVIR-MCI-dataset.zip"
)


class HttpRangeFile(io.RawIOBase):
    """Seekable read-only file over HTTP with a small read-ahead cache."""

    def __init__(self, url, block=1 << 20):
        self.url = url
        self.block = block
        self.pos = 0
        self.cache_start, self.cache = 0, b""
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req) as r:
            self.size = int(r.headers["Content-Length"])
            self.final_url = r.url
        self.bytes_fetched = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        else:
            self.pos = self.size + offset
        return self.pos

    def _fetch(self, start, end, attempts=6):
        """Range GET; retries 429/5xx and dropped connections with backoff (HF CDN sheds bursts)."""
        req = urllib.request.Request(
            self.final_url, headers={"Range": f"bytes={start}-{end - 1}"}
        )
        for k in range(attempts):
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                break
            except urllib.error.HTTPError as e:
                if e.code != 429 and e.code < 500 or k == attempts - 1:
                    raise
                if e.headers.get("Retry-After", "").isdigit():  # Zenodo says how long to back off
                    time.sleep(int(e.headers["Retry-After"]))
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if k == attempts - 1:
                    raise
            time.sleep(2 ** k)
        self.bytes_fetched += len(data)
        return data

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        end = self.pos + n
        c0, c1 = self.cache_start, self.cache_start + len(self.cache)
        if not (c0 <= self.pos and end <= c1):
            fetch_end = min(self.size, max(end, self.pos + self.block))
            self.cache_start, self.cache = self.pos, self._fetch(self.pos, fetch_end)
        off = self.pos - self.cache_start
        out = self.cache[off:off + n]
        self.pos += len(out)
        return out

    def readinto(self, b):
        data = self.read(len(b))
        b[: len(data)] = data
        return len(data)


def open_remote_zip(url=LEVIR_MCI_URL):
    raw = HttpRangeFile(url)
    return zipfile.ZipFile(raw), raw


def fetch_members(zf, raw, names, workers=16):
    """Fetch many members in parallel by exact byte range; CRC-checked.

    One request per member (local header + data) instead of serial seeks.
    """
    from concurrent.futures import ThreadPoolExecutor

    def one(name):
        info = zf.getinfo(name)
        start = info.header_offset
        span = 30 + len(info.orig_filename.encode()) + 1024 + info.compress_size
        buf = raw._fetch(start, min(raw.size, start + span))
        n, e = struct.unpack("<HH", buf[26:30])
        data = buf[30 + n + e: 30 + n + e + info.compress_size]
        if info.compress_type == zipfile.ZIP_DEFLATED:
            data = zlib.decompress(data, -15)
        elif info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"unsupported compression {info.compress_type}")
        if zlib.crc32(data) != info.CRC:
            raise ValueError(f"CRC mismatch for {name}")
        return name, data

    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(one, names))
