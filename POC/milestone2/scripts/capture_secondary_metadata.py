"""Capture the author-linked SECOND-CC ZIP directory and caption records.

Only archive metadata and caption JSON are read; no full 2.54-GB archive is
required. Source record and byte size are fixed to Zenodo version 16937571.
"""
import io
import json
from pathlib import Path
import zipfile
import requests

ROOT = Path(__file__).resolve().parents[1]
URL = "https://zenodo.org/records/16937571/files/SECOND-CC-AUG.zip?download=1"


class RangeReader(io.RawIOBase):
    def __init__(self):
        self.pos, self.size = 0, 2539187782
        self.cache_start, self.cache = 0, b""

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        return self.pos

    def read(self, n=-1):
        n = self.size - self.pos if n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        if not (self.cache_start <= self.pos and self.pos+n <= self.cache_start+len(self.cache)):
            start = self.pos
            end = min(self.size, max(start+n, start+(1 << 20))) - 1
            response = requests.get(URL, headers={"Range": f"bytes={start}-{end}"}, timeout=(15, 40))
            response.raise_for_status()
            if response.status_code != 206 or len(response.content) != end-start+1:
                raise ValueError("Server did not return the requested range")
            self.cache_start, self.cache = start, response.content
        offset = self.pos-self.cache_start
        data = self.cache[offset:offset+n]
        self.pos += len(data)
        return data


def main():
    index = ROOT / "evidence/second_index.json"
    captions = ROOT / "data/raw/second_captions.json"
    if index.exists() and captions.exists():
        print("Captured metadata already present")
        return
    with zipfile.ZipFile(RangeReader()) as z:
        rows = [{"name": r.filename, "offset": r.header_offset, "compressed": r.compress_size, "size": r.file_size, "crc32": r.CRC} for r in z.infolist()]
        index.parent.mkdir(parents=True, exist_ok=True)
        index.write_text(json.dumps(rows), encoding="utf-8")
        captions.parent.mkdir(parents=True, exist_ok=True)
        captions.write_bytes(z.read("SECOND-CC-AUG/SECOND-CC-AUG.json"))
        print("Captured", len(rows), "archive entries and source captions")


if __name__ == "__main__":
    main()
