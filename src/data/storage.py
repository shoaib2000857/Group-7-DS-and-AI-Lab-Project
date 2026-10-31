"""Read immutable dataset folders or ZIPs without extracting unsafe paths."""
from __future__ import annotations
import hashlib
import io
import bisect
import json
from pathlib import Path, PurePosixPath
import zipfile

import numpy as np
from PIL import Image


class SplitZipReader(io.RawIOBase):
    """Seek across verified-size archive parts; no reassembly or extraction."""
    def __init__(self, manifest: str|Path):
        self.manifest=Path(manifest);data=json.loads(self.manifest.read_text(encoding='utf-8'))
        if data.get('format')!='terrawatch-split-zip-v1':raise ValueError('Unknown split archive format')
        self.parts=[];self.starts=[0];self.pos=0;self.handles={}
        for part in data['parts']:
            p=PurePosixPath(part['name'])
            if len(p.parts)!=1 or p.name!=part['name'] or ':' in p.name or '\\' in p.name:raise ValueError('Unsafe split-part path')
            path=self.manifest.parent/p.name
            if path.stat().st_size!=part['size']:raise ValueError('Missing/incomplete archive part: '+p.name)
            self.parts.append(path);self.starts.append(self.starts[-1]+part['size'])
        self.size=self.starts[-1]
        if self.size!=data['original_size']:raise ValueError('Split archive length mismatch')

    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=io.SEEK_SET):
        new=offset if whence==io.SEEK_SET else self.pos+offset if whence==io.SEEK_CUR else self.size+offset
        if new<0:raise ValueError('Negative archive seek')
        self.pos=new;return new

    def read(self,n=-1):
        remaining=max(0,self.size-self.pos) if n<0 else max(0,min(n,self.size-self.pos));chunks=[]
        while remaining:
            i=bisect.bisect_right(self.starts,self.pos)-1
            if i not in self.handles:self.handles[i]=self.parts[i].open('rb')
            f=self.handles[i];f.seek(self.pos-self.starts[i]);wanted=min(remaining,self.starts[i+1]-self.pos)
            b=f.read(wanted)
            if len(b)!=wanted:raise ValueError('Archive part changed/truncated during read')
            chunks.append(b);self.pos+=len(b);remaining-=len(b)
        return b''.join(chunks)

    def readinto(self,b):
        data=self.read(len(b));b[:len(data)]=data;return len(data)

    def close(self):
        for f in self.handles.values():f.close()
        self.handles.clear();super().close()


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def pixel_hash(array: np.ndarray) -> str:
    return sha256(str(array.shape).encode() + array.dtype.str.encode() + array.tobytes())


class Store:
    """A portable relative-member namespace for a directory or ZIP archive."""
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.split_stream=SplitZipReader(self.path) if self.path.name.endswith('.zip.parts.json') else None
        self.archive = zipfile.ZipFile(self.split_stream) if self.split_stream else zipfile.ZipFile(self.path) if self.path.is_file() and zipfile.is_zipfile(self.path) else None
        if not self.path.exists():
            raise FileNotFoundError(self.path)

    def names(self) -> list[str]:
        names = self.archive.namelist() if self.archive else [p.relative_to(self.path).as_posix() for p in self.path.rglob('*') if p.is_file()]
        return sorted(n for n in names if not n.endswith('/') and '__MACOSX' not in n and not PurePosixPath(n).name.startswith('.'))

    def read(self, member: str) -> bytes:
        p = PurePosixPath(member.replace('\\', '/'))
        if p.is_absolute() or '..' in p.parts or ':' in p.parts[0]:
            raise ValueError(f'Unsafe dataset member: {member}')
        if self.archive:
            if self.archive.getinfo(p.as_posix()).file_size > 512 * 1024**2:
                raise ValueError('Image/annotation member exceeds safety limit')
            return self.archive.read(p.as_posix())
        return self.path.joinpath(*p.parts).read_bytes()

    def image(self, member: str) -> tuple[np.ndarray, str]:
        data = self.read(member)
        with Image.open(io.BytesIO(data)) as im:
            im.load()
            if im.mode == 'P':
                im = im.convert('RGB')
            array = np.array(im)
        return array, sha256(data)

    def close(self) -> None:
        if self.archive:
            self.archive.close()
        if self.split_stream:self.split_stream.close()

    def archive_stream(self):
        return SplitZipReader(self.path) if self.split_stream else self.path.open('rb')

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
