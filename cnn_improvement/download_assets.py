"""Download only the official MASSIVE English data and GloVe 6B 300d vectors.

Uses standard-library HTTPS, exact archive-member extraction and checksums.
The GloVe ZIP directory is read via byte ranges; only the requested compressed
member is transferred. Existing complete files and partial downloads are reused.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import struct
import tarfile
import time
import urllib.request
import zipfile
import zlib


MASSIVE_URL = "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.0.tar.gz"
MASSIVE_MEMBER = "1.0/data/en-US.jsonl"
GLOVE_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
GLOVE_MIRROR_URL = "https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip"
GLOVE_MEMBER = "glove.6B.300d.txt"
CHUNK = 1024 * 1024


def request(url: str, *, start: int | None = None, end: int | None = None,
            method: str = "GET"):
    headers = {"User-Agent": "SC4002-reproducible-training/1.0"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{'' if end is None else end}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=headers, method=method), timeout=120)


def remote_size(url: str) -> int:
    with request(url, method="HEAD") as response:
        return int(response.headers["Content-Length"])


class RangeReader(io.RawIOBase):
    """Seekable read-only HTTPS object for ZIP metadata, never full extraction."""

    def __init__(self, url: str, size: int):
        self.url, self.size, self.pos = url, size, 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if self.pos < 0:
            raise ValueError("Negative remote seek")
        return self.pos

    def read(self, size=-1):
        size = self.size - self.pos if size < 0 else min(size, self.size - self.pos)
        if size <= 0:
            return b""
        start = self.pos
        with request(self.url, start=start, end=start + size - 1) as response:
            if response.status != 206:
                raise RuntimeError("Source did not honor HTTP ranges")
            data = response.read()
        if len(data) != size:
            raise IOError(f"Short HTTP range: {len(data)} != {size}")
        self.pos += len(data)
        return data


def download(url: str, dest: Path, *, size: int, offset: int = 0,
             force_range: bool = False) -> None:
    """Resume a bounded public download using a .part file and atomic rename."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == size:
        print(f"Reusing {dest.name}: {size / 1e6:.1f} MB", flush=True)
        return
    partial = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(5):
        have = partial.stat().st_size if partial.exists() else 0
        if have > size:
            raise RuntimeError(f"Partial download is too large: {partial}")
        if have == size:
            partial.replace(dest)
            return
        ranged = force_range or have > 0 or offset > 0
        started, previous = time.monotonic(), time.monotonic()
        try:
            with request(url, start=offset + have if ranged else None,
                         end=offset + size - 1 if ranged else None) as response:
                if ranged and response.status != 206:
                    raise RuntimeError("Source did not honor a resumed/bounded range")
                if ranged:
                    content_range = response.headers.get("Content-Range", "")
                    if not content_range.startswith(f"bytes {offset + have}-"):
                        raise RuntimeError(f"Unexpected HTTP Content-Range: {content_range}")
                with partial.open("ab") as stream:
                    transferred = 0
                    while True:
                        block = response.read(min(CHUNK, size - have - transferred))
                        if not block:
                            break
                        stream.write(block)
                        transferred += len(block)
                        now = time.monotonic()
                        if now - previous > 10:
                            print(f"{dest.name}: {(have + transferred) / 1e6:.1f}/{size / 1e6:.1f} MB "
                                  f"({transferred / max(now - started, .001) / 1e6:.1f} MB/s)", flush=True)
                            previous = now
                        if have + transferred == size:
                            break
            if partial.stat().st_size != size:
                raise IOError(f"Incomplete download: {partial.stat().st_size}/{size}")
            partial.replace(dest)
            print(f"Downloaded {dest.name}: {size / 1e6:.1f} MB", flush=True)
            return
        except (OSError, RuntimeError) as error:
            if attempt == 4:
                raise
            print(f"Download attempt {attempt + 1} interrupted: {error}; resuming.", flush=True)
            time.sleep(2)


def file_record(path: Path, source: str, member: str) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * CHUNK):
            digest.update(block)
    return {"path": str(path), "source_url": source, "archive_member": member,
            "bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def get_massive(root: Path) -> dict:
    output = root / "data" / "en-US.jsonl"
    if not output.exists():
        archive = root / "data" / "amazon-massive-dataset-1.0.tar.gz"
        download(MASSIVE_URL, archive, size=remote_size(MASSIVE_URL))
        partial = output.with_suffix(".jsonl.part")
        with tarfile.open(archive, "r:gz") as container:
            info = container.getmember(MASSIVE_MEMBER)
            if not info.isfile():
                raise RuntimeError("The requested MASSIVE member is not a regular file")
            with container.extractfile(info) as source, partial.open("wb") as dest:
                while block := source.read(CHUNK):
                    dest.write(block)
        if partial.stat().st_size != info.size:
            raise RuntimeError("MASSIVE extraction size mismatch")
        partial.replace(output)
    with output.open(encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    counts = dict(Counter(row["partition"] for row in records))
    if counts != {"train": 11514, "dev": 2033, "test": 2974}:
        raise RuntimeError(f"Unexpected MASSIVE split counts: {counts}")
    if any(row["locale"] != "en-US" for row in records):
        raise RuntimeError("Non-English data found in requested locale")
    print(f"Verified English MASSIVE data: {counts}", flush=True)
    return {**file_record(output, MASSIVE_URL, MASSIVE_MEMBER), "split_counts": counts}


def get_glove(root: Path, url: str = GLOVE_MIRROR_URL) -> dict:
    output = root / "embeddings" / GLOVE_MEMBER
    size = remote_size(url)
    remote = RangeReader(url, size)
    with zipfile.ZipFile(remote) as container:
        info = container.getinfo(GLOVE_MEMBER)
    if info.compress_type != zipfile.ZIP_DEFLATED:
        raise RuntimeError(f"Unexpected ZIP compression method: {info.compress_type}")
    if info.flag_bits & 1:
        raise RuntimeError("Encrypted word-vector archive is unsupported")
    if not output.exists() or output.stat().st_size != info.file_size:
        remote.seek(info.header_offset)
        header = remote.read(30)
        signature, *_ = struct.unpack("<IHHHHHIIIHH", header)
        if signature != 0x04034B50:
            raise RuntimeError("Invalid ZIP local entry header")
        name_length, extra_length = struct.unpack("<HH", header[26:30])
        start = info.header_offset + 30 + name_length + extra_length
        source_suffix = ".stanford-hf.deflate" if url == GLOVE_MIRROR_URL else ".deflate"
        compressed = root / "embeddings" / (GLOVE_MEMBER + source_suffix)
        print(f"GloVe selected ZIP member: {info.compress_size / 1e6:.1f} MB compressed, "
              f"{info.file_size / 1e6:.1f} MB extracted", flush=True)
        download(url, compressed, size=info.compress_size, offset=start, force_range=True)
        partial = output.with_suffix(output.suffix + ".part")
        decoder, crc, total = zlib.decompressobj(-15), 0, 0
        with compressed.open("rb") as source, partial.open("wb") as dest:
            while block := source.read(CHUNK):
                unpacked = decoder.decompress(block)
                crc = zlib.crc32(unpacked, crc)
                total += len(unpacked)
                dest.write(unpacked)
            tail = decoder.flush()
            dest.write(tail)
            crc = zlib.crc32(tail, crc)
            total += len(tail)
        if not decoder.eof or crc != info.CRC or total != info.file_size:
            raise RuntimeError("GloVe decompression CRC/size validation failed")
        partial.replace(output)
    crc = 0
    with output.open("rb") as stream:
        while block := stream.read(8 * CHUNK):
            crc = zlib.crc32(block, crc)
    if crc != info.CRC:
        raise RuntimeError("Existing GloVe file does not match source ZIP CRC")
    print(f"Verified GloVe file: {output.name}, ZIP CRC {crc:08x}", flush=True)
    return {**file_record(output, url, GLOVE_MEMBER),
            "project_url": "https://nlp.stanford.edu/projects/glove/",
            "zip_crc32": f"{crc:08x}", "dimensions": 300,
            "archive_bytes": size, "selected_member_compressed_bytes": info.compress_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--only", choices=("massive", "glove", "both"), default="both")
    parser.add_argument("--glove-source", choices=("mirror", "stanford"), default="mirror",
                        help="Stanford NLP's Hugging Face mirror is the default; exact provenance is recorded.")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    manifest_path = args.root / "asset_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    glove_url = GLOVE_MIRROR_URL if args.glove_source == "mirror" else GLOVE_URL
    for name, action in (("massive", get_massive), ("glove", lambda root: get_glove(root, glove_url))):
        if args.only in ("both", name):
            manifest[name] = action(args.root)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Asset manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
