import argparse
import hashlib
import json
import re
import shutil
import struct
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download, snapshot_download

from download_index import DOMESTIC_PROXY, REVISION as MODELSCOPE_REVISION, download_index


MODEL_REVISION = "f52bf8ec8c7124536f0efb74aca902b2995e5bcd"
INDEX_REVISION = "a4d31160a035f30764604f4827cd8f1d0315eb86"
PARTS = {
    "part_aa": "a8a6a246951da4bbc8771a223283ef61963882a32864d9044ec00abb90fc3023",
    "part_ab": "b6d9bc943626fe7cb44de4c849e9379e7f272ab216c0552acbcf2390cc033c11",
}


def prepare_corpus(corpus_path, output, count):
    offsets_path = output / "corpus_offsets.npy"
    manifest_path = output / "corpus_manifest.json"
    stat = corpus_path.stat()
    if manifest_path.exists() and offsets_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if (manifest["size"], manifest["mtime_ns"], manifest["count"]) == (
            stat.st_size, stat.st_mtime_ns, count
        ):
            return manifest
    temporary = offsets_path.with_suffix(".tmp.npy")
    offsets = np.lib.format.open_memmap(temporary, mode="w+", dtype="<u8", shape=(count + 1,))
    digest = hashlib.sha256()
    position = 0
    seen = 0
    identifier = re.compile(rb'^\{"id": "([0-9]+)"')
    with corpus_path.open("rb") as stream:
        for row, line in enumerate(stream):
            match = identifier.match(line)
            if row >= count or match is None or int(match.group(1)) != row:
                raise ValueError(f"Unexpected corpus ID/order at row {row}")
            offsets[row] = position
            digest.update(line)
            position += len(line)
            seen = row + 1
    if seen != count:
        raise ValueError(f"Corpus has {seen} rows; index has {count}")
    offsets[count] = position
    offsets.flush()
    del offsets
    temporary.replace(offsets_path)
    manifest = {
        "path": str(corpus_path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns,
        "count": count, "sha256": digest.hexdigest(), "ids": "sequential strings, checked on every row",
        "source": "PeterJinGo/wiki-18-corpus", "source_revision": "69c1c00ffe7c5554c68d8548355cb22e46aabc51",
        "alignment": "row count and all IDs checked; vector/content sampling checked separately",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--corpus-only", action="store_true")
    parser.add_argument("--index-proxy", default=DOMESTIC_PROXY)
    parser.add_argument("--index-download-dir", type=Path)
    args = parser.parse_args()
    output = args.root / "data/search_r1"
    output.mkdir(parents=True, exist_ok=True)
    corpus_path = args.root / "data/wiki_18_corpus/wiki_corpus.jsonl"
    if not corpus_path.exists():
        import gzip
        compressed = hf_hub_download("PeterJinGo/wiki-18-corpus", "wiki-18.jsonl.gz", repo_type="dataset",
                                     revision="69c1c00ffe7c5554c68d8548355cb22e46aabc51", local_dir=output / "corpus")
        corpus_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = corpus_path.with_suffix(".tmp")
        with gzip.open(compressed, "rb") as source, temporary.open("wb") as destination:
            shutil.copyfileobj(source, destination, 16 * 1024 * 1024)
        temporary.replace(corpus_path)
    if args.corpus_only:
        print(prepare_corpus(corpus_path, output, 21015324), flush=True)
        return
    snapshot_download("intfloat/e5-base-v2", revision=MODEL_REVISION, local_dir=output / "e5-base-v2",
                      allow_patterns=["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                                      "special_tokens_map.json", "vocab.txt"])
    index_path = output / "index/e5_Flat.index"
    if not index_path.exists():
        download_directory = args.index_download_dir or index_path.parent
        download_index(download_directory, proxy=args.index_proxy)
        index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = index_path.with_suffix(".tmp")
        with temporary.open("wb") as destination:
            for name, expected in PARTS.items():
                part = download_directory / name
                digest = hashlib.sha256()
                with part.open("rb") as source:
                    for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
                        digest.update(block)
                        destination.write(block)
                if digest.hexdigest() != expected:
                    raise ValueError(f"Checksum mismatch: {part}")
                print(f"Verified and merged {name}", flush=True)
        temporary.replace(index_path)
    with index_path.open("rb") as source:
        header = source.read(16)
    if header[:4] != b"IxFI" or struct.unpack_from("<i", header, 4)[0] != 768:
        raise ValueError("Expected the official 768-dimensional E5 Flat inner-product index")
    count = struct.unpack_from("<q", header, 8)[0]
    print(prepare_corpus(corpus_path, output, count), flush=True)
    metadata = {"model": "intfloat/e5-base-v2", "model_revision": MODEL_REVISION,
                "index": "PeterJinGo/wiki-18-e5-index", "index_revision": INDEX_REVISION,
                "index_download_source": "https://www.modelscope.cn/datasets/yamseyoung/wiki-18-e5-index",
                "index_download_revision": MODELSCOPE_REVISION,
                "parts_sha256": PARTS, "count": count, "dimension": 768}
    (output / "assets_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
