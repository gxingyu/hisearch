# Modified for HiSearch; see root docs/PROVENANCE.md for local changes.
import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests


REVISION = "66f7f27585673bd50d9dbf7d296a8c5959231fee"
FILES = {"part_aa": 42949672960, "part_ab": 21609402413}
DOMESTIC_PROXY = os.environ.get("HISEARCH_INDEX_PROXY")
BLOCK_SIZE = 128 * 1024 * 1024


def download_index(directory, proxy=DOMESTIC_PROXY, workers=8):
    directory.mkdir(parents=True, exist_ok=True)
    proxies = {"http": proxy, "https": proxy} if proxy else None
    jobs = []
    descriptors = {}
    started = time.monotonic()
    downloaded = 0
    try:
        for name, size in FILES.items():
            destination = directory / name
            if destination.exists() and destination.stat().st_size == size:
                continue
            temporary = directory / f"{name}.download"
            checkpoints = directory / f".{name}.chunks"
            checkpoints.mkdir(exist_ok=True)
            descriptor = os.open(temporary, os.O_CREAT | os.O_RDWR, 0o644)
            if os.fstat(descriptor).st_size != size:
                os.ftruncate(descriptor, size)
                for checkpoint in checkpoints.iterdir():
                    checkpoint.unlink()
            descriptors[name] = descriptor
            for start in range(0, size, BLOCK_SIZE):
                end = min(start + BLOCK_SIZE, size) - 1
                marker = checkpoints / str(start)
                if not marker.exists():
                    jobs.append((name, start, end, marker))

        def fetch(job):
            name, start, end, marker = job
            url = f"https://www.modelscope.cn/api/v1/datasets/yamseyoung/wiki-18-e5-index/repo?Revision={REVISION}&FilePath={name}"
            for attempt in range(5):
                try:
                    with requests.get(url, headers={"Range": f"bytes={start}-{end}"}, proxies=proxies,
                                      stream=True, timeout=(30, 120)) as response:
                        response.raise_for_status()
                        expected = f"bytes {start}-{end}/{FILES[name]}"
                        if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                            raise ValueError(f"Invalid range response: {response.headers.get('Content-Range')}")
                        position = start
                        for block in response.iter_content(1024 * 1024):
                            if position + len(block) > end + 1:
                                raise ValueError("Oversized range response")
                            view = memoryview(block)
                            while view:
                                written = os.pwrite(descriptors[name], view, position)
                                position += written
                                view = view[written:]
                        if position != end + 1:
                            raise ValueError("Incomplete range response")
                    marker.touch()
                    return end - start + 1
                except (requests.RequestException, ValueError):
                    if attempt == 4:
                        raise
                    time.sleep(min(2 ** attempt, 8))

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(fetch, job) for job in jobs]
            errors = []
            for future in as_completed(futures):
                try:
                    downloaded += future.result()
                except Exception as error:
                    errors.append(error)
                    print(f"Chunk failed; rerun to resume: {error}", flush=True)
                    continue
                elapsed = time.monotonic() - started
                print(f"ModelScope: {downloaded / 1e9:.2f} GB downloaded this run, {downloaded / elapsed / 1e6:.1f} MB/s", flush=True)
            if errors:
                raise RuntimeError(f"{len(errors)} chunks failed; rerun to resume") from errors[0]
        for name, descriptor in descriptors.items():
            os.fsync(descriptor)
            os.close(descriptor)
            descriptors[name] = None
            (directory / f"{name}.download").replace(directory / name)
    finally:
        for descriptor in descriptors.values():
            if descriptor is not None:
                os.close(descriptor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parents[1] / "data/search_r1/index")
    parser.add_argument("--proxy", default=DOMESTIC_PROXY)
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    download_index(arguments.directory, arguments.proxy, arguments.workers)
