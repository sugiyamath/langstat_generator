
import os
import sys
import hashlib
import wet_loader
from collections import defaultdict
from multiprocessing.pool import Pool
from tqdm import tqdm

DEFAULT_CPUS = os.cpu_count()


def _create_hash(fname, logby=500000):
    hashes = defaultdict(int)
    arrived = set()
    out = []
    for i, (line, mode) in tqdm(enumerate(
            wet_loader.corpus_loader(wet_loader.file_loader(fname))),
                                miniters=logby, bar_format="{r_bar}",
                                file=sys.stdout):
        sys.stdout.flush()
        if mode is not None and not mode:
            try:
                h = hashlib.sha1(
                    bytes(line.lower(), encoding="utf-8")).digest()
            except Exception:
                print("hash_creator: Error {} {} {}".format(fname, i, line))
            if h in arrived:
                continue
            else:
                hashes[h] += 1
                if hashes[h] > 1:
                    arrived.add(h)
                    out.extend([h])
                    del hashes[h]
    return out


def create_hashes(files, num_cpus=DEFAULT_CPUS):
    pool = Pool(num_cpus)
    hashes_list = pool.map(_create_hash, files)
    hashes = []
    for h in tqdm(hashes_list):
        hashes.extend(h)
    pool.close()
    return set(hashes)


