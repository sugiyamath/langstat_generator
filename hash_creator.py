import os
import hashlib
import wet_loader
from collections import defaultdict
from multiprocessing.pool import Pool
from tqdm import tqdm

DEFAULT_CPUS = os.cpu_count()


def _create_hash(fname):
    hashes = defaultdict(int)
    arrived = set()
    for line, mode in tqdm(
            wet_loader.corpus_loader(wet_loader.file_loader(fname))):
        if mode is not None and not mode:
            h = hashlib.sha1(bytes(line.lower(), encoding="utf-8")).digest()
            if h in arrived:
                continue
            else:
                hashes[h] += 1
                if hashes[h] > 1:
                    arrived.add(h)
                    del hashes[h]
    return hashes.keys()


def create_hashes(files, num_cpus=DEFAULT_CPUS):
    pool = Pool(num_cpus)
    hashes_list = pool.map(_create_hash, files)
    hashes = []
    for h in tqdm(hashes_list):
        hashes.extend(list(h))
    pool.close()
    return set(hashes)


