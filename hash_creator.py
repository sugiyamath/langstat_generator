import os
import hashlib
import wet_loader
from collections import defaultdict
from multiprocessing.pool import Pool
from tqdm import tqdm

DEFAULT_CPUS = os.cpu_count()


def _create_hash(fname):
    hashes = defaultdict(int)
    for line, mode in tqdm(
            wet_loader.corpus_loader(wet_loader.file_loader(fname))):
        if mode is not None and not mode:
            hashes[hashlib.sha1(bytes(line.lower(),
                                      encoding="utf-8")).digest()] += 1
    return hashes


def create_hashes(files, num_cpus=DEFAULT_CPUS):
    pool = Pool(num_cpus)
    hashes_list = pool.map(_create_hash, files)
    hashes = defaultdict(int)
    for h in hashes_list:
        hashes.update(h)
    pool.close()
    return hashes
