import sys
import hashlib
from collections import defaultdict
from tqdm import tqdm
import gzip
from multiprocessing.pool import Pool
from multiprocessing.managers import BaseManager, DictProxy
import os


class MyManager(BaseManager):
    pass


MyManager.register('defaultdict', defaultdict, DictProxy)

_hashes = defaultdict(int)


def _file_loader(fname):
    with gzip.open(fname, "rt") as f:
        for line in f:
            yield line


def _corpus_loader(line_generator):
    wstr = "WARC/1.0"
    header_mode = False
    for line in line_generator:
        line = line.strip()
        if not header_mode and not line:
            yield line, None
            continue
        elif not header_mode and line == wstr:
            header_mode = True
        elif header_mode:
            if not line:
                header_mode = False
                yield line, None
                continue
        yield line, header_mode


def _corpus_loader_dedup(line_generator, hashes):
    wstr = "WARC/1.0"
    ustr = "WARC-Target-URI"
    header_mode = None
    out = []
    url = None
    domain = None
    for line in line_generator:
        line = line.strip()
        if line == wstr:
            header_mode = True
            if out:
                if domain is not None and url is not None:
                    yield {"url": url, "domain": domain, "data": out}
                url = None
                domain = None
                out = []
            continue
        if header_mode:
            if line.startswith(ustr):
                url = line.split(ustr+":")[1].strip()
                domain = url.split("//")[1].split("/")[0]
            if line:
                continue
            else:
                header_mode = False
        else:
            h = hashes[hashlib.sha1(bytes(line.lower(), encoding="utf-8")).digest()]
            if h > 1:
                out.append(line)


def _split_by_lang(batch):
    out = {}
    for line in batch["data"]:
        lang = lid_model.predict(line)[0][0].split("__")[-1]
        if lang not in out:
            out[lang] = []
        out[lang].append(line)
    out = max(out.items(), key=lambda x: len(x[1]))
    return {"lang": out[0],
            "length": len(''.join(out[1])),
            "url": batch["url"],
            "domain": batch["domain"]}


def _output(results):
    out = {}
    sep = "_____"
    for result in results:
        d = result["domain"] + sep + result["lang"] 
        if d not in out:
            out[d] = 0
        out[d] += result["length"]
    for key, value in out.items():
        key = key.split(sep)
        if len(key) == 2:
            print('\t'.join(list(map(str, [key[0], key[1], value]))))


def _create_hash(fname, shared_dict):
    for line, mode in tqdm(_corpus_loader(_file_loader(fname))):
        if mode is not None and not mode:
            shared_dict[hashlib.sha1(bytes(line.lower(), encoding="utf-8")).digest()] += 1


def create_hashes(files):
    pool = Pool(os.cpu_count())
    mgr = MyManager()
    mgr.start()
    hashes = mgr.defaultdict(int)
    for fname in files:
        pool.apply_async(_create_hash, (fname, _hashes))
    pool.close()
    pool.join()
    return hashes


def main():
    files = [x.strip() for x in sys.stdin]
    hashes = create_hashes(files)
    for fname in files:
        line_gen = (x for x in _file_loader(fname))
        pool = Pool(os.cpu_count())
        results = pool.map(_split_by_lang, _corpus_loader_dedup(line_gen, hashes))
        _output(results)


if __name__ == "__main__":
    main()
