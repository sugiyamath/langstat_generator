import gc
import sys
import hashlib
from collections import defaultdict
from tqdm import tqdm
import gzip
from multiprocessing.pool import Pool
from multiprocessing import Process
import fasttext
import os
import text_normalizer
import kenlm
import sentencepiece as spm
import json
import random
import string
from functools import partial

bin_dir = sys.argv[1]
lid_model = fasttext.load_model(os.path.join(bin_dir, "lid.bin"))

ls1 = {x.split(".")[0] for x in os.listdir(os.path.join(bin_dir, "lm_sp"))
       if x.endswith(".arpa.bin")}
ls2 = {x.split(".")[0] for x in os.listdir(os.path.join(bin_dir, "lm_sp"))
       if x.endswith(".sp.model")}
langs = ls1 & ls2

shared_data = {lang: None for lang in langs}

lm = None
sp = None


def _add_lang_score(line):
    global shared_data
    result = json.loads(line.strip())
    doc_score = 0
    doc_length = 0
    for line in result["data"]:
        line = text_normalizer.normalize(line)
        pieces = ' '.join(sp.EncodeAsPieces(line))
        if len(pieces):
            doc_score += lm.score(' '.join(pieces))
            doc_length += len(pieces)
    if doc_length:
        result["perplexity"] = 10.0**(-doc_score/doc_length)
    else:
        result["perplexity"] = 0.0
    del(result["data"])
    return result


def _file_loader(fname):
    with gzip.open(fname, "rt") as f:
        for line in f:
            yield line


def _file_loader_bulk(fnames):
    for fname in fnames:
        for line in _file_loader(fname):
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
            if h < 2:
                out.append(line)


def _detect_lang(batch):
    out = {}
    lscores = []
    for line in batch["data"]:
        tmp_pred = lid_model.predict(line)
        lang = tmp_pred[0][0].split("__")[-1]
        lscores.append(float(tmp_pred[1][0]))
        if lang not in out:
            out[lang] = []
        out[lang].append(line)
    lscore = sum(x/len(lscores) for x in lscores)
    out = max(out.items(), key=lambda x: len(x[1]))
    return {"lang": out[0],
            "language_score": float(lscore),
            "length": len(''.join(out[1])),
            "url": batch["url"],
            "domain": batch["domain"],
            "data": batch["data"]}


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))


def _save_to_tmp(result, tmp_dir, fprefix, langs):
    if result["lang"] in langs:
        with open(os.path.join(
                tmp_dir, fprefix+"_{}".format(result["lang"])), "a") as f:
            f.write(json.dumps(result)+"\n")


def _initializer(lang, lm_s, sp_s):
    global shared_data
    shared_data[lang] = (lm_s, sp_s)


def _add_lang_score_bulk(fprefix, tmp_dir="./tmp"):
    global shared_data
    target_langs = {x.split("_")[-1] for x in os.listdir(tmp_dir)
                    if x.startswith(fprefix)}
    for lang in tqdm(target_langs):
        lm_s, sp_s = _load_lm(bin_dir, lang)
        pool = Pool(os.cpu_count(), _initializer, (lang, lm_s, sp_s))
        with open(os.path.join(
                tmp_dir, fprefix+"_{}".format(lang))) as f:
            out = pool.imap_unordered(_add_lang_score, f)
            yield list(out)
            del(lm_s)
            del(sp_s)
            del(shared_data[lang])
        pool.close()
        gc.collect()


def _add_lang_score_bulk(fprefix, tmp_dir="./tmp"):
    global shared_data
    target_langs = {x.split("_")[-1] for x in os.listdir(tmp_dir)
                    if x.startswith(fprefix)}
    for lang in tqdm(target_langs):
        lm_s, sp_s = _load_lm(bin_dir, lang)
        pool = Pool(os.cpu_count(), _initializer, (lang, lm_s, sp_s))
        with open(os.path.join(
                tmp_dir, fprefix+"_{}".format(lang))) as f:
            out = pool.imap_unordered(_add_lang_score, f)
            yield list(out)
            del(lm_s)
            del(sp_s)
            del(shared_data[lang])
        pool.close()
        gc.collect()


def _output(results, score_outpath, langstat_outpath):
    out = {}
    sep = "_____"
    with open(score_outpath, "a") as f:
        for result in results:
            d = result["domain"] + sep + result["lang"] 
            if d not in out:
                out[d] = 0
            out[d] += result["length"]
            f.write("{}\t{}\t{}\t{}\t{}\n".format(
                result["url"], result["domain"], result["lang"],
                result["language_score"], result["perplexity"]))
    with open(langstat_outpath, "a") as f:
        for key, value in tqdm(out.items()):
            key = key.split(sep)
            if len(key) == 2:
                f.write('{}\t{}\t{}\n'.format(key[0], key[1], value))


def _create_hash(fname):
    hashes = defaultdict(int)
    for line, mode in tqdm(_corpus_loader(_file_loader(fname))):
        if mode is not None and not mode:
            hashes[hashlib.sha1(bytes(line.lower(), encoding="utf-8")).digest()] += 1
    return hashes


def create_hashes(files):
    pool = Pool(os.cpu_count())
    hashes_list = pool.map(_create_hash, files)
    hashes = defaultdict(int)
    for h in hashes_list:
        hashes.update(h)
    pool.close()
    return hashes


def _load_lm(bin_dir, lang):
    kpath = os.path.join(bin_dir, "lm_sp", lang+".arpa.bin")
    spath = os.path.join(bin_dir, "lm_sp", lang+".sp.model")
    lm = kenlm.Model(kpath)
    sp = spm.SentencePieceProcessor()
    sp.Load(spath)
    return lm, sp


def _group_n(lis, n):
    for i in range(0, len(lis), n):
        yield lis[i:i+n]


def _save_bulk(results, tmp_dir, fprefix, langs):
    _save_func = partial(_save_to_tmp,
                         tmp_dir=tmp_dir, fprefix=fprefix, langs=langs)
    for result in results:
        _save_func(result)
    del(results)
    gc.collect()


class LoaderProxy:
    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        for result in self._loader:
            yield result


def _check_process(ps):
    stack = []
    for i, p in enumerate(ps):
        if p.is_alive():
            continue
        else:
            p.close()
            stack.append(i)
    for i in stack[::-1]:
        ps.pop(i)
    return ps
            
            
def _parallel_s(files, hashes, tmp_dir, fprefix, langs):
    loaders = [LoaderProxy((_detect_lang(x)
                            for x in _corpus_loader_dedup(
                                    tqdm(_file_loader(fname)), hashes)))
               for fname in files]
    _save_func = partial(_save_bulk,
                         tmp_dir=tmp_dir,
                         fprefix=fprefix,
                         langs=langs)
    count = 0
    ps = []
    loader = loaders.pop(0)
    p = Process(target=_save_func, args=(loader, ))
    p.start()
    ps.append(p)
    while loaders or ps:
        ps = _check_process(ps)
        while len(ps) < os.cpu_count():
            print(count)
            count += 1
            loader = loaders.pop(0)
            p = Process(target=_save_func, args=(loader, ))
            p.start()
            ps.append(p)
    del(loaders)
    gc.collect()


def _parallel_t(fprefix, score_outpath, langstat_outpath):
    pass

    
def main(score_outpath, langstat_outpath, tmp_dir="./tmp"):
    #pool = Pool(os.cpu_count())
    fprefix = randomString(20)
    files = [x.strip() for x in sys.stdin]
    #hashes = create_hashes(files)
    hashes = defaultdict(int)
    _parallel_s(files, hashes, tmp_dir, fprefix, langs)
    #func = partial(_output,
    #               score_outpath=score_outpath,
    #               langstat_outpath=langstat_outpath)
    #for results in tqdm(_add_lang_score_bulk(fprefix)):
    #    func(results)
    #line_gen = (x for x in tqdm(_file_loader_bulk(files)))
    #results = (_detect_lang(x) for x in _corpus_loader_dedup(line_gen, hashes))
    #_save_func = partial(_save_to_tmp,
    #                     tmp_dir=tmp_dir, fprefix=fprefix, langs=langs)
    #list(pool.imap_unordered(_save_func, results))
    #pool.close()


if __name__ == "__main__":
    main(sys.argv[2], sys.argv[3])
