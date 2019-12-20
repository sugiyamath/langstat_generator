import sys
import hashlib
from collections import defaultdict, ChainMap
from tqdm import tqdm
import gzip
from multiprocessing.pool import Pool
import fasttext
import os
import text_normalizer
import kenlm
import sentencepiece as spm
from functools import partial

bin_dir = sys.argv[1]
lid_model = fasttext.load_model(os.path.join(bin_dir, "lid.bin"))


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
            "language_score": lscore,
            "length": len(''.join(out[1])),
            "url": batch["url"],
            "domain": batch["domain"],
            "data": batch["data"]}


def _add_lang_score(result, lm, sp):
    pp_scores = []
    for line in result["data"]:
        line = text_normalizer.normalize(line)
        pieces = ' '.join(sp.encode_as_pieces(line))
        log_score = lm.score(' '.join(pieces))
        if len(pieces):
            pp_scores.append(10.0**(-log_score/len(pieces)))    
    pp_score = sum(x/len(pp_scores) for x in pp_scores)
    result["perplexity"] = pp_score
    del(result["data"])
    return result


def _split_by_lang(results):
    while results:
        stack = []
        current_lang = None
        ignore = False
        for i, result in enumerate(results):
            if current_lang is None:
                current_lang = result["lang"]
                try:
                    lm, sp = _load_lm(bin_dir, current_lang)
                except:
                    ignore = True
            if result["lang"] == current_lang:
                if not ignore:
                    yield _add_lang_score(result, lm, sp)
                stack.append(i)
        for i in stack[::-1]:
            results.pop(i)


def _output(results, score_outpath, langstat_outpath):
    out = {}
    sep = "_____"
    with open(score_outpath, "a") as f:
        for result in tqdm(results):
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

        
def main(score_outpath, langstat_outpath):
    pool = Pool(os.cpu_count())
    files = [x.strip() for x in sys.stdin]
    hashes = create_hashes(files)
    line_gen = (x for x in tqdm(_file_loader_bulk(files)))
    results = pool.map(_detect_lang, _corpus_loader_dedup(line_gen, hashes))
    results = _split_by_lang(results)
    _output(results, score_outpath, langstat_outpath)


if __name__ == "__main__":
    main(sys.argv[2], sys.argv[3])
