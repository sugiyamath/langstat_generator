import gc
import os
import sys
import json
import fasttext
import wet_loader
from functools import partial
from multiprocessing import Process

LID_MODEL = None
DEFAULT_CPUS = os.cpu_count()


class LoaderProxy:
    def __init__(self, loader):
        self._loader = loader

    def __iter__(self):
        for result in self._loader:
            yield result


def _detect_lang(batch):
    out = {}
    lscores = []
    for line in batch["data"]:
        tmp_pred = LID_MODEL.predict(line)
        lang = tmp_pred[0][0].split("__")[-1]
        lscores.append(float(tmp_pred[1][0]))
        if lang not in out:
            out[lang] = []
        out[lang].append(line)
    lscore = sum(x / len(lscores) for x in lscores)
    out = max(out.items(), key=lambda x: len(x[1]))
    sys.stdout.flush()
    return {
        "lang": out[0],
        "language_score": float(lscore),
        "length": len(''.join(out[1])),
        "url": batch["url"],
        "domain": batch["domain"],
        "data": batch["data"]
    }


def _save_to_tmp(result, tmp_dir, fprefix, langs):
    if result["lang"] in langs:
        with open(
                os.path.join(tmp_dir, fprefix + "_{}".format(result["lang"])),
                "a") as f:
            f.write(json.dumps(result) + "\n")


def _save_bulk(results, tmp_dir, fprefix, langs):
    _save_func = partial(_save_to_tmp,
                         tmp_dir=tmp_dir,
                         fprefix=fprefix,
                         langs=langs)
    for result in results:
        _save_func(result)
    del results
    gc.collect()


def _check_process(ps):
    stack = []
    for i, p in enumerate(ps):
        if p.is_alive():
            continue
        else:
            try:
                p.close()
            except AttributeError:
                pass
            stack.append(i)
    for i in stack[::-1]:
        ps.pop(i)
    return ps


def do(files,
       hashes,
       tmp_dir,
       fprefix,
       langs,
       bin_dir,
       num_cpus=DEFAULT_CPUS):
    global LID_MODEL
    LID_MODEL = fasttext.load_model(os.path.join(bin_dir, "lid.bin"))
    loaders = [
        LoaderProxy((_detect_lang(x)
                     for x in wet_loader.corpus_loader_dedup(
                             wet_loader.file_loader(fname), hashes)))
        for fname in files
    ]
    _save_func = partial(_save_bulk,
                         tmp_dir=tmp_dir,
                         fprefix=fprefix,
                         langs=langs)
    ps = []
    while loaders or ps:
        ps = _check_process(ps)
        while len(ps) < num_cpus:
            if loaders:
                loader = loaders.pop(0)
                p = Process(target=_save_func, args=(loader, ))
                p.start()
                ps.append(p)
            else:
                ps = _check_process(ps)
                if not ps:
                    break
    del loaders
    gc.collect()
