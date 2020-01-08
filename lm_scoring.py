import gc
import os
import json
import text_normalizer
import kenlm
import sentencepiece as spm
from multiprocessing.pool import Pool

LM_MODEL = None
SP_MODEL = None
DEFAULT_CPUS = os.cpu_count()


def _load_lm(lang, bin_dir):
    kpath = os.path.join(bin_dir, "lm_sp", lang + ".arpa.bin")
    spath = os.path.join(bin_dir, "lm_sp", lang + ".sp.model")
    lm = kenlm.Model(kpath)
    sp = spm.SentencePieceProcessor()
    sp.Load(spath)
    return lm, sp


def _initializer(lm_s, sp_s):
    global LM_MODEL, SP_MODEL
    LM_MODEL = lm_s
    SP_MODEL = sp_s


def _jl_loader(tmp_dir, fprefix, lang):
    with open(os.path.join(tmp_dir, fprefix + "_{}".format(lang))) as f:
        for line in f:
            yield line


def _output(results, score_outpath, langstat_outpath):
    out = {}
    sep = "_____"
    urllist = set()
    with open(score_outpath, "a") as f:
        for result in results:
            if result["url"] in urllist:
                continue
            d = result["domain"] + sep + result["lang"]
            if d not in out:
                out[d] = 0
            out[d] += result["length"]
            urllist.add(result["url"])
            f.write("{}\t{}\t{}\t{}\t{}\n".format(result["url"],
                                                  result["domain"],
                                                  result["lang"],
                                                  result["language_score"],
                                                  result["perplexity"]))
    del urllist
    gc.collect()
    with open(langstat_outpath, "a") as f:
        for key, value in out.items():
            key = key.split(sep)
            if len(key) == 2:
                f.write('{}\t{}\t{}\n'.format(key[0], key[1], value))


def _add_lang_score(line):
    global LM_MODEL, SP_MODEL
    result = json.loads(line.strip())
    doc_score = 0
    doc_length = 0
    for line in result["data"]:
        line = text_normalizer.normalize(line)
        pieces = ' '.join(SP_MODEL.EncodeAsPieces(line))
        if len(pieces):
            doc_score += LM_MODEL.score(' '.join(pieces))
            doc_length += len(pieces)
    if doc_length:
        result["perplexity"] = 10.0**(-doc_score / doc_length)
    else:
        result["perplexity"] = 0.0
    del result["data"]
    return result


def _add_lang_score_bulk(line_gen,
                         lang,
                         score_outpath,
                         langstat_outpath,
                         bin_dir,
                         num_cpus=DEFAULT_CPUS):
    lm_s, sp_s = _load_lm(lang, bin_dir)
    pool = Pool(num_cpus, _initializer, (lm_s, sp_s))
    _output(pool.imap_unordered(_add_lang_score, line_gen), score_outpath,
            langstat_outpath)
    pool.close()
    del lm_s
    del sp_s
    gc.collect()


def do(fprefix,
       score_outpath,
       langstat_outpath,
       bin_dir,
       tmp_dir="./tmp",
       num_cpus=DEFAULT_CPUS):
    target_langs = list({
        x.split("_")[-1]
        for x in os.listdir(tmp_dir) if x.startswith(fprefix)
    })
    for lang in target_langs:
        loader = _jl_loader(tmp_dir, fprefix, lang)
        _add_lang_score_bulk(loader, lang, score_outpath, langstat_outpath,
                             bin_dir, num_cpus)
