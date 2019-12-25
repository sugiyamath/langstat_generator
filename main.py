import gc
import os
import random
import string
import sys

import hash_creator
import lang_separator
import lm_scoring


def _available_langs(bin_dir):
    ls1 = {
        x.split(".")[0]
        for x in os.listdir(os.path.join(bin_dir, "lm_sp"))
        if x.endswith(".arpa.bin")
    }
    ls2 = {
        x.split(".")[0]
        for x in os.listdir(os.path.join(bin_dir, "lm_sp"))
        if x.endswith(".sp.model")
    }
    langs = ls1 & ls2
    return langs


def _random_string(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))


def main(bin_dir, score_outpath, langstat_outpath, tmp_dir="./tmp"):
    langs = _available_langs(bin_dir)
    fprefix = _random_string(20)
    files = [x.strip() for x in sys.stdin]
    hashes = hash_creator.create_hashes(files)
    lang_separator.do(files, hashes, tmp_dir, fprefix, langs, bin_dir)
    del hashes
    gc.collect()
    lm_scoring.do(fprefix, score_outpath, langstat_outpath, bin_dir, tmp_dir)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
