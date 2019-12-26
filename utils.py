import os
import random
import string


def available_langs(bin_dir):
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


def random_string(length=10):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))
