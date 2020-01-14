
import os
import numpy as np
from collections import defaultdict
from tqdm import tqdm


def insert_mean_and_sd(d, out, i, j):
    for domain, values in tqdm(d.items()):
        out[domain][i] = float(np.mean(values))
        out[domain][j] = float(np.std(values))


def merge(in1_file, in2_file, out_file):
    sep = "____"
    ldict_ls = defaultdict(list)
    ldict_lm = defaultdict(list)
    out = {}

    print("Load score")
    with open(in1_file, "r") as f:
        for line in tqdm(f):
            line = line.strip().split("\t")
            if len(line) == 5:
                domain = line[1]+sep+line[2]
                lang_score = float(line[3])
                lm_score = float(line[4])
                ldict_ls[domain].extend([lang_score])
                ldict_lm[domain].extend([lm_score])
                if domain not in out:
                    out[domain] = [None, 0.0, 0.0, 0.0, 0.0]
    print("Insert mean and sd")
    insert_mean_and_sd(ldict_ls, out, 1, 2)
    insert_mean_and_sd(ldict_lm, out, 3, 4)

    print("Load langstat")
    with open(in2_file, "r") as f:
        for line in tqdm(f):
            line = line.strip().split("\t")
            domain = line[0]+sep+line[1]
            value = int(line[2])
            if domain in out:
                if out[domain][0] is None:
                    out[domain][0] = value
                elif out[domain][0] < value:
                    out[domain][0] = value
    print("Save")
    with open(out_file, "w") as f:
        for domain, values in tqdm(out.items()):
            if values[0] is not None:
                f.write(
                    '\t'.join(list(map(str, domain.split(sep)+values)))+'\n')


def main(target_dir):
    in1_files = [fname for fname in os.listdir(target_dir)
                 if fname.startswith("langstat")]
    in2_files = [fname for fname in os.listdir(target_dir)
                 if fname.startswith("lmscore")]

    files = []
    stack1 = in1_files[:]
    stack2 = in2_files[:]
    while stack1 and stack2:
        target1 = stack1.pop(0)
        iden1 = target1.split("_")
        pop_index = None
        for i, fname in enumerate(stack2):
            if pop_index is None:
                iden2 = fname.split("_")
                if iden1[1] == iden2[1] and iden1[2] == iden2[2]:
                    pop_index = i
        if pop_index is not None:
            target2 = stack2.pop(pop_index)
            files.append((target1, target2))
    for fnames in tqdm(files):
        merge(
            os.path.join(target_dir, fnames[1]),
            os.path.join(target_dir, fnames[0]),
            os.path.join(target_dir, fnames[0]+".mgd"))


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
