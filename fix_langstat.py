
def fix(lines):
    prev = ""
    prev_value = 0
    for line in lines:
        tmp = line.strip().split("\t")
        current = '\t'.join(tmp[:2])
        current_value = int(tmp[2])
        if prev == current:
            prev_value += current_value
            continue
        else:
            if prev and prev_value > 0:
                yield prev+"\t"+str(prev_value)
        prev = current
        prev_value = current_value
    yield prev+'\t'+str(prev_value)


if __name__ == "__main__":
    import sys
    lines = (x for x in sys.stdin) # this is sorted
    for x in fix(lines):
        sys.stdout.write(x+"\n")
