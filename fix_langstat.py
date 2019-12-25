
def fix(lines):
    prev = ""
    prev_value = 0
    count = 1
    for line in lines:
        tmp = line.strip().split("\t")
        current = '\t'.join(tmp[:2])
        current_value = int(tmp[2])
        if prev == current:
            prev_value += current_value
            count += 1
            continue
        else:
            if prev and prev_value > 0:
                yield prev+"\t"+str(float(prev_value)/float(count))
            count = 1
        prev = current
        prev_value = current_value
    yield prev+'\t'+str(prev_value)


if __name__ == "__main__":
    import sys
    lines = (x for x in sys.stdin)
    for x in fix(lines):
        sys.stdout.write(x+"\n")
