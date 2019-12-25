def sharding(lines, node_id, shard_id, total_nodes=7, wet_per_shard=50):
    count = 0
    for i, line in enumerate(lines):
        if i % total_nodes == node_id:
            count += 1
            shard = count // wet_per_shard
            if shard_id == shard:
                yield line


if __name__ == "__main__":
    import sys
    lines = (x for x in sys.stdin)
    for line in sharding(lines, int(sys.argv[1]), int(sys.argv[2])):
        sys.stdout.write(line)
