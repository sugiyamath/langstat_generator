import gc
import os
import sys
import tempfile

import hash_creator
import lang_separator
import lm_scoring
import data_downloader
import utils
import sharding


def main(node_id,
         max_shard_num,
         bin_dir,
         out_dir,
         total_nodes=7,
         wet_per_shard=50,
         num_cpus=12,
         num_dl_parallel=1):
    node_id = int(node_id)
    max_shard_num = int(max_shard_num)
    total_nodes = int(total_nodes)
    wet_per_shard = int(wet_per_shard)
    num_cpus = int(num_cpus)
    num_dl_parallel = int(num_dl_parallel)
    urls = [x.strip() for x in sys.stdin]
    for shard_id in range(max_shard_num):
        with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
            target_urls = list(
                sharding.sharding(urls, node_id, shard_id, total_nodes,
                                  wet_per_shard))
            langs = utils.available_langs(bin_dir)
            fprefix = utils.random_string(20)
            files = data_downloader.download_bulk(target_urls, tmp_dir,
                                                  num_dl_parallel)
            hashes = hash_creator.create_hashes(files, num_cpus)
            lang_separator.do(files, hashes, tmp_dir, fprefix, langs, bin_dir,
                              num_cpus)
            del hashes
            gc.collect()
            score_outpath = os.path.join(
                out_dir, "lmscore_{}_{}.txt".format(node_id, shard_id))
            langstat_outpath = os.path.join(
                out_dir, "langstat_{}_{}.txt".format(node_id, shard_id))
            lm_scoring.do(fprefix, score_outpath, langstat_outpath, bin_dir,
                          tmp_dir, num_cpus)


if __name__ == "__main__":
    main(*sys.argv[1:])
