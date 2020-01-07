import gc
import os
import sys
import time
import tempfile
import datetime

import hash_creator
import lang_separator
import lm_scoring
import data_downloader
import utils
import sharding


def _pt(st):
    return str(datetime.timedelta(seconds=time.time() - st))


def _plog(msg, nid, sid, st):
    print("{}\t{}\tnid:{}, sid:{}, msg:{} ".format(
        datetime.time(), _pt(st), nid, sid, msg))


def main(node_id,
         max_shard_num,
         bin_dir,
         out_dir,
         total_nodes=7,
         wet_per_shard=50,
         num_cpus=12,
         num_dl_parallel=0):
    node_id = int(node_id)
    max_shard_num = int(max_shard_num)
    total_nodes = int(total_nodes)
    wet_per_shard = int(wet_per_shard)
    num_cpus = int(num_cpus)
    num_dl_parallel = int(num_dl_parallel)
    is_target_http = None

    st = time.time()
    ss = ','.join(map(str, list(range(max_shard_num))))
    _plog("TASK START", node_id, ss, st)
    langs = utils.available_langs(bin_dir)
    
    if num_dl_parallel > 0:
        all_targets = [x.strip() for x in sys.stdin]
        is_target_http = True
    else:
        all_targets = [x.strip() for x in sys.stdin if "CC-MAIN" in x]
        is_target_http = False

    for shard_id in range(max_shard_num):
        _plog("start shard", node_id, shard_id, st)
        with tempfile.TemporaryDirectory(dir=".") as tmp_dir:
            targets = list(
                sharding.sharding(all_targets, node_id, shard_id, total_nodes,
                                  wet_per_shard))
            if is_target_http:
                files = data_downloader.download_bulk(targets, tmp_dir,
                                                      num_dl_parallel)
                files = [os.path.join(tmp_dir, x)
                         for x in os.listdir(tmp_dir)
                         if x.startswith("CC-MAIN")]
            else:
                files = targets

            fprefix = utils.random_string(20)

            _plog("hash_creator: start", node_id, shard_id, st)
            hashes = hash_creator.create_hashes(files, num_cpus)
            _plog("hash_creator: done", node_id, shard_id, st)

            _plog("lang_separator: start", node_id, shard_id, st)
            lang_separator.do(
                files, hashes, tmp_dir, fprefix, langs, bin_dir, num_cpus)
            _plog("lang_separator: done", node_id, shard_id, st)

            _plog("hash_creaning: start", node_id, shard_id, st)
            del hashes
            gc.collect()
            _plog("hash_creaning: done", node_id, shard_id, st)

            _plog("lm_scoring: start", node_id, shard_id, st)
            score_outpath = os.path.join(
                out_dir, "lmscore_{}_{}.txt".format(node_id, shard_id))
            langstat_outpath = os.path.join(
                out_dir, "langstat_{}_{}.txt".format(node_id, shard_id))
            lm_scoring.do(
                fprefix, score_outpath, langstat_outpath, bin_dir,
                tmp_dir, num_cpus)
            _plog("lm_scoring: done", node_id, shard_id, st)
    _plog("TASK DONE", node_id, ss, st)


if __name__ == "__main__":
    main(*sys.argv[1:])
