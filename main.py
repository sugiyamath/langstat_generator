
import gc
import os
import sys
import time
import shutil
import tempfile
import datetime

import hash_creator
import lang_separator
import lm_scoring
import data_downloader
import utils
import sharding
from functools import partial


def _pt(st):
    return str(datetime.timedelta(seconds=time.time() - st))


def _plog(msg, nid, sid, st):
    print("{}\t{}\tnid:{}, sid:{}, msg:{} ".format(
        datetime.datetime.now(), _pt(st), nid, sid, msg))


def _rmall(target_dir):
    for fname in os.listdir(target_dir):
        fpath = os.path.join(target_dir, fname)
        os.unlink(fpath)


def _mvall(suffix, source_dir, target_dir):
    for fname in os.listdir(source_dir):
        if fname.endswith(suffix):
            shutil.move(os.path.join(source_dir, fname),
                        os.path.join(target_dir, fname))


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
    _plog_t = partial(_plog, nid=node_id, sid=ss, st=st)
    _plog_t("TASK START")
    langs = utils.available_langs(bin_dir)
    
    if num_dl_parallel > 0:
        all_targets = [x.strip() for x in sys.stdin]
        is_target_http = True
    else:
        all_targets = [x.strip() for x in sys.stdin if "CC-MAIN" in x]
        is_target_http = False

    for shard_id in range(max_shard_num):
        _plog_s = partial(_plog, nid=node_id, sid=shard_id, st=st)
        _plog_s("start shard")
        score_outpath = None
        langstat_outpath = None
        with tempfile.TemporaryDirectory(dir="/tmp") as tmp_dir:
            try:
                #targets = all_targets
                targets = list(
                    sharding.sharding(all_targets,
                                      node_id,
                                      shard_id,
                                      total_nodes,
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

                _plog_s("hash_creator: start")
                hashes = hash_creator.create_hashes(files, num_cpus)
                _plog_s("hash_creator: done")

                _plog_s("lang_separator: start")
                lang_separator.do(
                    files, hashes, tmp_dir, fprefix, langs, bin_dir, num_cpus)
                _plog_s("lang_separator: done")

                _plog_s("hash_creaning: start")
                del hashes
                gc.collect()
                _plog_s("hash_creaning: done")

                _plog_s("lm_scoring: start")
                score_outpath = os.path.join(
                    tmp_dir, "lmscore_{}_{}.txt".format(node_id, shard_id))
                langstat_outpath = os.path.join(
                    tmp_dir, "langstat_{}_{}.txt".format(node_id, shard_id))
                lm_scoring.do(
                    fprefix, score_outpath, langstat_outpath, bin_dir,
                    tmp_dir, num_cpus)
                _plog_s("lm_scoring: done")
            except Exception as e:
                _plog_s("Error: {}".format(str(e)))
            finally:
                _mvall(".txt", tmp_dir, out_dir)
                _rmall(tmp_dir)
            
    _plog("TASK DONE", node_id, ss, st)


if __name__ == "__main__":
    main(*sys.argv[1:])
