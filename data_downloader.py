import os
import requests
from multiprocessing.pool import Pool
from tqdm import tqdm
from functools import partial

DEFAULT_CPUS = os.cpu_count()


def _download_file(url, tmp_dir):
    local_filename = url.split('/')[-1]
    filepath = os.path.join(tmp_dir, local_filename)
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filepath, 'wb') as f:
            for chunk in tqdm(r.iter_content(chunk_size=8192)):
                if chunk:
                    f.write(chunk)
    return filepath


def download_bulk(urls, tmp_dir, num_cpus=DEFAULT_CPUS):
    pool = Pool(num_cpus)
    func = partial(_download_file, tmp_dir=tmp_dir)
    files = pool.map(func, urls)
    pool.close()
    return files
