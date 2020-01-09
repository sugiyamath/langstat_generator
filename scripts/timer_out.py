import os
import sys
import time
from multiprocessing import Process


def print_symbol(symbol=".", sleep=0.4):
    while True:
        print(symbol, end='', flush=True)
        time.sleep(sleep)


if __name__ == "__main__":
    node_id = sys.argv[1]
    log_dir = sys.argv[2]
    p = Process(target=print_symbol, args=(".", ))
    p.start()
    for line in sys.stdin:
        with open(os.path.join(log_dir, "{}.log".format(node_id)), "a") as f:
            f.write(line)
    p.terminate()
    while p.is_alive():
        pass
    p.close()
        
            
    
