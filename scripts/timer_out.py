import sys
import time
from multiprocessing import Process


def print_symbol(symbol=".", sleep=0.4):
    while True:
        print(symbol, end='', flush=True)
        time.sleep(sleep)


if __name__ == "__main__":
    p = Process(target=print_symbol, args=(".", ))
    p.start()
    for line in sys.stdin:
        print("\n" + line, flush=True)
    p.terminate()
    while p.is_alive():
        pass
    p.close()
        
            
    
