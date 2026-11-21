"""Independent in-Pod wall-clock cutoff; never removes the network volume."""
import os
import runpy
import time
from pathlib import Path

if __name__ == '__main__':
    time.sleep(float(os.environ.get('MAX_RUNTIME_SECONDS', '86400')))
    module = runpy.run_path(str(Path(__file__).with_name('batch_train.py')))
    print('Maximum runtime reached; terminating training Pod', flush=True)
    os.sync()
    module['terminate']()
