"""Verify staged artifacts and start training with a persistent Python environment."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request


def terminate():
    while True:
        request = urllib.request.Request('https://api.runpod.io/v2/pods/' + os.environ['RUNPOD_POD_ID'],
                                         method='DELETE', headers={'Authorization': 'Bearer ' + os.environ['RUNPOD_SHUTDOWN_KEY']})
        try:
            with urllib.request.urlopen(request, timeout=30):
                return
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return
            print('Shutdown retry HTTP', error.code, flush=True)
        except Exception as error:
            print('Shutdown retry', type(error).__name__, flush=True)
        time.sleep(10)


if __name__ == '__main__':
    try:
        # The deadline includes extraction and dependency setup, not just training.
        watchdog = "import os,time,runpy; time.sleep(float(os.environ.get('MAX_RUNTIME_SECONDS','86400'))); runpy.run_path('/workspace/staging/bootstrap.py')['terminate']()"
        subprocess.Popen([sys.executable, '-u', '-c', watchdog], start_new_session=True)
        if not os.path.ismount('/workspace'):
            raise RuntimeError('Network volume is not mounted at /workspace')
        for item in json.loads(Path('/workspace/staging/manifest.json').read_text()):
            digest = hashlib.sha256()
            with (Path('/workspace') / item['key']).open('rb') as stream:
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != item['sha256']:
                raise RuntimeError('Artifact checksum mismatch: ' + item['key'])
            print('Checksum verified:', item['key'], flush=True)
        Path('/workspace/code').mkdir(exist_ok=True)
        with tarfile.open('/workspace/staging/source.tar.gz') as archive:
            archive.extractall('/workspace/code', filter='data')
        marker = Path('/workspace/staging/dataset-extracted')
        if not marker.exists():
            Path('/workspace/datasets').mkdir(exist_ok=True)
            with tarfile.open('/workspace/staging/dataset.tar') as archive:
                archive.extractall('/workspace/datasets', filter='data')
            marker.write_text('complete\n')
        environment = Path('/workspace/venv')
        if not (environment / 'bin/python').exists():
            subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(environment)], check=True)
        subprocess.run([str(environment / 'bin/python'), '-u', '/workspace/code/scripts/runpod/batch_train.py'], check=True)
    except BaseException as error:
        print('Bootstrap failed:', type(error).__name__, str(error), flush=True)
    finally:
        os.sync()
        terminate()
