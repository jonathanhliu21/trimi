"""One-shot scheduled queue check and bounded, deterministic Runpod fallback."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from dotenv import dotenv_values
import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = ROOT / 'data/color_palm_camera_hand_dinov3_20261125_01'
CREDENTIALS = ROOT.parent / '.env'
SUPPORTED_S3 = {'EU-CZ-1', 'EU-RO-1', 'EUR-IS-1', 'EUR-NO-1', 'US-CA-2',
                'US-GA-2', 'US-IL-1', 'US-KS-2', 'US-MD-1', 'US-MO-1', 'US-MO-2',
                'US-NC-1', 'US-NC-2', 'US-NE-1', 'US-WA-1'}
GPU_IDS = ['NVIDIA RTX PRO 6000 Blackwell Server Edition',
           'NVIDIA RTX PRO 6000 Blackwell Workstation Edition',
           'NVIDIA H100 PCIe', 'NVIDIA H100 NVL', 'NVIDIA H200 NVL',
           'NVIDIA H100 80GB HBM3', 'NVIDIA H200']


class RunpodError(RuntimeError):
    def __init__(self, tool, capacity=False):
        super().__init__('Runpod tool failed: ' + tool)
        self.capacity = capacity


def save(path, obj):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, indent=2))
    temporary.replace(path)


class MCP:
    def __init__(self, key):
        self.session = requests.Session()
        self.session.headers.update(Authorization='Bearer ' + key,
                                    Accept='application/json, text/event-stream')
        self.sequence = 0
        self.rpc('initialize', {'protocolVersion': '2024-11-05', 'capabilities': {},
                               'clientInfo': {'name': 'trimi-urgent-followup', 'version': '1'}})
        self.tools = {t['name']: t for t in self.rpc('tools/list', {})['tools']}

    def rpc(self, method, params):
        self.sequence += 1
        response = self.session.post('https://mcp.runpod.io/', json={
            'jsonrpc': '2.0', 'id': self.sequence, 'method': method, 'params': params}, timeout=120)
        response.raise_for_status()
        if response.headers.get('Mcp-Session-Id'):
            self.session.headers['Mcp-Session-Id'] = response.headers['Mcp-Session-Id']
        lines = [line[6:] for line in response.text.splitlines() if line.startswith('data: ')]
        value = json.loads(lines[-1] if lines else response.text)
        if value.get('error'):
            raise RuntimeError('MCP protocol error; response withheld to protect credentials')
        return value['result']

    def call(self, tool, **arguments):
        assert tool in self.tools, tool
        result = self.rpc('tools/call', {'name': tool, 'arguments': arguments})
        if result.get('isError'):
            # Never include a create response or echoed environment in logs.
            message = str(result).lower()
            raise RunpodError(tool, any(word in message for word in ('capacity', 'no available', 'out of stock')))
        content = '\n'.join(c['text'] for c in result.get('content', []) if c['type'] == 'text')
        return json.loads(content) if content else {}

    def pages(self, name, field, **arguments):
        rows = []
        while True:
            result = self.call(name, **arguments)
            rows.extend(result[field])
            pagination = result.get('pagination', {})
            if not pagination.get('hasNextPage'):
                return rows
            arguments['cursor'] = pagination['nextCursor']


def s3(keys, dc):
    return boto3.client('s3', endpoint_url=f'https://s3api-{dc.lower()}.runpod.io', region_name=dc,
                        aws_access_key_id=keys['RUNPOD_S3_API_KEY_USER'],
                        aws_secret_access_key=keys['RUNPOD_S3_API_KEY_SECRET'],
                        config=Config(connect_timeout=20, read_timeout=180, retries={'max_attempts': 3},
                                      request_checksum_calculation='when_required',
                                      response_checksum_validation='when_required'))


def candidates(mcp, ceiling):
    gpus = mcp.pages('list-gpu-types', 'gpus', include=['AVAILABILITY'], product=['POD'],
                     cloud='SECURE', minCudaVersion='12.8', limit=100)
    dcs = mcp.pages('list-data-centers', 'dataCenters', networkVolumeTypes=['STANDARD'], limit=100)
    allowed = {dc['id'] for dc in dcs} & SUPPORTED_S3
    result = []
    for gpu in gpus:
        price = gpu.get('price', {}).get('secure', 0)
        if gpu['id'] not in GPU_IDS or not 0 < price <= ceiling or gpu['memory'] < 80:
            continue
        for dc in gpu.get('dataCenters', []):
            if dc['id'] in allowed and dc.get('availability') in ('LOW', 'MEDIUM', 'HIGH'):
                result.append({'gpu': gpu['id'], 'data_center': dc['id'], 'price': price})
    return sorted(result, key=lambda x: (x['data_center'] != 'EUR-IS-1',
                                         GPU_IDS.index(x['gpu']), x['price'], x['data_center']))


def follow_queue(mcp):
    deployment = json.loads((ORIGINAL / 'deployment.json').read_text())
    row = deployment['pods'][0]
    queued = mcp.call('get-queued-pod', id=row['queue_id'])
    if queued.get('podId'):
        return {'action': 'original_deployed', 'pod_id': queued['podId'], 'deployment': str(ORIGINAL / 'deployment.json')}
    return None


def stage(keys, mcp, selection, journal):
    dc = selection['data_center']
    if dc == 'EUR-IS-1':
        return '8zlot4olf9'
    if dc in journal.get('staged_volumes', []):
        return journal['volumes'][dc]
    if dc not in journal['volumes']:
        mcp.call('describe-tool', name='create-network-volume')
        volume = mcp.call('create-network-volume', body={
            'name': journal['campaign'] + '_' + dc, 'size': 20, 'dataCenter': dc, 'type': 'STANDARD'})
        volume = volume.get('networkVolume', volume)
        journal['volumes'][dc] = volume['id']
        save(ORIGINAL / 'followup/progress.json', journal)
    volume_id = journal['volumes'][dc]
    # Read back every newly created resource without logging credentials.
    listed = mcp.pages('list-network-volumes', 'networkVolumes', limit=100)
    assert any(v['id'] == volume_id and v['dataCenter'] == dc for v in listed)
    source, target = s3(keys, 'EUR-IS-1'), s3(keys, dc)
    manifest = json.loads((ORIGINAL / 'manifest.json').read_text())
    cache = ORIGINAL / 'followup/artifact_cache'
    cache.mkdir(exist_ok=True)
    for artifact in manifest['artifacts']:
        path = cache / Path(artifact['key']).name
        if not path.exists() or path.stat().st_size != artifact['size']:
            source.download_file('8zlot4olf9', artifact['key'], str(path))
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(8 * 1024**2), b''):
                digest.update(block)
        assert digest.hexdigest() == artifact['sha256']
        try:
            target.upload_file(str(path), volume_id, artifact['key'], Config=TransferConfig(
                multipart_threshold=64 * 1024**2, multipart_chunksize=64 * 1024**2, max_concurrency=4))
        except Exception:
            # Multipart completion can time out after successfully committing.
            assert target.head_object(Bucket=volume_id, Key=artifact['key'])['ContentLength'] == artifact['size']
        assert target.head_object(Bucket=volume_id, Key=artifact['key'])['ContentLength'] == artifact['size']
        print('Staged artifact:', artifact['key'], flush=True)
    journal.setdefault('staged_volumes', []).append(dc)
    save(ORIGINAL / 'followup/progress.json', journal)
    return volume_id


def start_watchdog(deployment):
    unit = 'trimi-watch-' + deployment.parent.name.replace('_', '-')
    subprocess.run(['systemd-run', '--user', '--collect', '--unit=' + unit,
                    '--property=Restart=on-failure', '--property=RestartSec=15',
                    sys.executable, '-u', str(ROOT / 'scripts/runpod/watch_campaign.py'),
                    '--deployment', str(deployment), '--credentials', str(CREDENTIALS)], check=True)


def verify(keys, outcome):
    """Record actual runtime evidence; never equate pod creation with verified training."""
    path = Path(outcome['deployment'])
    local = path.parent
    deployment = json.loads(path.read_text())
    target = s3(keys, deployment['data_center'])
    row = deployment['pods'][0]
    prefix = row['status_key'].rsplit('/', 1)[0] + '/'
    baseline = yaml.safe_load((ORIGINAL / 'reference_camera_config.yaml').read_text())
    expected = json.loads((ORIGINAL / 'dataset_report.json').read_text())
    def read(name):
        return target.get_object(Bucket=deployment['volume_id'], Key=prefix + name)['Body'].read().decode()
    def remove_hand(value):
        if isinstance(value, dict):
            return {key: remove_hand(item) for key, item in value.items() if key != 'hand_pose_abs'}
        if isinstance(value, list):
            return [remove_hand(item) for item in value]
        return value
    while time.time() < row['deadline_timestamp']:
        result = {'successful_launch': False, 'checked_at': time.time(), 'pod_id': outcome['pod_id']}
        state = {}
        try:
            state = json.loads(read('status.json'))
            result['phase'] = state['phase']
            config = yaml.safe_load(read('resolved_config.yaml'))
            split = json.loads(read('split_verification.json'))
            assert set(config['shape_meta']['obs']) == {'camera_0', 'hand_pose_abs'}
            assert config['shape_meta']['obs']['hand_pose_abs']['shape'] == [13]
            assert set(split['observations']) == {'camera_0', 'hand_pose_abs'}
            assert len(split['train']) == 128 and len(split['validation']) == 23
            assert split['train'] == expected['train_all_indices']
            assert split['validation'] == expected['validation_indices']
            for key in ('policy', 'optimizer', 'ema', 'dataloader', 'val_dataloader', 'training',
                        'horizon', 'num_obs_steps', 'num_action_steps', 'num_inference_steps'):
                assert remove_hand(config[key]) == remove_hand(baseline[key]), key
            result['observations_and_hyperparameters_verified'] = True
            result['smoke_verified'] = all(state.get(key) for key in (
                'backbone_verified', 'checkpoint_reload_verified', 'observations_verified', 'gpu_verified'))
            result['main_backbone_verified'] = 'BACKBONE_VERIFIED' in read('train.log')
            head = target.head_object(Bucket=deployment['volume_id'], Key=prefix + 'checkpoints/latest.ckpt')
            result['main_checkpoint_verified'] = head['ContentLength'] > 1_000_000_000
            query = 'query($name:String!){project(name:"trimi-training",entityName:"jonathanhliu21-ucla"){run(name:$name){state summaryMetrics history(samples:2)}}}'
            response = requests.post('https://api.wandb.ai/graphql', auth=('api', keys['WANDB_API_KEY']),
                                     json={'query': query, 'variables': {'name': row['wandb_run_id']}}, timeout=30)
            response.raise_for_status()
            run = response.json()['data']['project']['run']
            summary = run['summaryMetrics']
            summary = json.loads(summary) if isinstance(summary, str) else summary
            result['online_metrics_verified'] = math.isfinite(float(summary['train_loss'])) and bool(run['history'])
            result['successful_launch'] = all(result.get(key) for key in (
                'observations_and_hyperparameters_verified', 'smoke_verified', 'main_backbone_verified',
                'main_checkpoint_verified', 'online_metrics_verified'))
        except Exception as error:
            result['pending_or_failed_check'] = type(error).__name__
        save(local / 'verification.json', result)
        try:
            target.upload_file(str(local / 'verification.json'), deployment['volume_id'],
                               f"experiments/{deployment['campaign']}/verification.json")
        except Exception:
            pass
        if result['successful_launch'] or state.get('phase') in ('completed', 'failed', 'timed_out'):
            outcome['runtime_verification'] = result
            save(ORIGINAL / 'followup/result.json', outcome)
            return
        time.sleep(60)


def launch(keys, mcp, selection, volume_id, campaign):
    local = ROOT / 'data' / campaign
    local.mkdir(exist_ok=True)
    dc = selection['data_center']
    target = s3(keys, dc)
    for name in ('manifest.json', 'dataset_report.json', 'reference_camera_config.yaml'):
        (local / name).write_bytes((ORIGINAL / name).read_bytes())
    manifest = json.loads((local / 'manifest.json').read_text())
    manifest['campaign'] = campaign
    save(local / 'manifest.json', manifest)
    for name in ('manifest.json', 'dataset_report.json'):
        target.upload_file(str(local / name), volume_id, f'experiments/{campaign}/staging/{name}')
    bootstrap = ROOT / 'scripts/runpod/color_palm_bootstrap.py'
    key = f'experiments/{campaign}/staging/bootstrap.py'
    target.upload_file(str(bootstrap), volume_id, key)
    assert target.get_object(Bucket=volume_id, Key=key)['Body'].read() == bootstrap.read_bytes()
    now = time.time()
    wandb_id = os.urandom(4).hex()
    env = {key: keys[key] for key in ('WANDB_API_KEY', 'RUNPOD_SHUTDOWN_KEY')}
    family = 'RTX PRO 6000' if 'RTX PRO 6000' in selection['gpu'] else ('H200' if 'H200' in selection['gpu'] else 'H100')
    env.update(TRIMI_CAMPAIGN=campaign, TRIMI_DATASET_NAME='data_trimorph_color_palm_pick_place',
               TRIMI_OBSERVATION_MODE='camera_hand', TRIMI_VARIANT='camera_hand_all',
               TRIMI_MAX_TRAIN_EPISODES='null', TRIMI_CHECKPOINT_TOPK='0',
               TRIMI_KEEP_SMOKE_CHECKPOINTS='0', TRIMI_MIN_FREE_GIB='6',
               TRIMI_WANDB_RUN_ID=wandb_id, TRIMI_BOOTSTRAP='/workspace/' + key,
               TRIMI_DEADLINE_TIMESTAMP=str(now + 86400), MAX_RUNTIME_SECONDS='86400',
               TRIMI_RUNTIME_FROM_START='1', TRIMI_EXPECTED_GPU_FAMILY=family,
               TRIMI_MIN_GPU_MEMORY_GIB='90' if family == 'RTX PRO 6000' else '74')
    mcp.call('describe-tool', name='create-pod')
    print('Creating one GPU:', selection['gpu'], dc, selection['price'], 'USD/hour', flush=True)
    pod = mcp.call('create-pod', body={
        'name': campaign, 'image': 'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404',
        'cloud': 'SECURE', 'gpu': {'id': selection['gpu'], 'count': 1, 'minCudaVersion': '12.8'},
        'disk': 80, 'dataCenterIds': [dc], 'mounts': {'network': [{'volumeId': volume_id, 'path': '/workspace'}]},
        'ports': [], 'entrypoint': ['python3', '-u'], 'cmd': ['/workspace/' + key], 'env': env})
    pod = pod.get('pod', pod)
    row = {'id': pod['id'], 'variant': 'camera_hand_all', 'wandb_run_id': wandb_id,
           'wandb_url': f'https://wandb.ai/jonathanhliu21-ucla/trimi-training/runs/{wandb_id}',
           'status_key': f'experiments/{campaign}/runs/camera_hand_all/status.json',
           'checkpoint_path': f'/workspace/experiments/{campaign}/runs/camera_hand_all/checkpoints/latest.ckpt',
           'deadline_timestamp': now + 86400, 'gpu_cost_per_hour': selection['price']}
    deployment = {'campaign': campaign, 'volume_id': volume_id, 'data_center': dc,
                  'launch_complete': True, 'pods': [row]}
    save(local / 'deployment.json', deployment)
    start_watchdog(local / 'deployment.json')
    live = mcp.call('get-pod', id=row['id'])
    live = live.get('pod', live)
    assert live['id'] == row['id']
    target.upload_file(str(local / 'deployment.json'), volume_id, f'experiments/{campaign}/deployment.json')
    return {'action': 'fallback_created', 'deployment': str(local / 'deployment.json'),
            'pod_id': row['id'], **selection, 'volume_id': volume_id,
            'wandb_url': row['wandb_url'], 'checkpoint_path': row['checkpoint_path'],
            'runtime_verification': 'pending'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--read-only', action='store_true')
    args = parser.parse_args()
    keys = dotenv_values(CREDENTIALS)
    mcp = MCP(keys['RUNPOD_SHUTDOWN_KEY'])
    plan = json.loads((ORIGINAL / 'followup/schedule.json').read_text())
    if args.read_only:
        print(json.dumps({'queue_result': follow_queue(mcp), 'candidates': candidates(mcp, plan['max_gpu_hourly_price'])}, indent=2))
        return
    assert time.time() >= plan['check_timestamp'], 'Refuse to switch before the scheduled check'
    result_path = ORIGINAL / 'followup/result.json'
    original = follow_queue(mcp)
    if original:
        save(result_path, original)
        verify(keys, original)
        return
    progress = ORIGINAL / 'followup/progress.json'
    journal = json.loads(progress.read_text()) if progress.exists() else {
        'campaign': 'color_palm_camera_hand_urgent_' + time.strftime('%Y%m%d_%H%M%S', time.gmtime()),
        'volumes': {}, 'queue_cancelled': False}
    save(progress, journal)
    until = plan['check_timestamp'] + 86400
    attempt = 0
    while time.time() < until:
        original = follow_queue(mcp)
        if original:
            save(result_path, original)
            verify(keys, original)
            return
        available = candidates(mcp, plan['max_gpu_hourly_price'])
        if len(journal['volumes']) >= 2:
            available = [x for x in available if x['data_center'] == 'EUR-IS-1' or x['data_center'] in journal['volumes']]
        if not available:
            save(result_path, {'action': 'waiting_for_comparable_gpu', 'checked_at': time.time()})
            time.sleep(60)
            continue
        if not journal['queue_cancelled']:
            queued = json.loads((ORIGINAL / 'deployment.json').read_text())['pods'][0]['queue_id']
            mcp.call('delete-queued-pod', id=queued)
            original = follow_queue(mcp)
            if original:
                save(result_path, original)
                verify(keys, original)
                return
            assert mcp.call('get-queued-pod', id=queued)['status'] in ('CANCELLED', 'EXPIRED', 'FAILED')
            journal['queue_cancelled'] = True
            save(progress, journal)
        # Stage only one volume at a time; rescan stock after copying data.
        selected = next((x for x in available if x['data_center'] in journal['volumes']), available[0])
        volume = stage(keys, mcp, selected, journal)
        fresh = candidates(mcp, plan['max_gpu_hourly_price'])
        selected = next((x for x in fresh if x['data_center'] == selected['data_center']), None)
        if selected is None:
            time.sleep(60)
            continue
        attempt += 1
        # Avoid replaying a potentially committed create after an ambiguous network error.
        try:
            result = launch(keys, mcp, selected, volume, journal['campaign'] + '_' + str(attempt))
        except RunpodError as error:
            if error.capacity:
                save(result_path, {'action': 'retrying_after_capacity_race', 'checked_at': time.time()})
                time.sleep(30)
                continue
            save(result_path, {'action': 'fallback_failed', 'error_type': type(error).__name__,
                               'message': 'Inspect progress and deployment before retrying a create.'})
            raise
        save(result_path, result)
        for region, unused in journal['volumes'].items():
            if unused != volume:
                try:
                    mcp.call('delete-network-volume', id=unused)
                    listed = mcp.pages('list-network-volumes', 'networkVolumes', limit=100)
                    assert all(item['id'] != unused for item in listed)
                except Exception as error:
                    print('Unused volume cleanup pending:', unused, type(error).__name__, flush=True)
        verify(keys, result)
        return
    save(result_path, {'action': 'no_comparable_gpu_within_cutoff'})


if __name__ == '__main__':
    main()
