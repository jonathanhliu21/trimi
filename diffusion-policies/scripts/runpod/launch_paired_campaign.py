"""Stage, launch and verify a paired DINOv3 shortcut-policy campaign on Runpod.

One pod trains on camera_0 only, the other on camera_0 + the 13-joint hand_pose_abs, with
identical hyperparameters and train/validation splits. Credentials are read from ../.env and
never printed. Each pod deletes itself on completion or failure (color_palm_bootstrap.py), runs
an in-pod watchdog, and is also watched by a local systemd unit (watch_campaign.py); all three
enforce a 24-hour ceiling. Nothing here deletes the network volume or any checkpoint.

    python launch_paired_campaign.py stage  --campaign NAME --dataset /path/to/dataset
    python launch_paired_campaign.py launch --campaign NAME
    python launch_paired_campaign.py verify --campaign NAME
"""
import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time

from boto3.s3.transfer import TransferConfig
from botocore.exceptions import ClientError
from dotenv import dotenv_values
import numpy as np
import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from urgent_followup import MCP, RunpodError, s3, save  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
CREDENTIALS = ROOT.parent / '.env'
VOLUME_ID, VOLUME_NAME, DC = '8zlot4olf9', 'trimi-m3-dinov3-training', 'EUR-IS-1'
IMAGE = 'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404'
# Full (non-MIG) 96 GB RTX PRO 6000 cards, in order of preference.
GPUS = ['NVIDIA RTX PRO 6000 Blackwell Server Edition',
        'NVIDIA RTX PRO 6000 Blackwell Workstation Edition']
ENTITY, PROJECT = 'jonathanhliu21-ucla', 'trimi-training'
VARIANTS = {'camera': ['camera_0'], 'camera_hand': ['camera_0', 'hand_pose_abs']}
TASK = ROOT / 'shortcut_policy/config/tasks/trimi_image_hand_cond.yaml'
LOADER_DIRS = (('camera/mp4_files', '.mp4'), ('processed_encoder_t', '.npy'),
               ('absolute_action_t', '.npy'), ('syncs', '.npz'))
SOURCE_PATHS = ('diffusion_policy', 'shared', 'shortcut_policy', 'scripts',
                'requirements-dinov3.txt', 'requirements-train.txt', 'setup.py')
# Already on the volume and checksum-verified by every bootstrap.
SHARED_ARTIFACTS = [
    {'key': 'experiments/palm_multi_dinov3_20261124_01/staging/dinov3.tar.gz', 'size': 13351085,
     'sha256': '4fc02663917638f903bb7ce805d26bac9d4f7915da60629e2b0686ba07a5c556'},
    {'key': 'models/dinov3_vits16_pretrain_lvd1689m-08c60483.pth', 'size': 86531063,
     'sha256': '08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d'}]
# User-approved removal (2026-11-25): staged copy of an older dataset whose local copy
# data/color_palm_dinov3_20261124_01/dataset.tar has the same size and sha256.
OLD_ARCHIVE = {'key': 'experiments/color_palm_dinov3_20261124_01/staging/dataset.tar', 'size': 5681879040,
               'sha256': '4b8b8f9db3800604c20199cc2f55832550a464bf58b7a4ee3d6e4c26b2156807',
               'local': ROOT / 'data/color_palm_dinov3_20261124_01/dataset.tar'}
MAX_RUNTIME = 86400
STARTUP_GRACE = 2700
# Each run peaks at two checkpoints (old latest + atomic temp file) of ~1.5 GB.
MIN_FREE_GIB = 4.5
KEEP_ARCHIVE_FREE_GIB = 7.0
NAME = re.compile(r'^[A-Za-z0-9_-]+$')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def keys():
    values = dotenv_values(CREDENTIALS)
    missing = [k for k in ('WANDB_API_KEY', 'RUNPOD_SHUTDOWN_KEY', 'RUNPOD_S3_API_KEY_USER',
                           'RUNPOD_S3_API_KEY_SECRET') if not values.get(k)]
    if missing:
        sys.exit('Missing credentials in ' + str(CREDENTIALS) + ': ' + ', '.join(missing))
    return values


def exists(client, key):
    try:
        return client.head_object(Bucket=VOLUME_ID, Key=key)['ContentLength']
    except ClientError as error:
        if error.response.get('Error', {}).get('Code') in ('404', 'NoSuchKey', 'NotFound'):
            return None
        raise


def get_val_mask():
    """The loader's own split function, loaded verbatim (sampler.py also imports numba)."""
    tree = ast.parse((ROOT / 'shared/utils/sampler.py').read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_val_mask')
    namespace = {'np': np}
    exec(compile(ast.Module([node], type_ignores=[]), 'sampler.py', 'exec'), namespace)
    return namespace['get_val_mask']


def dataset_report(dataset):
    """Mirror MidasImageDataset's episode filtering and split without decoding video."""
    task = yaml.safe_load(TASK.read_text())['dataset']
    stride, first_cut = int(task['sync_downsample_stride']), int(task['first_cut'])
    val_ratio, seed = float(task['val_ratio']), int(task['seed'])
    files = {d: sorted(f for f in os.listdir(dataset / d) if f.endswith(ext)) for d, ext in LOADER_DIRS}
    counts = {d: len(v) for d, v in files.items()}
    if len(set(counts.values())) != 1:
        raise RuntimeError(f'Loader file count mismatch: {counts}')
    kept, steps = [], 0
    for i, video in enumerate(files['camera/mp4_files']):
        hand = np.load(dataset / 'processed_encoder_t' / files['processed_encoder_t'][i], mmap_mode='r')
        eef = np.load(dataset / 'absolute_action_t' / files['absolute_action_t'][i], mmap_mode='r')
        sync = np.load(dataset / 'syncs' / files['syncs'][i])
        if hand.ndim != 2 or hand.shape[1] != 13 or eef.ndim != 2 or eef.shape[1] != 6:
            raise RuntimeError(f'Episode {i}: hand {hand.shape} / eef {eef.shape} would abort the loader')
        lengths = {len(sync[k]) for k in ('frame_idx', 'encoder_idx', 'pose_idx')}
        if len(lengths) != 1 or lengths.pop() <= first_cut:
            continue
        n = len(sync['frame_idx'][first_cut:][::stride])
        if n == 0 or len(hand) == 0 or len(eef) == 0 or (dataset / 'camera/mp4_files' / video).stat().st_size == 0:
            continue
        kept.append(Path(video).stem)
        steps += n
    val = get_val_mask()(n_episodes=len(kept), val_ratio=val_ratio, seed=seed)
    return {'dataset_path': str(dataset), 'file_counts': counts, 'episodes': kept,
            'n_episodes': len(kept), f'steps_after_stride_{stride}': steps,
            'validation_indices': np.flatnonzero(val).tolist(),
            'train_all_indices': np.flatnonzero(~val).tolist(), 'seed': seed, 'val_ratio': val_ratio}


def build_dataset_tar(dataset, target):
    with tarfile.open(target, 'w') as archive:
        for directory, ext in LOADER_DIRS:
            archive.add(dataset / directory.split('/')[0], arcname=f'{dataset.name}/{directory.split("/")[0]}',
                        recursive=False)
            if '/' in directory:
                archive.add(dataset / directory, arcname=f'{dataset.name}/{directory}', recursive=False)
            for name in sorted(os.listdir(dataset / directory)):
                if name.endswith(ext):
                    archive.add(dataset / directory / name, arcname=f'{dataset.name}/{directory}/{name}')


def build_source_tar(target):
    listed = subprocess.run(['git', 'ls-files', '-co', '--exclude-standard', '-z', '--', *SOURCE_PATHS],
                            cwd=ROOT, check=True, capture_output=True).stdout.decode().split('\0')
    names = sorted(n for n in listed if n and '__pycache__' not in n and not n.endswith('.pyc')
                   and (ROOT / n).is_file())
    with tarfile.open(target, 'w:gz') as archive:
        for name in names:
            archive.add(ROOT / name, arcname=name)
    return len(names)


def upload(client, path, key, reuse=False):
    size = Path(path).stat().st_size
    if reuse and exists(client, key) == size:
        print('Already staged:', key, flush=True)
        return
    for attempt in range(3):
        try:
            client.upload_file(str(path), VOLUME_ID, key, Config=TransferConfig(
                multipart_threshold=64 * 1024**2, multipart_chunksize=64 * 1024**2, max_concurrency=4))
        except Exception as error:
            # Multipart completion can time out after the object was committed.
            message = str(error)
            for secret in dotenv_values(CREDENTIALS).values():
                if secret:
                    message = message.replace(secret, '***')
            print(f'Upload attempt {attempt + 1} returned {type(error).__name__}: {message[:600]}', flush=True)
        if exists(client, key) == size:
            break
        time.sleep(30)
    if exists(client, key) != size:
        raise RuntimeError('Staged size mismatch: ' + key)
    print(f'Staged {key} ({size / 1024**3:.2f} GiB)', flush=True)


def delete_old_archive(credentials, local):
    client = s3(credentials, DC)
    record = {'key': OLD_ARCHIVE['key'], 'size': OLD_ARCHIVE['size'], 'sha256': OLD_ARCHIVE['sha256'],
              'local_copy': str(OLD_ARCHIVE['local'])}
    remote = exists(client, OLD_ARCHIVE['key'])
    if remote is None:
        record['result'] = 'already absent'
    else:
        copy = OLD_ARCHIVE['local']
        if remote != OLD_ARCHIVE['size'] or copy.stat().st_size != OLD_ARCHIVE['size'] \
                or digest(copy) != OLD_ARCHIVE['sha256']:
            raise RuntimeError('Old archive no longer matches its verified local copy; not deleting')
        client.delete_object(Bucket=VOLUME_ID, Key=OLD_ARCHIVE['key'])
        if exists(client, OLD_ARCHIVE['key']) is not None:
            raise RuntimeError('Old archive still present after delete')
        record['result'] = 'deleted; absence verified by read-back'
    record['timestamp'] = time.time()
    save(local / 'cleanup.json', record)
    print('Old staged archive:', record['result'], flush=True)


def stage(args):
    if not args.dataset:
        sys.exit('ABORTED: no dataset path supplied. Re-run with --dataset /path/to/dataset_dir.')
    dataset = Path(args.dataset).expanduser().resolve()
    if not dataset.is_dir():
        sys.exit(f'ABORTED: dataset path {dataset} does not exist. Supply the dataset directory with --dataset.')
    if not NAME.match(args.campaign) or not NAME.match(dataset.name):
        sys.exit('Campaign and dataset directory names must be simple identifiers')
    credentials = keys()
    local = ROOT / 'data' / args.campaign
    local.mkdir(parents=True, exist_ok=True)
    client = s3(credentials, DC)
    prefix = f'experiments/{args.campaign}/'
    if client.list_objects_v2(Bucket=VOLUME_ID, Prefix=prefix + 'runs/', MaxKeys=1).get('KeyCount'):
        sys.exit(f'Campaign {args.campaign} already has run output on the volume; choose a new name')
    for artifact in SHARED_ARTIFACTS:
        if exists(client, artifact['key']) != artifact['size']:
            raise RuntimeError('Shared artifact missing on volume: ' + artifact['key'])

    report = dataset_report(dataset)
    save(local / 'dataset_report.json', report)
    print(f"Dataset {dataset.name}: {report['n_episodes']} episodes, {len(report['train_all_indices'])} train /"
          f" {len(report['validation_indices'])} validation, {report['file_counts']}", flush=True)
    if report['n_episodes'] != report['file_counts']['syncs']:
        raise RuntimeError('Loader would drop episodes; inspect before training')

    archives = {'dataset.tar': local / 'dataset.tar', 'source.tar.gz': local / 'source.tar.gz'}
    if not archives['dataset.tar'].exists():
        build_dataset_tar(dataset, archives['dataset.tar'].with_suffix('.partial'))
        archives['dataset.tar'].with_suffix('.partial').replace(archives['dataset.tar'])
    print('Source files archived:', build_source_tar(archives['source.tar.gz']), flush=True)
    artifacts = [{'key': prefix + 'staging/' + name, 'size': path.stat().st_size, 'sha256': digest(path)}
                 for name, path in archives.items()] + SHARED_ARTIFACTS
    manifest = {'campaign': args.campaign, 'dataset': dataset.name, 'dataset_source': str(dataset),
                'artifacts': artifacts,
                'credential_sha256': {k: hashlib.sha256(credentials[k].encode()).hexdigest()
                                      for k in ('WANDB_API_KEY', 'RUNPOD_SHUTDOWN_KEY')}}
    save(local / 'manifest.json', manifest)

    if args.delete_old_archive:
        delete_old_archive(credentials, local)
    for name, path in archives.items():
        upload(client, path, prefix + 'staging/' + name, reuse=name == 'dataset.tar')
    for name in ('manifest.json', 'dataset_report.json'):
        client.upload_file(str(local / name), VOLUME_ID, prefix + 'staging/' + name)
    bootstrap = ROOT / 'scripts/runpod/color_palm_bootstrap.py'
    client.upload_file(str(bootstrap), VOLUME_ID, prefix + 'staging/bootstrap.py')
    if client.get_object(Bucket=VOLUME_ID, Key=prefix + 'staging/bootstrap.py')['Body'].read() != bootstrap.read_bytes():
        raise RuntimeError('Bootstrap read-back mismatch')
    save(local / 'staged.json', {'campaign': args.campaign, 'timestamp': time.time(),
                                 'bootstrap_sha256': digest(bootstrap)})
    print('Staging complete:', prefix + 'staging/', flush=True)


def sanitized(pod):
    pod = pod.get('pod', pod)
    out = {k: pod.get(k) for k in ('id', 'name', 'desiredStatus', 'costPerHr', 'adjustedCostPerHr',
                                    'image', 'imageName', 'networkVolumeId', 'dataCenterId', 'gpuCount')}
    gpu = pod.get('gpu') or {}
    out['gpu'] = {k: gpu.get(k) for k in ('id', 'count', 'displayName')} if isinstance(gpu, dict) else gpu
    machine = pod.get('machine') or {}
    out['machine'] = {k: machine.get(k) for k in ('gpuTypeId', 'dataCenterId', 'location')} if isinstance(machine, dict) else None
    return {k: v for k, v in out.items() if v not in (None, {}, [])}


def stock(mcp):
    gpus = mcp.pages('list-gpu-types', 'gpus', include=['AVAILABILITY'], product=['POD'],
                     cloud='SECURE', minCudaVersion='12.8', limit=100)
    result = []
    for gpu_id in GPUS:
        row = next((g for g in gpus if g['id'] == gpu_id), None)
        if row and row.get('memory', 0) >= 90:
            for dc in row.get('dataCenters', []):
                if dc['id'] == DC and dc.get('availability') in ('LOW', 'MEDIUM', 'HIGH'):
                    result.append({'gpu': gpu_id, 'availability': dc['availability'],
                                   'price': row.get('price', {}).get('secure')})
    return result


def find_pod(mcp, name):
    listed = mcp.call('list-pods')
    rows = listed.get('pods', listed if isinstance(listed, list) else [])
    return next((p for p in rows if p.get('name') == name), None)


def launch(args):
    credentials = keys()
    local = ROOT / 'data' / args.campaign
    if not (local / 'staged.json').exists():
        sys.exit('Stage the campaign first')
    manifest = json.loads((local / 'manifest.json').read_text())
    mcp = MCP(credentials['RUNPOD_SHUTDOWN_KEY'])
    volume = mcp.call('get-network-volume', id=VOLUME_ID)
    volume = volume.get('networkVolume', volume)
    assert volume['name'] == VOLUME_NAME and volume['dataCenter'] == DC, 'Unexpected network volume'
    path = local / 'deployment.json'
    deployment = json.loads(path.read_text()) if path.exists() else {
        'campaign': args.campaign, 'volume_id': VOLUME_ID, 'volume_name': VOLUME_NAME, 'data_center': DC,
        'dataset': manifest['dataset'], 'image': IMAGE, 'max_runtime_hours': 24, 'pods': [],
        'launch_complete': False}
    mcp.call('describe-tool', name='create-pod')
    try:
        create_pods(args, credentials, manifest, mcp, deployment, path)
    finally:
        deployment['launch_complete'] = True
        save(path, deployment)
        if deployment['pods']:
            s3(credentials, DC).upload_file(str(path), VOLUME_ID, f'experiments/{args.campaign}/deployment.json')
            ensure_watchdog(args.campaign, path)


def ensure_watchdog(campaign, path):
    unit = 'trimi-watch-' + campaign.replace('_', '-')
    state = lambda: subprocess.run(['systemctl', '--user', 'is-active', unit], capture_output=True, text=True).stdout.strip()
    if state() != 'active':
        subprocess.run(['systemd-run', '--user', '--collect', '--unit=' + unit,
                        '--property=Restart=on-failure', '--property=RestartSec=15',
                        sys.executable, '-u', str(ROOT / 'scripts/runpod/watch_campaign.py'),
                        '--deployment', str(path), '--credentials', str(CREDENTIALS)], check=True)
    print('Local watchdog unit:', unit, state(), flush=True)


def create_pods(args, credentials, manifest, mcp, deployment, path):
    for variant in VARIANTS:
        if any(p['variant'] == variant for p in deployment['pods']):
            continue
        name = f'{args.campaign}_{variant}'
        existing = find_pod(mcp, name)
        wandb_id = 't' + os.urandom(4).hex()
        run = f'experiments/{args.campaign}/runs/{variant}'
        selection, pod = None, existing
        for attempt in range(30):
            if pod:
                break
            options = stock(mcp)
            print(f'{variant}: stock read in {DC}:', options, flush=True)
            for option in options:
                now = time.time()
                env = {k: credentials[k] for k in ('WANDB_API_KEY', 'RUNPOD_SHUTDOWN_KEY')}
                env.update(TRIMI_CAMPAIGN=args.campaign, TRIMI_DATASET_NAME=manifest['dataset'],
                           TRIMI_OBSERVATION_MODE=variant, TRIMI_VARIANT=variant,
                           TRIMI_MAX_TRAIN_EPISODES='null', TRIMI_CHECKPOINT_TOPK='0',
                           TRIMI_KEEP_SMOKE_CHECKPOINTS='0', TRIMI_MIN_FREE_GIB=str(MIN_FREE_GIB),
                           TRIMI_WANDB_RUN_ID=wandb_id,
                           TRIMI_BOOTSTRAP=f'/workspace/experiments/{args.campaign}/staging/bootstrap.py',
                           TRIMI_DEADLINE_TIMESTAMP=str(now + MAX_RUNTIME), MAX_RUNTIME_SECONDS=str(MAX_RUNTIME),
                           TRIMI_RUNTIME_FROM_START='1', TRIMI_EXPECTED_GPU_FAMILY='RTX PRO 6000',
                           TRIMI_MIN_GPU_MEMORY_GIB='90')
                print(f"{variant}: creating 1x {option['gpu']} in {DC} at {option['price']} USD/hour", flush=True)
                try:
                    pod = mcp.call('create-pod', body={
                        'name': name, 'image': IMAGE, 'cloud': 'SECURE',
                        'gpu': {'id': option['gpu'], 'count': 1, 'minCudaVersion': '12.8'},
                        'disk': 80, 'dataCenterIds': [DC],
                        'mounts': {'network': [{'volumeId': VOLUME_ID, 'path': '/workspace'}]},
                        'ports': [], 'entrypoint': ['python3', '-u'],
                        'cmd': [env['TRIMI_BOOTSTRAP']], 'env': env})
                    selection = dict(option, created_timestamp=now)
                    break
                except RunpodError as error:
                    if not error.capacity:
                        raise
                    print(f"{variant}: capacity taken for {option['gpu']}", flush=True)
                except requests.RequestException as error:
                    # Never replay an ambiguous create: look for the pod by its unique name.
                    print(f'{variant}: create returned {type(error).__name__}; checking by name', flush=True)
                    time.sleep(20)
                    pod = find_pod(mcp, name)
                    if pod:
                        selection = dict(option, created_timestamp=now)
                        break
            if not pod:
                time.sleep(60)
        if not pod:
            raise RuntimeError(f'No full RTX PRO 6000 could be placed in {DC} for {variant}')
        if selection is None:
            raise RuntimeError(f'{name} already existed before this launch; inspect it before continuing')
        pod_id = pod.get('pod', pod)['id']
        record = {
            'id': pod_id, 'variant': variant, 'mode': variant, 'observations': VARIANTS[variant],
            'gpu': selection['gpu'], 'gpu_availability_at_create': selection['availability'],
            'catalog_price_per_hour': selection['price'],
            'wandb_run_id': wandb_id, 'wandb_run_name': name,
            'wandb_url': f'https://wandb.ai/{ENTITY}/{PROJECT}/runs/{wandb_id}',
            'run_directory': f'/workspace/{run}', 'status_key': f'{run}/status.json',
            'checkpoint_path': f'/workspace/{run}/checkpoints/latest.ckpt',
            'created_timestamp': selection['created_timestamp'],
            'startup_deadline_timestamp': selection['created_timestamp'] + STARTUP_GRACE,
            'deadline_timestamp': selection['created_timestamp'] + MAX_RUNTIME}
        # Record and watch the pod before anything else can fail.
        deployment['pods'].append(record)
        save(path, deployment)
        ensure_watchdog(args.campaign, path)
        live = mcp.call('get-pod', id=pod_id)
        live = live.get('pod', live)
        print(f'{variant}: read-back', json.dumps(sanitized(live)), 'fields:', sorted(live), flush=True)
        assert live['id'] == pod_id
        record['cost_per_hour'] = live.get('costPerHr', live.get('adjustedCostPerHr'))
        save(path, deployment)


def strip_hand(value):
    if isinstance(value, dict):
        return {k: strip_hand(v) for k, v in value.items() if k != 'hand_pose_abs'}
    if isinstance(value, list):
        return [strip_hand(v) for v in value]
    return value


def differences(a, b, path=''):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(set(a) | set(b)):
            out += differences(a.get(key), b.get(key), f'{path}.{key}' if path else str(key))
        return out
    return [] if a == b else [path]


# Fields that legitimately differ between the two runs.
PER_RUN = {'logging.name', 'logging.id', 'multi_run.run_dir', 'multi_run.wandb_name_base',
           'tasks.env_runner.output_dir'}


def verify(args):
    credentials = keys()
    local = ROOT / 'data' / args.campaign
    deployment = json.loads((local / 'deployment.json').read_text())
    expected = json.loads((local / 'dataset_report.json').read_text())
    client = s3(credentials, DC)
    pods = {p['variant']: p for p in deployment['pods']}
    archive_key = f'experiments/{args.campaign}/staging/dataset.tar'
    stop = time.time() + args.minutes * 60

    def read(pod, name):
        key = pod['status_key'].rsplit('/', 1)[0] + '/' + name
        return client.get_object(Bucket=VOLUME_ID, Key=key)['Body'].read().decode()

    while True:
        result = {'checked_at': time.time(), 'runs': {}, 'successful_launch': False}
        states, configs, splits = {}, {}, {}
        for variant, pod in pods.items():
            run = {'pod_id': pod['id'], 'wandb_url': pod['wandb_url'], 'checkpoint_path': pod['checkpoint_path']}
            result['runs'][variant] = run
            try:
                state = json.loads(read(pod, 'status.json'))
                states[variant] = state
                run['phase'] = state.get('phase')
                if state.get('phase') in ('failed', 'timed_out'):
                    run['error'] = f"{state.get('error_type')}: {state.get('error')}"
                    continue
                run['smoke_verified'] = all(state.get(k) for k in (
                    'backbone_verified', 'checkpoint_reload_verified', 'observations_verified', 'gpu_verified'))
                run['reload_log_verified'] = 'reload verified' in read(pod, 'reload.log')
                run['gpu'] = json.loads(read(pod, 'gpu_verification.json').strip().splitlines()[-1])
                config = yaml.safe_load(read(pod, 'resolved_config.yaml'))
                split = json.loads(read(pod, 'split_verification.json'))
                configs[variant], splits[variant] = config, split
                obs = config['shape_meta']['obs']
                backbone = config['policy']['obs_encoder']['vision_backbone']
                checks = {
                    'observations': sorted(obs) == sorted(VARIANTS[variant]) == sorted(split['observations']),
                    'hand_13_joints': 'hand_pose_abs' not in obs or (obs['hand_pose_abs']['shape'] == [13]
                                                                    and split['observations']['hand_pose_abs'][-1] == 13),
                    'dinov3_trainable': backbone['name'] == 'dinov3_vits16' and backbone['trainable'] is True
                                        and config['training']['freeze_encoder'] is False,
                    'online_wandb_config': config['logging']['mode'] == 'online' and config['logging']['project'] == PROJECT
                                           and config['logging']['id'] == pod['wandb_run_id'],
                    'split_matches_report': split['episodes'] == expected['n_episodes']
                                            and split['train'] == expected['train_all_indices']
                                            and split['validation'] == expected['validation_indices'],
                    'epochs_500': config['training']['num_epochs'] == 500,
                }
                run['config_checks'] = checks
                run['main_backbone_verified'] = 'BACKBONE_VERIFIED' in read(pod, 'train.log')
                size = exists(client, pod['status_key'].rsplit('/', 1)[0] + '/checkpoints/latest.ckpt')
                run['main_checkpoint_bytes'] = size
                run['main_checkpoint_verified'] = bool(size and size > 1_000_000_000)
                query = ('query($name:String!){project(name:"%s",entityName:"%s"){run(name:$name)'
                         '{state displayName summaryMetrics history(samples:3)}}}' % (PROJECT, ENTITY))
                response = requests.post('https://api.wandb.ai/graphql', auth=('api', credentials['WANDB_API_KEY']),
                                         json={'query': query, 'variables': {'name': pod['wandb_run_id']}}, timeout=30)
                response.raise_for_status()
                wandb_run = ((response.json().get('data') or {}).get('project') or {}).get('run')
                if wandb_run:
                    summary = wandb_run['summaryMetrics']
                    summary = json.loads(summary) if isinstance(summary, str) else summary
                    run['wandb_state'], run['wandb_name'] = wandb_run['state'], wandb_run['displayName']
                    run['online_summary'] = {k: summary.get(k) for k in ('epoch', 'global_step', 'train_loss', 'val_loss', 'lr')}
                    run['online_metrics_verified'] = (wandb_run['state'] == 'running' and bool(wandb_run['history'])
                                                      and math.isfinite(float(summary.get('train_loss', 'nan'))))
            except Exception as error:
                run.setdefault('pending', type(error).__name__ + (': ' + str(error)[:160] if isinstance(error, (AssertionError, KeyError)) else ''))
        # Free the staged dataset archive only when space is tight and both pods have extracted it.
        if all(states.get(v, {}).get('phase') not in (None, 'verifying', 'extracting') for v in pods) \
                and exists(client, archive_key) is not None:
            free = []
            for pod in pods.values():
                try:
                    found = re.findall(r'Checkpoint volume free space: ([0-9.]+) GiB', read(pod, 'bootstrap.log'))
                    free += [float(x) for x in found[-1:]]
                except Exception:
                    free = []
                    break
            if len(free) == len(pods) and min(free) < KEEP_ARCHIVE_FREE_GIB:
                client.delete_object(Bucket=VOLUME_ID, Key=archive_key)
                result['staged_dataset_archive'] = ('deleted after both pods extracted it (free space '
                                                    f'{min(free):.2f} GiB); local copy kept')
                print(result['staged_dataset_archive'], flush=True)
        if len(configs) == len(pods):
            a, b = (strip_hand(configs[v]) for v in pods)
            diff = [d for d in differences(a, b) if d not in PER_RUN]
            result['identical_hyperparameters'] = not diff
            if diff:
                result['hyperparameter_differences'] = diff
            result['identical_splits'] = len({json.dumps([splits[v]['train'], splits[v]['validation']]) for v in pods}) == 1
        runs_ok = all(r.get('smoke_verified') and r.get('reload_log_verified') and r.get('main_backbone_verified')
                      and r.get('main_checkpoint_verified') and r.get('online_metrics_verified')
                      and all((r.get('config_checks') or {'x': False}).values()) for r in result['runs'].values())
        result['successful_launch'] = bool(runs_ok and result.get('identical_hyperparameters') and result.get('identical_splits'))
        save(local / 'verification.json', result)
        try:
            client.upload_file(str(local / 'verification.json'), VOLUME_ID, f'experiments/{args.campaign}/verification.json')
        except Exception:
            pass
        print(time.strftime('%H:%M:%S'), json.dumps({v: {k: r.get(k) for k in ('phase', 'pending', 'error')}
                                                     for v, r in result['runs'].items()}), flush=True)
        failed = [v for v, s in states.items() if s.get('phase') in ('failed', 'timed_out')]
        if result['successful_launch'] or failed or time.time() > stop:
            print(json.dumps(result, indent=2), flush=True)
            return 0 if result['successful_launch'] else 1
        time.sleep(60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['stage', 'launch', 'verify'])
    parser.add_argument('--campaign', required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--delete-old-archive', action='store_true')
    parser.add_argument('--minutes', type=float, default=120)
    args = parser.parse_args()
    if not NAME.match(args.campaign):
        sys.exit('Campaign must be a simple identifier')
    sys.exit({'stage': stage, 'launch': launch, 'verify': verify}[args.command](args))


if __name__ == '__main__':
    main()
