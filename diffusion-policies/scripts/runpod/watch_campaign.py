"""Independent local fallback: terminate only pods in this launch manifest."""
import argparse
import json
import math
import time
from pathlib import Path

import boto3
import requests
from botocore.config import Config
from dotenv import dotenv_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--deployment', required=True)
    parser.add_argument('--credentials', required=True)
    args = parser.parse_args()
    keys = dotenv_values(args.credentials)
    initial = json.loads(Path(args.deployment).read_text())
    region = initial.get('data_center', 'EUR-IS-1')
    s3 = boto3.client('s3', endpoint_url=f'https://s3api-{region.lower()}.runpod.io', region_name=region,
                      aws_access_key_id=keys['RUNPOD_S3_API_KEY_USER'],
                      aws_secret_access_key=keys['RUNPOD_S3_API_KEY_SECRET'],
                      config=Config(connect_timeout=10, read_timeout=20, retries={'max_attempts': 1},
                                    request_checksum_calculation='when_required', response_checksum_validation='when_required'))
    headers = {'Authorization': 'Bearer ' + keys['RUNPOD_SHUTDOWN_KEY']}
    gone = set()
    clock_path = Path(args.deployment).with_name('watchdog_clocks.json')
    clocks = json.loads(clock_path.read_text()) if clock_path.exists() else {}
    while True:
        deployment = json.loads(Path(args.deployment).read_text())
        for pod in deployment['pods']:
            if pod.get('queue_id') in gone:
                continue
            if not pod.get('id') and pod.get('queue_id'):
                try:
                    queued = requests.get('https://api.runpod.io/v2/queued-pods/' + pod['queue_id'],
                                          headers=headers, timeout=20)
                    queued.raise_for_status()
                    queued = queued.json()
                    if queued.get('podId'):
                        pod['id'] = queued['podId']
                        pod['queue_status'] = queued['status']
                        Path(args.deployment).write_text(json.dumps(deployment, indent=2))
                        print('Queue deployed:', pod['queue_id'], pod['id'], flush=True)
                    elif queued.get('status') in ('EXPIRED', 'FAILED', 'CANCELLED'):
                        gone.add(pod['queue_id'])
                        print('Queue ended:', pod['queue_id'], queued['status'], flush=True)
                        continue
                    else:
                        continue
                except Exception as error:
                    print('Queue read retry:', type(error).__name__, flush=True)
                    continue
            if pod['id'] in gone:
                continue
            url = 'https://api.runpod.io/v2/pods/' + pod['id']
            reason = None
            try:
                response = requests.get(url, headers=headers, timeout=20)
                if response.status_code == 404:
                    gone.add(pod['id'])
                    print('Termination verified:', pod['id'], flush=True)
                    continue
                response.raise_for_status()
                live = response.json()
                live = live.get('pod', live)
                phase = str(live.get('status', '')).upper()
                if phase in ('EXPIRED', 'FAILED', 'EXITED', 'STOPPED'):
                    reason = 'pod ' + phase.lower()
                if pod.get('runtime_from_start'):
                    # Anchor to allocation, conservatively before bootstrap startup.
                    if phase not in ('QUEUED', 'PENDING', 'EXPIRED') and pod['id'] not in clocks:
                        clocks[pod['id']] = time.time()
                        clock_path.write_text(json.dumps(clocks, indent=2))
                    if pod['id'] in clocks and time.time() >= clocks[pod['id']] + 86400:
                        reason = '24-hour runtime cutoff'
                    elif pod['id'] not in clocks and time.time() >= pod['queue_expires_timestamp']:
                        reason = 'queue expired'
                try:
                    obj = s3.get_object(Bucket=deployment['volume_id'], Key=pod['status_key'])
                    state = json.loads(obj['Body'].read())
                except s3.exceptions.NoSuchKey:
                    state = {}
                if state.get('pod_id') != pod['id'] and time.time() >= pod.get('startup_deadline_timestamp', math.inf):
                    # The container never started its bootstrap (e.g. image pull or mount failure).
                    reason = 'no bootstrap status before startup deadline'
                elif state.get('pod_id') == pod['id'] and state.get('phase') in ('completed', 'failed', 'timed_out'):
                    reason = state['phase']
                elif state.get('pod_id') == pod['id'] and time.time() - state['heartbeat'] > 900:
                    reason = 'heartbeat missing for 15 minutes'
            except Exception as error:
                print('Read retry:', pod['id'], type(error).__name__, flush=True)
            if time.time() >= pod['deadline_timestamp']:
                reason = 'absolute safety cutoff'
            if reason:
                try:
                    result = requests.delete(url, headers=headers, timeout=30)
                    print('Termination requested:', pod['id'], reason, result.status_code, flush=True)
                    if result.status_code == 404:
                        gone.add(pod['id'])
                except requests.RequestException as error:
                    print('Termination retry:', type(error).__name__, flush=True)
        if deployment.get('launch_complete') and len(gone) == len(deployment['pods']):
            return
        time.sleep(30)


if __name__ == '__main__':
    main()
