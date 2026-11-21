"""Local fallback watcher, detached from chat; requires this computer to stay on."""
import argparse
import json
import time
from pathlib import Path

import boto3
from botocore.config import Config
from dotenv import dotenv_values
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pod', required=True)
    parser.add_argument('--volume', required=True)
    parser.add_argument('--credentials', required=True)
    parser.add_argument('--hours', type=float, default=24)
    args = parser.parse_args()
    keys = dotenv_values(args.credentials)
    s3 = boto3.client('s3', endpoint_url='https://s3api-eur-is-1.runpod.io', region_name='EUR-IS-1',
                      aws_access_key_id=keys['RUNPOD_S3_API_KEY_USER'],
                      aws_secret_access_key=keys['RUNPOD_S3_API_KEY_SECRET'],
                      config=Config(request_checksum_calculation='when_required', response_checksum_validation='when_required'))
    headers = {'Authorization': 'Bearer ' + keys['RUNPOD_SHUTDOWN_KEY']}
    url = 'https://api.runpod.io/v2/pods/' + args.pod
    deadline = time.monotonic() + args.hours * 3600
    while True:
        reason = None
        try:
            response = requests.get(url, headers=headers, timeout=20)
            if response.status_code == 404:
                print('Pod termination verified:', args.pod, flush=True)
                return
            obj = s3.get_object(Bucket=args.volume, Key='runs/m3_screw_shortcut_dinov3_001/status.json')
            state = json.loads(obj['Body'].read())
            if state.get('pod_id') == args.pod and state.get('phase') in ('completed', 'failed', 'smoke_completed'):
                reason = state['phase']
        except Exception as error:
            print('Read retry:', type(error).__name__, flush=True)
        if time.monotonic() >= deadline:
            reason = 'maximum runtime'
        if reason:
            try:
                result = requests.delete(url, headers=headers, timeout=30)
                print('Termination requested:', reason, result.status_code, flush=True)
                if result.status_code == 404:
                    return
            except requests.RequestException as error:
                print('Termination retry:', type(error).__name__, flush=True)
        time.sleep(30)


if __name__ == '__main__':
    main()
