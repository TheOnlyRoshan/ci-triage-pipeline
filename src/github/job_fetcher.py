from pathlib import Path

import requests
from dotenv import load_dotenv

from src import config_loader
from src.github.auth import get_github_token

load_dotenv()
def get_failed_step(job_id: str, owner: str, repo: str, token: str) -> str:
    custom_headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json', 'X-GitHub-Api-Version': '2022-11-28'}
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/jobs/{job_id}"
    response = requests.get(url, headers=custom_headers, timeout=5)
    response.raise_for_status()
    steps = response.json()['steps']
    failed_steps = [step for step in steps if step['conclusion'] == 'failure']

    if len(failed_steps) == 0:
        raise ValueError(f"No failed step found for job_id={job_id}")
    if len(failed_steps) > 1:
        names = [s['name'] for s in failed_steps]
        raise ValueError(f"Multiple failed steps found for job_id={job_id}: {names}")

    return failed_steps[0]['name']