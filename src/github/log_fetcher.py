from datetime import datetime, timedelta

import requests
from dotenv import load_dotenv

from src.config_loader import load_config, find_project_root
from src.github.auth import get_github_token
from src.github.job_fetcher import get_failed_step


def fetch_job_log(job_id: str, owner: str, repo: str, token: str) -> str:
    custom_headers = {'Authorization': f'Bearer {token}',
                      'X-GitHub-Api-Version': '2022-11-28'}
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/jobs/{job_id}/logs"
    response = requests.get(url, headers=custom_headers, timeout=5)
    response.raise_for_status()
    response.encoding = 'utf-8'
    return response.text


def extract_step_window(log_text: str, step: dict) -> str:
    start_dt = datetime.fromisoformat(step['started_at'])
    end_dt = datetime.fromisoformat(step['completed_at']) # timedelta(seconds=1)

    filtered_log_lines = []
    for line in log_text.splitlines():
        if not line.strip():
            continue
        timestamp_str = line.split(" ", 1)[0]
        try:
            line_dt = datetime.fromisoformat(timestamp_str)
        except ValueError:
            continue
        if start_dt <= line_dt < end_dt:
            filtered_log_lines.append(line)

    return "\n".join(filtered_log_lines)


load_dotenv()
config = load_config(find_project_root() / 'config.yaml')
token = get_github_token(config)
log = fetch_job_log(config.github.job_id, config.github.owner, config.github.repo, token)
step = get_failed_step(config.github.job_id, config.github.owner, config.github.repo, token)
# print(log)
print('The failed step log:', extract_step_window(log, step))
