from datetime import datetime

import requests


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
    end_dt = datetime.fromisoformat(step['completed_at'])

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
