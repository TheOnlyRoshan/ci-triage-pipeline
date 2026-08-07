"""Fetch a job's log from GitHub Actions and slice it to one step.

The /logs endpoint returns the whole job's output — every step concatenated.
Isolating the failed step is done by timestamp range, not by '##[group]'
markers: a step's 'name' in job metadata differs from its log marker text
("Run tests" vs "##[group]Run pytest"), and one step can contain several
groups plus ungrouped lines, so there is no 1-to-1 mapping to match on.
"""
from datetime import datetime

import requests


def fetch_job_log(job_id: str, owner: str, repo: str, token: str) -> str:
    """Download the full log text for a workflow job.

    Args:
        job_id: GitHub Actions job ID.
        owner: Repository owner (user or org).
        repo: Repository name.
        token: GitHub token. Required even for public repos — unlike job
            metadata, the log endpoint always demands authentication.

    Returns:
        The job's complete log as text, one timestamped line per entry,
        covering every step of the job.

    Raises:
        requests.HTTPError: On a non-2xx response.
    """
    custom_headers = {'Authorization': f'Bearer {token}',
                      'X-GitHub-Api-Version': '2022-11-28'}
    url = f"https://api.github.com/repos/{owner}/{repo}/actions/jobs/{job_id}/logs"
    response = requests.get(url, headers=custom_headers, timeout=5)
    response.raise_for_status()
    # GitHub omits a charset, so requests falls back to ISO-8859-1 and mangles
    # non-ASCII characters (em-dashes in test comments become 'â').
    response.encoding = 'utf-8'
    return response.text


def extract_step_window(log_text: str, step: dict) -> str:
    """Slice a job log down to the lines emitted during one step.

    Each log line is prefixed with an ISO 8601 timestamp; each line is kept if
    its timestamp falls in [started_at, completed_at).

    The window is deliberately unpadded. GitHub reports step boundaries only to
    whole-second precision while log lines carry sub-second precision, so the
    final '##[error]' line often lands just after completed_at and is clipped.
    Padding by a second recovers it but also pulls in the entire post-job
    cleanup block (~17 lines of git config), which would crowd out the pytest
    failure in the tail window downstream. Precision is preferred over recall.

    Args:
        log_text: Full job log from fetch_job_log.
        step: Failed step dict from get_failed_step; 'started_at' and
            'completed_at' are read from it.

    Returns:
        The step's log lines joined by newlines, timestamps still attached
        (log_preprocessor strips them). Blank lines and lines whose leading
        token will not parse as a timestamp are dropped.
    """
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
