"""Fetch job metadata from the GitHub Actions API.

Exposes the failed step of a workflow job. The step dict carries 'started_at'
and 'completed_at', which log_fetcher uses to slice the job log down to just
that step's output.
"""
import requests
from dotenv import load_dotenv

load_dotenv()


def get_failed_step(job_id: str, owner: str, repo: str, token: str) -> dict:
    """Return the single failed step of a workflow job.

    Calls GET /repos/{owner}/{repo}/actions/jobs/{job_id} and filters the
    'steps' array for conclusion == 'failure'. The whole step dict is returned
    rather than just its name, because downstream log slicing keys off the
    step's 'started_at' / 'completed_at' timestamps.

    Args:
        job_id: GitHub Actions job ID.
        owner: Repository owner (user or org).
        repo: Repository name.
        token: GitHub token. Job metadata is readable unauthenticated on public
            repos, but the log endpoint is not, so a token is required anyway.

    Returns:
        The failed step's metadata dict: name, status, conclusion, number,
        started_at, completed_at.

    Raises:
        ValueError: If the job has no failed step, or more than one. Multiple
            failures are ambiguous, since which step caused the failure is a
            judgement the caller has to make, so this refuses to guess.
        requests.HTTPError: On a non-2xx response.
    """
    custom_headers = {'Authorization': f'Bearer {token}', 'Accept': 'application/vnd.github+json',
                      'X-GitHub-Api-Version': '2022-11-28'}
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

    return failed_steps[0]
