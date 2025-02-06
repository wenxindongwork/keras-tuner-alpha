"""
 Copyright 2025 Google LLC

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      https://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 """

from ray.job_submission import JobSubmissionClient, JobStatus
import time
import argparse
from pathlib import Path
import os

def wait_until_status(client, job_id, status_to_wait_for, timeout_seconds=60):
    start = time.time()
    while time.time() - start <= timeout_seconds:
        status = client.get_job_status(job_id)
        print(f"status: {status}")
        if status in status_to_wait_for:
            break
        time.sleep(1)


def setup_parser(parser):
    
    parser.description = (
        "Submit a distributed training job to a Ray cluster. "
        "This command requires a running Ray cluster - make sure to set up your "
        "cluster first using 'ray up' before running this command."
    )

    # Format the help text to preserve newlines
    parser.formatter_class = argparse.RawDescriptionHelpFormatter

    parser.add_argument(
        "job_file",
        type=str,
        help=(
            "Path to the Python script to run on the Ray cluster. "
            "Example: 'my_example/sft.py'"
        ),
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        default="http://localhost:8265",
        help=(
            "Address of the Ray cluster dashboard. If your have already forwarded the Ray Dashboard to "
            "your local machine, please use http://localhost:port. Otherwise, you can find the "
            "external_ip of your Ray Head node with `ray get-head-ip cluster.yaml` and use `http://ip:8265.`"
            "Default: http://localhost:8265"
        ),
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default="./",
        help=(
            "Directory containing the job script and its dependencies. "
            "All files in this directory will be uploaded to the Ray cluster. "
            "Default: current directory"
        ),
    )
    parser.add_argument(
        "--exclude-files",
        type=str,
        default="",
        help=(
            "Files/directories to exclude from upload to Ray cluster. "
            "Comma-separated list of paths relative to working-dir. "
            "Example: 'data/raw,*.pyc,__pycache__'"
        ),
    )
    parser.add_argument(
        "--pip-dependencies",
        type=str,
        default="",
        help=(
            "Additional Python packages to install on the Ray cluster. "
            "Comma-separated list of package names with optional versions. "
            "Example: 'tensorflow==2.13.0,pytorch>=2.0.0'"
            "Note: These may be installed permanently on the cluster."
        ),
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help=(
            "HuggingFace authentication token for accessing private models/datasets. "
            "Get your token from https://huggingface.co/settings/tokens"
        ),
    )


def validate_paths(job_file: str, working_dir: str) -> None:
    job_path = Path(os.path.join(working_dir, job_file))
    work_dir = Path(working_dir)

    if not job_path.exists():
        raise FileNotFoundError(f"Job file not found: {job_file}")

    if not work_dir.exists():
        raise FileNotFoundError(f"Working directory not found: {working_dir}")


def main(args=None):
    """Entry point for the 'kithara multihost' command."""
    if args is None:
        parser = argparse.ArgumentParser()
        setup_parser(parser)
        args = parser.parse_args()
    validate_paths(args.job_file, args.working_dir)

    client = JobSubmissionClient(args.ray_address)
    print("Initialized job client")

    # Environment variables
    env_vars = {"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    if args.hf_token:
        env_vars["HF_TOKEN"] = args.hf_token

    # Default excludes combined with user-provided excludes
    excludes = [
        "**/test_assets/",
        ".git/",
        "*.Dockerfile",
        *args.exclude_files.split(","),
    ]

    job_id = client.submit_job(
        entrypoint=f"python3.11 {args.job_file}",
        runtime_env={
            "working_dir": args.working_dir,
            "excludes": excludes,
            "pip": args.pip_dependencies.split(","),
            "env_vars": env_vars,
        },
    )

    print(f"Submitted job with {job_id=}")

    wait_until_status(
        client, job_id, {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED}
    )

    # See full logs in the Ray Dashboard
    logs = client.get_job_logs(job_id)
    print(logs)


# Attach setup_parser to main function
main.setup_parser = setup_parser

if __name__ == "__main__":
    main()
