#!/usr/bin/env python3
import argparse
import sys

from ci_common import CiError, cancel_older_workflow_runs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--github-token", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--workflow-name", required=True)
    parser.add_argument("--branch", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    cancel_older_workflow_runs(
        args.repository,
        args.github_token,
        args.workflow_name,
        args.branch,
        args.run_id,
    )


if __name__ == "__main__":
    try:
        main()
    except CiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
