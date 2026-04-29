#!/usr/bin/env python3
import argparse
import json
import shutil
import sys
from pathlib import Path

from ci_common import (
    CiError,
    add_args,
    add_jenkins_crumb,
    basic_headers,
    bool_text,
    cancel_superseded,
    choice,
    commit_status,
    copy_contents,
    extract_single_tar,
    github_headers,
    job_path,
    json_request,
    output,
    print_console_tail,
    require_sha,
    start_file_job,
    stop_build,
    wait_build,
    wait_workflow_job_artifact,
    zip_directory,
)


REPO = "Devsh-Graphics-Programming/Nabla"
BRANCH = "ptCLI"
BUILD_CONFIG = "Release"
SCENES = {"both", "public", "private"}
MAX_PACKAGE_BYTES = 200 * 1024 * 1024


def package_path(base, relative):
    path = Path(str(relative))
    if path.is_absolute() or ".." in path.parts:
        raise CiError("EX40 package manifest contains an unsafe path.")
    return base / path


def source_values(run, scene_set, publish, require_green=True):
    if run.get("head_branch") != BRANCH:
        raise CiError(f"Expected {BRANCH}, got {run.get('head_branch')}.")
    if run.get("name") != "Build" or (require_green and run.get("conclusion") != "success"):
        raise CiError(f"Build run {run.get('id')} is not green.")
    return {
        "run_id": run["id"],
        "build_config": BUILD_CONFIG,
        "scene_set": choice(scene_set, SCENES, "scene set"),
        "publish": bool_text(publish),
        "branch": BRANCH,
        "sha": require_sha(run.get("head_sha")),
        "run_attempt": run["run_attempt"],
        "workflow": run["name"],
    }


def resolve(args):
    if args.repository != REPO:
        raise CiError(f"Expected repository {REPO}.")
    headers = github_headers(args.github_token)
    if args.event_name == "push":
        if args.branch != BRANCH:
            raise CiError(f"Expected branch {BRANCH}.")
        run = wait_workflow_job_artifact(
            REPO,
            "build-nabla.yml",
            BRANCH,
            require_sha(args.sha),
            headers,
            "Build",
            "Nabla (windows-2022, msvc-17.13.6, Release)",
            "run-windows-*-msvc-Release-install",
        )
        values = source_values(run, "both", "true", require_green=False)
    else:
        run = json_request("GET", f"https://api.github.com/repos/{REPO}/actions/runs/{args.run_id}", headers)
        values = source_values(run, args.scene_set, args.publish)
    output(values)
    print(f"Resolved Build run {values['run_id']} on {BRANCH} at {values['sha']}.")


def package(args):
    artifact = Path(args.artifact_dir)
    extract_single_tar(artifact, "*-install.tar")
    install_root = artifact / "build-ct" / "install"
    root = install_root
    out = Path(args.package_root)
    shutil.rmtree(out, ignore_errors=True)
    prefix = Path()
    copy_contents(root / "runtime", out / prefix / "runtime")
    copy_contents(root / "exe" / "examples_tests" / "40_PathTracer" / "bin", out / prefix / "exe" / "examples_tests" / "40_PathTracer" / "bin")
    manifest = root / "EX40Runtime.json"
    if manifest.is_file():
        shutil.copy2(manifest, out / prefix / manifest.name)
        data = json.loads(manifest.read_text(encoding="utf-8"))
        if data.get("schema") != "devsh.nabla.example-runtime.v1" or data.get("component") != "EX40Runtime":
            raise CiError("EX40 package manifest is not supported.")
        exe = [package_path(out / prefix, data["executable"])]
        required = [package_path(out / prefix, data[name]) for name in ["nabla_runtime", "dxc_runtime", "report_template"]]
    else:
        exe = sorted(out.rglob("40_pathtracer*.exe"))
        required = []
    if not exe or not exe[0].is_file() or not list(out.rglob("Nabla*.dll")) or not list(out.rglob("dxcompiler.dll")):
        raise CiError("EX40 package is missing required runtime files.")
    if any(not path.exists() for path in required) or not (exe[0].parent / "report").is_dir():
        raise CiError("EX40 report template was not found in the package.")
    zip_directory(out, args.zip_path, MAX_PACKAGE_BYTES)
    output({"zip_path": args.zip_path})
    print(f"Prepared EX40 runtime package: {Path(args.zip_path).stat().st_size} bytes.")


def set_status(args, suite, state, target, description):
    commit_status(REPO, args.sha, args.github_token, f"jenkins/path-tracer-{suite}", state, target, description)


def start_suite(args, headers, suite):
    job = f"ci/ditt/real/ex40-{suite}"
    set_status(args, suite, "pending", f"https://github.com/{REPO}/actions/runs/{args.source_run_id}", f"Waiting for Jenkins {suite} path tracer run.")
    cancel_superseded(args.jenkins_url, headers, job, {"SOURCE_REPOSITORY": REPO, "SOURCE_BRANCH": BRANCH}, {
        "SOURCE_RUN_ID": args.source_run_id,
        "SOURCE_RUN_ATTEMPT": args.source_run_attempt,
    })
    fields = {
        "FAIL_ON_RENDER_FAILURE": "false",
        "PUBLISH": args.publish,
        "SOURCE_REPOSITORY": REPO,
        "SOURCE_BRANCH": BRANCH,
        "SOURCE_SHA": args.sha,
        "SOURCE_RUN_ID": args.source_run_id,
        "SOURCE_RUN_ATTEMPT": args.source_run_attempt,
        "SOURCE_WORKFLOW": args.source_workflow,
    }
    number = start_file_job(args.jenkins_url, headers, job, fields, "EX40_PACKAGE_FILE", args.package_path)
    build_url = f"{args.jenkins_url.rstrip('/')}/{job_path(job)}/{number}/"
    set_status(args, suite, "pending", build_url, f"Jenkins {suite} path tracer build #{number} is running.")
    print(f"Started Jenkins {job} #{number}: {build_url}")
    try:
        result = wait_build(args.jenkins_url, headers, job, number, int(args.jenkins_timeout_minutes) * 60)
    except CiError:
        stop_build(args.jenkins_url, headers, job, number)
        print_console_tail(args.jenkins_url, headers, job, number)
        set_status(args, suite, "failure", build_url, f"Jenkins {suite} path tracer did not complete.")
        raise
    if result not in {"SUCCESS", "UNSTABLE"}:
        print_console_tail(args.jenkins_url, headers, job, number)
        set_status(args, suite, "failure", build_url, f"Jenkins {suite} path tracer finished with {result}.")
        raise CiError(f"Jenkins {job} #{number} finished with {result}.")
    set_status(args, suite, "success", build_url, f"Jenkins {suite} path tracer {result.lower()}.")
    return job, number, result, build_url


def trigger(args):
    if args.repository != REPO or choice(args.branch, {BRANCH}, "branch") != BRANCH:
        raise CiError("Invalid source repository or branch.")
    args.sha = require_sha(args.sha)
    args.scene_set = choice(args.scene_set, SCENES, "scene set")
    args.publish = bool_text(args.publish)
    if not args.jenkins_url.startswith("https://") or not args.jenkins_user or not args.jenkins_token:
        raise CiError("Invalid Jenkins connection settings.")
    if not args.source_run_id.isdigit() or not args.source_run_attempt.isdigit():
        raise CiError("Invalid source run metadata.")
    if not args.jenkins_timeout_minutes.isdigit() or not 10 <= int(args.jenkins_timeout_minutes) <= 720:
        raise CiError("Invalid Jenkins timeout.")
    package_path = Path(args.package_path)
    if not package_path.is_file() or package_path.stat().st_size > MAX_PACKAGE_BYTES:
        raise CiError("Invalid package path or size.")
    headers = basic_headers(args.jenkins_user, args.jenkins_token)
    add_jenkins_crumb(args.jenkins_url, headers)
    suites = ["public", "private"] if args.scene_set == "both" else [args.scene_set]
    for result in [start_suite(args, headers, suite) for suite in suites]:
        print(f"{result[0]} #{result[1]} {result[2]} {result[3]}")


def parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    resolve_parser = sub.add_parser("resolve")
    add_args(resolve_parser, ["github-token", "repository", "event-name"])
    add_args(resolve_parser, ["branch", "sha", "run-id"], False, "")
    resolve_parser.add_argument("--scene-set", default="both")
    resolve_parser.add_argument("--publish", default="true")
    resolve_parser.set_defaults(func=resolve)
    package_parser = sub.add_parser("package")
    add_args(package_parser, ["artifact-dir", "package-root", "zip-path"])
    package_parser.set_defaults(func=package)
    trigger_parser = sub.add_parser("trigger")
    add_args(trigger_parser, ["jenkins-url", "jenkins-user", "jenkins-token", "github-token", "package-path", "repository", "branch", "sha", "source-run-id", "source-run-attempt", "source-workflow", "scene-set", "publish"])
    trigger_parser.add_argument("--jenkins-timeout-minutes", default="300")
    trigger_parser.set_defaults(func=trigger)
    return parser


if __name__ == "__main__":
    try:
        args = parser().parse_args()
        args.func(args)
    except CiError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
